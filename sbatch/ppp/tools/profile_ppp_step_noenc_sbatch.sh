#!/bin/bash
# Re-profile one PPP *noenc* training step on the GPU with synthetic inputs (no data staging).
# Safe wrt running experiments: profile_ppp_step.py builds the model in a tempdir and never touches
# PatchPerPix/experiments/ppp_experiments/; outputs go to a separate dir (metrics/ppp/profile_noenc/),
# not the decoder profile's metrics/ppp/profile/.
# From repo root:  sbatch sbatch/ppp/tools/profile_ppp_step_noenc_sbatch.sh
# Env: CONFIGS, VARIANTS, EXTRA, OUT_DIR

#SBATCH --job-name=ppp_profile_noenc
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err
# Partition override: sbatch --partition=h200_public sbatch/ppp/tools/profile_ppp_step_noenc_sbatch.sh

set -uo pipefail
module purge
# Run from the checkout/worktree it was submitted from (submit from repo root).
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
OUT_DIR="${OUT_DIR:-metrics/ppp/profile_noenc/${SLURM_JOB_ID}}"
mkdir -p "${OUT_DIR}"

SETUP=flylight/setups/setup01
CONFIGS="${CONFIGS:---config ${SETUP}/configs/base/original_default.toml --config ${SETUP}/configs/presets/our-noenc-c1k_v1.toml}"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1 \
  > "${OUT_DIR}/gpu_usage_log.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

# Exactness check for training.fast_affinities before timing it (RUN_AFF_TEST=0 skips).
if [[ "${RUN_AFF_TEST:-1}" == "1" ]]; then
  singularity exec --nv --overlay env/ppp.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; \
      python -u ../../sbatch/ppp/tools/test_fast_affinities.py" \
    || echo "test_fast_affinities: FAILED (rc=$?)"
fi

# Numerical check for training.channels_last (same weights, NCDHW vs NDHWC; RUN_CL_TEST=0 skips).
if [[ "${RUN_CL_TEST:-1}" == "1" ]]; then
  singularity exec --nv --overlay env/ppp.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; \
      python -u ../../sbatch/ppp/tools/test_channels_last.py ${CONFIGS}" \
    || echo "test_channels_last: FAILED (rc=$?)"
fi

# Numerical check for torch.compile of UNet + heads + loss (RUN_COMPILE_TEST=1 enables).
if [[ "${RUN_COMPILE_TEST:-0}" == "1" ]]; then
  singularity exec --nv --overlay env/ppp.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; \
      python -u ../../sbatch/ppp/tools/test_compile_equiv.py ${CONFIGS}" \
    2>&1 | tee "${OUT_DIR}/test_compile_equiv.log"
  echo "test_compile_equiv: rc=${PIPESTATUS[0]}"
fi

# One process per variant (a CUDA fault kills the context). Trace dir is relative to PatchPerPix/experiments.
# Variants compose with "+": fast_aff, cl (channels_last), bf16, no_amp, bsN, inN (see profile_ppp_step.py).
#   H200: VARIANTS="base fast_aff fast_aff+cl fast_aff+bs36 fast_aff+cl+bs36 fast_aff+bs63 fast_aff+cl+bs63" \
#         sbatch --partition=h200_public sbatch/ppp/tools/profile_ppp_step_noenc_sbatch.sh
for v in ${VARIANTS:-base fast_aff cl fast_aff+cl}; do
  singularity exec --nv --overlay env/ppp.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; \
      python -u ../../sbatch/ppp/tools/profile_ppp_step.py ${CONFIGS} --variants ${v} \
        --trace-dir ../../${OUT_DIR} ${EXTRA:-}" 2>&1 | tee "${OUT_DIR}/variant_${v}.log"
  rc=${PIPESTATUS[0]}
  [[ ${rc} -eq 0 ]] || echo "variant ${v}: process failed (rc=${rc})"
done

# One table across all variant processes.
echo; echo "== combined summary (${SLURM_JOB_PARTITION:-?}: s per step; ms/smp = step / batch size)"
grep -h -A1 "^variant  .* sections$" "${OUT_DIR}"/variant_*.log | grep -v -e "^variant " -e "^--$" || true
grep -h "variant .* failed" "${OUT_DIR}"/variant_*.log || true
