#!/bin/bash
# One-step GPU profile of PPP training with synthetic inputs (no data staging).
# From repo root:  sbatch sbatch/ppp/tools/profile_ppp_step_sbatch.sh
# Output: sbatch/ppp/logs/ppp_profile_step-<jobid>.out, trace in metrics/ppp/profile/
# Env: CONFIGS (override the --config stack), VARIANTS (e.g. "base no_amp"), EXTRA (extra script args)

#SBATCH --job-name=ppp_profile_step
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -uo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p metrics/ppp/profile

SETUP=flylight/setups/setup01
CONFIGS="${CONFIGS:---config ${SETUP}/original_default_train_code.toml --config ${SETUP}/our-decoder-c1k_v1.toml}"

# GPU-utilization logger (public partitions cancel low-util jobs; also audits this run)
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1 \
  > metrics/ppp/profile/gpu_usage_log.csv &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

# One process per variant: a CUDA fault (e.g. illegal memory access in fp32 at bs30) kills the
# context, so variants must not share a process.
for v in ${VARIANTS:-base no_cudnn_benchmark channels_last bs16 in112 no_amp}; do
  singularity exec --nv --overlay env/ppp.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; \
      python -u ../../sbatch/ppp/tools/profile_ppp_step.py ${CONFIGS} --variants ${v} ${EXTRA:-}" \
    || echo "variant ${v}: process failed (rc=$?)"
done
