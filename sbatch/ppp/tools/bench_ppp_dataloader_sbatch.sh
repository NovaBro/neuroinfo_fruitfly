#!/bin/bash
# CPU-only benchmark of the PPP training data pipeline (no GPU billed).
# From repo root:  sbatch sbatch/ppp/tools/bench_ppp_dataloader_sbatch.sh
# Output: sbatch/ppp/logs/ppp_bench_dataloader-<jobid>.out
#
# Stages completely+partly tars to $SLURM_TMPDIR (same as ppp_job.sh STAGE_DATA=1),
# then runs bench_ppp_dataloader.py for:
#   A) default_l40s + basic_fmaps24_400k   (completely, bs18, elastic/overlay off)
#   B) default_l40s + basic-ppp-noenc-cp   (completely+partly, bs18)
#   C) B's throughput again with BLAS/OMP/blosc threads capped at 1 per worker
# Env: SKIP_A=1, B_CONFIGS (--config stack for B/C), BENCH_DATASETS (tars to stage), LIST_DATASETS (dirs to list for B/C),
#      B_ARGS (extra bench args for B, e.g. "--variants none,full"),
#      PREFETCH_TEST=1 (only: GPU-idle per step with/without ThreadedPrefetch)

#SBATCH --job-name=ppp_bench_dataloader
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=02:30:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

STAGE_ROOT="${SLURM_TMPDIR}/ppp_data"
for ds in ${BENCH_DATASETS:-completely partly}; do
  mkdir -p "${STAGE_ROOT}/${ds}/train"
  tar -xf "fisbe/${ds}/train_ppp.tar" -C "${STAGE_ROOT}/${ds}/train" &
done
wait
echo "staged: $(du -sh "${STAGE_ROOT}" | cut -f1)"

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
SETUP=flylight/setups/setup01
run() {
  singularity exec --overlay env/ppp.ext3:ro --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" "${SIF}" \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; $1"
}
# B/C config stack; override e.g. B_CONFIGS="--config ${SETUP}/original_default.toml --config ${SETUP}/our-noenc-cs.toml"
B_CONFIGS="${B_CONFIGS:---config ${SETUP}/default_l40s.toml --config ${SETUP}/basic-ppp-noenc-cp.toml}"
BENCH="python -u ../../sbatch/ppp/tools/bench_ppp_dataloader.py --fisbe-root ${STAGE_ROOT}"

if [[ "${PREFETCH_TEST:-0}" == 1 ]]; then
  echo "######## prefetch: basic_fmaps24_400k, simulated 3.5s GPU step"
  run "${BENCH} --config ${SETUP}/default_l40s.toml --config ${SETUP}/basic_fmaps24_400k.toml \
    --variants '' --prefetch 0 2 --simulate-step 3.5 --throughput-batches 20"
  exit 0
fi

[[ "${SKIP_A:-0}" == 1 ]] || { echo "######## A: basic_fmaps24_400k"
run "${BENCH} --config ${SETUP}/default_l40s.toml --config ${SETUP}/basic_fmaps24_400k.toml"; }

echo "######## B: basic-ppp-noenc-cp (completely+partly)"
run "${BENCH} ${B_CONFIGS} --list-datasets ${LIST_DATASETS:-completely partly} ${B_ARGS:-}"

echo "######## C: B throughput, threads capped"
run "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLOSC_NTHREADS=1; \
  ${BENCH} ${B_CONFIGS} --list-datasets ${LIST_DATASETS:-completely partly} --variants ''"
