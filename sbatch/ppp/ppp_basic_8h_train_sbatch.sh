#!/bin/bash
# PatchPerPix basic train only (stable exp id ppp_basic_8h).
# Prefer: bash sbatch/ppp/ppp_basic_8h_chain.sh
# Or alone: sbatch sbatch/ppp/ppp_basic_8h_train_sbatch.sh
#SBATCH --job-name=ppp_basic_train
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192g
#SBATCH --time=08:00:00
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/ppp_basic_train.out
#SBATCH --error=sbatch/ppp/ppp_basic_train.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

GPU_LOGGER_PID=
cleanup() {
  if [[ -n "${GPU_LOGGER_PID}" ]]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nvidia-smi \
  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 3 \
  > gpu_usage_log_ppp_basic_train.csv &
GPU_LOGGER_PID=$!

# Stage train/val Zarrs to node-local SSD so gunpowder PreCache can keep the GPU fed.
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "Error: SLURM_TMPDIR is unset; cannot stage training data" >&2
  exit 1
fi
SRC_ROOT="fisbe/completely"
STAGE_ROOT="${SLURM_TMPDIR}/ppp_data"
echo "Staging train/val -> ${STAGE_ROOT}"
df -h "${SLURM_TMPDIR}"
STAGE_T0=$(date +%s)
mkdir -p "${STAGE_ROOT}"
rsync -a --delete "${SRC_ROOT}/train/" "${STAGE_ROOT}/train/"
rsync -a --delete "${SRC_ROOT}/val/" "${STAGE_ROOT}/val/"
STAGE_T1=$(date +%s)
echo "Staging finished in $((STAGE_T1 - STAGE_T0))s"
df -h "${SLURM_TMPDIR}"

singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_8h.toml \
    --train-data ${STAGE_ROOT}/train \
    --val-data ${STAGE_ROOT}/val \
    -d train \
    --app flylight --root ../../metrics/ppp -id ppp_basic_8h"
