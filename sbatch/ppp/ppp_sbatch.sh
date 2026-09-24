#!/bin/bash
#SBATCH --job-name=ppp
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384g
#SBATCH --constraint='h200'
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/ppp/ppp.out
#SBATCH --error=sbatch/ppp/ppp.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

GPU_LOGGER_PID=
cleanup() {
  if [[ -n "$GPU_LOGGER_PID" ]]; then
    kill "$GPU_LOGGER_PID" 2>/dev/null || true
    wait "$GPU_LOGGER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nohup nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 3 > gpu_usage_log_ppp.csv &
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

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Whole
# singularity exec --nv \
# --overlay env/ppp.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
# env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d train validate_checkpoints predict decode label evaluate --app flylight --root ppp_experiments --test-checkpoint last'

# DECODER:

# Train
# singularity exec --nv \
# --overlay env/ppp.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
# env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d train --app flylight --root ppp_experiments'

# Checkpoint
# singularity exec --nv \
# --overlay env/ppp.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
# env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d validate_checkpoints predict decode label evaluate -id ppp_experiments/flylight_setup01_260623_104736_902321 --app flylight --root ppp_experiments --test-checkpoint last'


# NO DECODER:
PPP_METRIC_ROOT="../../metrics/ppp"
# Train 3-4 Hours
singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u run_ppp.py --setup setup01 \
  --config flylight/setups/setup01/default_l40s.toml -d train \
  --train-data ${STAGE_ROOT}/train \
  --val-data ${STAGE_ROOT}/val \
  --app flylight --root ${PPP_METRIC_ROOT}"

# Checkpoint
  # Validatioin
# singularity exec --nv \
#   --overlay env/ppp.ext3:ro \
#   /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#   /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
#   env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_l40s.toml -d validate_checkpoints -id flylight_setup01_260714_113405_392086/ --app flylight --root ${PPP_METRIC_ROOT} --test-checkpoint last"
  # Prediction
# singularity exec --nv \
#   --overlay env/ppp.ext3:ro \
#   /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#   /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
#   env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_l40s.toml -d predict label evaluate -id flylight_setup01_260714_113405_392086/ --app flylight --root ${PPP_METRIC_ROOT} --test-checkpoint last"

# Current
# PatchPerPix/experiments/ppp_experiments/flylight_setup01_260630_164437_336076



