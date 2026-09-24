#!/bin/bash
#SBATCH --job-name=vesselfm-finetune-smoke
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|h200|l40s'
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/vesselfm/%x-%j.out
#SBATCH --error=sbatch/vesselfm/%x-%j.err

# Stage 6 smoke: short finetune on FISBe patches.
# Defaults are smoke-sized; override for fuller runs:
#   NUM_SHOTS=72 MAX_STEPS=1200 VAL_CHECK=200 RUN_NAME=fisbe_full \
#     sbatch sbatch/vesselfm/vesselfm_finetune_sbatch.sh
# Submit from repo root.

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/env/vesselfm.ext3" ]]; then
  echo "Error: expected overlay at ${REPO_ROOT}/env/vesselfm.ext3 (submit from repo root)" >&2
  exit 1
fi

NUM_SHOTS="${NUM_SHOTS:-8}"
MAX_STEPS="${MAX_STEPS:-50}"
VAL_CHECK="${VAL_CHECK:-25}"
RUN_NAME="${RUN_NAME:-fisbe_smoke}"
CKPT_IN="${REPO_ROOT}/metrics/vesselfm/checkpoints/vesselFM_all_lightning.ckpt"
CKPT_OUT="${REPO_ROOT}/metrics/vesselfm/finetune_ckpts"
HF_HOME="${REPO_ROOT}/.cache/huggingface"

mkdir -p "${CKPT_OUT}" gpu_log "${HF_HOME}"

if [[ ! -f "${CKPT_IN}" ]]; then
  echo "Error: missing wrapped ckpt: ${CKPT_IN}" >&2
  exit 1
fi
if [[ ! -d "${REPO_ROOT}/fisbe/vesselfm-fisbe-finetune/train" ]]; then
  echo "Error: missing finetune train patches" >&2
  exit 1
fi

echo "NUM_SHOTS=${NUM_SHOTS} MAX_STEPS=${MAX_STEPS} VAL_CHECK=${VAL_CHECK} RUN_NAME=${RUN_NAME}"
echo "CKPT_IN=${CKPT_IN}"
echo "CKPT_OUT=${CKPT_OUT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

# >>>> GPU Tracking >>>>
GPU_LOGGER_PID=
cleanup() {
  if [[ -n "${GPU_LOGGER_PID}" ]]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nohup nvidia-smi \
  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 3 \
  > "gpu_log/gpu_vesselfm_finetune_${SLURM_JOB_ID}.csv" &
GPU_LOGGER_PID=$!
# <<<< GPU Tracking <<<<

singularity exec --nv \
  --overlay env/vesselfm.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    conda activate vesselfm
    export HF_HOME=\"${HF_HOME}\"
    export PYTHONUNBUFFERED=1
    cd \"${REPO_ROOT}/vesselFM\"
    echo \"running vesselfm/seg/finetune.py (smoke)\"
    python -u vesselfm/seg/finetune.py \
      data=eval_fisbe \
      num_shots=${NUM_SHOTS} \
      run_name=${RUN_NAME} \
      path_to_chkpt=\"${CKPT_IN}\" \
      chkpt_folder=\"${CKPT_OUT}\" \
      offline=True \
      devices=[0] \
      dataloader.num_workers=4 \
      trainer.lightning_trainer.max_steps=${MAX_STEPS} \
      trainer.lightning_trainer.val_check_interval=${VAL_CHECK}
  "
