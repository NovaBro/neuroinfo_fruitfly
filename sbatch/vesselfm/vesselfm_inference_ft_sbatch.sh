#!/bin/bash
#SBATCH --job-name=vesselfm-infer-ft
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|h200|l40s'
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/vesselfm/%x-%j.out
#SBATCH --error=sbatch/vesselfm/%x-%j.err

# Stage 7: inference with a finetuned raw state_dict on FISBe completely/test.
# Requires: metrics/vesselfm/checkpoints/fisbe_finetuned.pt (or set CKPT_PATH).
# Submit from repo root:
#   CKPT_PATH=.../fisbe_finetuned.pt sbatch sbatch/vesselfm/vesselfm_inference_ft_sbatch.sh

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/env/vesselfm.ext3" ]]; then
  echo "Error: expected overlay at ${REPO_ROOT}/env/vesselfm.ext3 (submit from repo root)" >&2
  exit 1
fi

IMAGE_PATH="${REPO_ROOT}/fisbe/vesselfm-fisbe/test"
MASK_PATH="${REPO_ROOT}/fisbe/vesselfm-fisbe/test_masks"
OUTPUT_FOLDER="${REPO_ROOT}/metrics/vesselfm/fisbe-completely-test-ft"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/metrics/vesselfm/checkpoints/fisbe_finetuned.pt}"
HF_HOME="${REPO_ROOT}/.cache/huggingface"

mkdir -p "${OUTPUT_FOLDER}" gpu_log "${HF_HOME}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Error: ckpt not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${IMAGE_PATH}" || ! -d "${MASK_PATH}" ]]; then
  echo "Error: missing image/mask dirs" >&2
  exit 1
fi

echo "IMAGE_PATH=${IMAGE_PATH}"
echo "MASK_PATH=${MASK_PATH}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

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
  > "gpu_log/gpu_vesselfm_infer_ft_${SLURM_JOB_ID}.csv" &
GPU_LOGGER_PID=$!

singularity exec --nv \
  --overlay env/vesselfm.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh
    conda activate vesselfm
    export HF_HOME=\"${HF_HOME}\"
    export PYTHONUNBUFFERED=1
    cd \"${REPO_ROOT}/vesselFM\"
    echo \"running vesselfm/seg/inference.py with finetuned ckpt\"
    python -u vesselfm/seg/inference.py \
      image_path=\"${IMAGE_PATH}\" \
      mask_path=\"${MASK_PATH}\" \
      output_folder=\"${OUTPUT_FOLDER}\" \
      ckpt_path=\"${CKPT_PATH}\" \
      device=cuda:0
  "
