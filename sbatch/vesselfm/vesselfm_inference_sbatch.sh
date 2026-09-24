#!/bin/bash
#SBATCH --job-name=vesselfm-infer-test
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|h200'
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/vesselfm/%x-%j.out
#SBATCH --error=sbatch/vesselfm/%x-%j.err

# Zero-shot vesselFM inference on FISBe completely/test (NIfTI) with GT masks.
# Submit from repo root (after env/vesselfm.ext3 install job has finished):
#   sbatch sbatch/vesselfm/vesselfm_inference_sbatch.sh

set -euo pipefail
module purge

# Prefer submit directory (repo root). BASH_SOURCE can point at a Slurm spool copy.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/env/vesselfm.ext3" ]]; then
  echo "Error: expected overlay at ${REPO_ROOT}/env/vesselfm.ext3 (submit from repo root)" >&2
  exit 1
fi

IMAGE_PATH="${REPO_ROOT}/fisbe/vesselfm-fisbe/test"
MASK_PATH="${REPO_ROOT}/fisbe/vesselfm-fisbe/test_masks"
OUTPUT_FOLDER="${REPO_ROOT}/metrics/vesselfm/fisbe-completely-test"
HF_HOME="${REPO_ROOT}/.cache/huggingface"

mkdir -p "${OUTPUT_FOLDER}" gpu_log "${HF_HOME}"

if [[ ! -d "${IMAGE_PATH}" ]]; then
  echo "Error: image_path not found: ${IMAGE_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MASK_PATH}" ]]; then
  echo "Error: mask_path not found: ${MASK_PATH}" >&2
  exit 1
fi

echo "IMAGE_PATH=${IMAGE_PATH}"
echo "MASK_PATH=${MASK_PATH}"
echo "OUTPUT_FOLDER=${OUTPUT_FOLDER}"
echo "HF_HOME=${HF_HOME}"
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
  > "gpu_log/gpu_vesselfm_infer_test_${SLURM_JOB_ID}.csv" &
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
    echo \"running vesselfm/seg/inference.py\"
    python -u vesselfm/seg/inference.py \
      image_path=\"${IMAGE_PATH}\" \
      mask_path=\"${MASK_PATH}\" \
      output_folder=\"${OUTPUT_FOLDER}\" \
      device=cuda:0
  "
