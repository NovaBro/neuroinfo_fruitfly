#!/bin/bash
# Shared settings for sbatch/nnunet/*_sbatch.sh. Source AFTER `cd "${REPO_ROOT}"`:
#   REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"; cd "${REPO_ROOT}"
#   source sbatch/nnunet/nnunet_common.sh
# Everything here is overridable from the environment (sbatch --export=ALL,VAR=...).

SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
NNUNET_OVERLAY="${NNUNET_OVERLAY:-${REPO_ROOT}/env/nnunet.ext3}"     # conda env nnunet (py3.11, torch 2.8.0, nnunetv2 2.8.1)
EVAL_OVERLAY="${EVAL_OVERLAY:-${REPO_ROOT}/env/evaluate.ext3}"       # conda env evalinstseg (decode + FISBe scoring)
NNUNET_ENV_PREFIX=/ext3/miniforge3/envs/nnunet

# ---- dataset / model identity ----
DATASET_ID="${DATASET_ID:-501}"
DATASET_NAME="${DATASET_NAME:-Dataset501_FISBeCompletely}"
PLANNER="${PLANNER:-nnUNetPlannerResEncL}"
PLANS="${PLANS:-nnUNetResEncUNetLPlans}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer_ckpt10}"    # stock nnUNetTrainer + checkpoint_latest every 10 epochs
FOLD="${FOLD:-0}"                             # fold 0 = FISBe completely train(18)/val(5), splits_final.json

# ---- nnU-Net paths (all gitignored) ----
export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/fisbe/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/fisbe/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/metrics/nnunet/nnUNet_results}"
export nnUNet_extTrainer="${nnUNet_extTrainer:-${REPO_ROOT}/nnunet_work_folder/trainers}"
RUN_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

# ---- threads / caches ----
# nnU-Net's CLI does NOT set these (only run_training.py's __main__ does); one thread per DA worker.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONUNBUFFERED=1
NN_TMP="${SLURM_TMPDIR:-/tmp/nnunet_${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "${NN_TMP}"
# Keep triton/inductor/matplotlib caches off /home (30k-file quota).
export TRITON_CACHE_DIR="${NN_TMP}/triton" TORCHINDUCTOR_CACHE_DIR="${NN_TMP}/inductor" MPLCONFIGDIR="${NN_TMP}/mpl"

# nnunet_exec "<shell command>": run inside the nnunet overlay (GPU passthrough if present).
NV_FLAG=()
command -v nvidia-smi >/dev/null 2>&1 && NV_FLAG=(--nv)
nnunet_exec() {
  singularity exec "${NV_FLAG[@]}" --overlay "${NNUNET_OVERLAY}:ro" --bind "${NN_TMP}" "${SIF}" /bin/bash -c \
    "source /ext3/env.sh; conda activate ${NNUNET_ENV_PREFIX}; set -eo pipefail; $1"
}

# eval_exec "<shell command>": run inside the evaluate overlay (evalinstseg, zarr 2, skimage, scipy).
eval_exec() {
  singularity exec --overlay "${EVAL_OVERLAY}:ro" --bind "${NN_TMP}" "${SIF}" /bin/bash -c \
    "source /ext3/env.sh; conda activate evalinstseg; set -eo pipefail; $1"
}

# start_gpu_logger <tag>: background nvidia-smi CSV (public GPU partitions cancel low-util jobs; keep it).
GPU_LOGGER_PID=
start_gpu_logger() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  mkdir -p gpu_log/nnunet
  local q="timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
  nvidia-smi --query-gpu="${q}" --format=csv -l 3 > "gpu_log/nnunet/gpu_usage_log_${1}_${SLURM_JOB_ID:-local}.csv" &
  GPU_LOGGER_PID=$!
  trap 'kill ${GPU_LOGGER_PID} 2>/dev/null || true' EXIT
}

# stage_preprocessed: copy this dataset's preprocessed folder to node-local SSD and point nnU-Net at it.
stage_preprocessed() {
  local src="${nnUNet_preprocessed}/${DATASET_NAME}"
  [[ -d "${src}" ]] || { echo "Error: ${src} missing (run nnunet_prep_sbatch.sh)" >&2; exit 1; }
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    mkdir -p "${SLURM_TMPDIR}/nnUNet_preprocessed"
    local t0=$SECONDS
    rsync -a "${src}" "${SLURM_TMPDIR}/nnUNet_preprocessed/"
    export nnUNet_preprocessed="${SLURM_TMPDIR}/nnUNet_preprocessed"
    echo "staged ${src} -> ${nnUNet_preprocessed} in $((SECONDS - t0)) s ($(du -sh "${nnUNet_preprocessed}" | cut -f1))"
  fi
}

echo "[nnunet_common] job=${SLURM_JOB_ID:-local} host=$(hostname) dataset=${DATASET_NAME} plans=${PLANS} trainer=${TRAINER} fold=${FOLD}"
