#!/bin/bash
#SBATCH --job-name=nnunet-train
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250g
#SBATCH --time=47:00:00
#SBATCH --output=sbatch/nnunet/logs/%x-%j.out
#SBATCH --error=sbatch/nnunet/logs/%x-%j.err

# nnU-Net 3d_fullres training of fold 0 (FISBe completely train 18 / val 5), ResEnc-L plans,
# 1000 epochs x 250 iterations (stock schedule) with trainer nnUNetTrainer_ckpt10 (= nnUNetTrainer,
# checkpoint_latest every 10 epochs). Always runs with --c, so the same script both starts and
# resumes; chain several 47 h jobs with --dependency=afterany:<prev> (see nnunet_submit_train.sh).
#
# When training has finished (checkpoint_final.pth), the job instead runs the post-training
# validation exports needed for decode tuning, keeping both checkpoints' probabilities:
#   fold_0/validation_final/<case>.npz   (checkpoint_final, written by --npz at the end of training)
#   fold_0/validation_best/<case>.npz    (checkpoint_best, via --val --val_best --npz)
# and restores fold_0/validation/ to the checkpoint_final version.
#
# Submit from repo root:  sbatch sbatch/nnunet/nnunet_train_sbatch.sh
# Env overrides: NPROC_DA (24), TRAINER, PLANS, FOLD, nnUNet_compile (unset = on).

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "${REPO_ROOT}"
source sbatch/nnunet/nnunet_common.sh
export nnUNet_n_proc_DA="${NPROC_DA:-24}"

TRAIN_CMD="nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -p ${PLANS} -tr ${TRAINER}"

if [[ -f "${RUN_DIR}/validation_final/summary.json" && -f "${RUN_DIR}/validation_best/summary.json" ]]; then
  echo "Training and both validation exports already done in ${RUN_DIR}; nothing to do."
  exit 0
fi

start_gpu_logger "train_${TRAINER}_f${FOLD}"
stage_preprocessed
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [[ ! -f "${RUN_DIR}/checkpoint_final.pth" ]]; then
  echo "=== train (--c) $(date) nnUNet_n_proc_DA=${nnUNet_n_proc_DA} ==="
  # --c: resume from checkpoint_latest if present (starts fresh with a warning otherwise).
  # --npz: the final validation also writes per-case probabilities (needed for instance decoding).
  nnunet_exec "${TRAIN_CMD} --npz --c"
fi

[[ -f "${RUN_DIR}/checkpoint_final.pth" ]] || { echo "Error: no checkpoint_final.pth after training" >&2; exit 1; }

if [[ ! -f "${RUN_DIR}/validation_final/summary.json" ]]; then
  if [[ ! -f "${RUN_DIR}/validation/summary.json" ]] || ! ls "${RUN_DIR}"/validation/*.npz >/dev/null 2>&1; then
    echo "=== validation (checkpoint_final, --npz) $(date) ==="
    nnunet_exec "${TRAIN_CMD} --val --npz"
  fi
  rsync -a --delete "${RUN_DIR}/validation/" "${RUN_DIR}/validation_final/"
fi

if [[ ! -f "${RUN_DIR}/validation_best/summary.json" ]]; then
  echo "=== validation (checkpoint_best, --npz) $(date) ==="
  # --val_best writes into the same validation/ folder; copy it out, then restore the final one.
  nnunet_exec "${TRAIN_CMD} --val --val_best --npz"
  rsync -a --delete "${RUN_DIR}/validation/" "${RUN_DIR}/validation_best/"
  rsync -a --delete "${RUN_DIR}/validation_final/" "${RUN_DIR}/validation/"
fi

ls "${RUN_DIR}"
echo "=== done $(date) ==="
