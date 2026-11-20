#!/bin/bash
#SBATCH --job-name=nnunet-predict
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250g
#SBATCH --time=06:00:00
#SBATCH --output=sbatch/nnunet/logs/%x-%j.out
#SBATCH --error=sbatch/nnunet/logs/%x-%j.err

# nnU-Net inference on the FISBe completely test images (imagesTs) with fold 0, saving region
# probabilities (.npz) for instance decoding. Output:
#   metrics/nnunet/${EXP}/test/nnunet_out_${CKPT}/<case>.{tif,json,npz,pkl}
# Submit from repo root:
#   CKPT=final sbatch --export=ALL sbatch/nnunet/nnunet_predict_sbatch.sh      (or CKPT=best)
# Env: CKPT (final|best), EXP (fisbe501_resencL_f0), INPUT_DIR (default imagesTs), TTA=0 to disable mirroring.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "${REPO_ROOT}"
source sbatch/nnunet/nnunet_common.sh

CKPT="${CKPT:-final}"
EXP="${EXP:-fisbe501_resencL_f0}"
INPUT_DIR="${INPUT_DIR:-${nnUNet_raw}/${DATASET_NAME}/imagesTs}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/metrics/nnunet/${EXP}/test/nnunet_out_${CKPT}}"
[[ -f "${RUN_DIR}/checkpoint_${CKPT}.pth" ]] || { echo "Error: ${RUN_DIR}/checkpoint_${CKPT}.pth missing" >&2; exit 1; }

# Pre-flight: Tiff3DIO falls back to spacing (1,1,1) without a <case>.json sidecar -> 2.27x resampling.
n_img=$(ls "${INPUT_DIR}"/*_0000.tif | wc -l)
n_json=$(ls "${INPUT_DIR}"/*.json | wc -l)
[[ "${n_img}" -gt 0 && "${n_img}" -eq "${n_json}" ]] || { echo "Error: ${n_img} cases but ${n_json} spacing sidecars in ${INPUT_DIR}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
start_gpu_logger "predict_${CKPT}"
TTA_FLAG=""
[[ "${TTA:-1}" == 0 ]] && TTA_FLAG="--disable_tta"

echo "=== predict ${n_img} cases ckpt=${CKPT} -> ${OUT_DIR} $(date) ==="
nnunet_exec "nnUNetv2_predict -i '${INPUT_DIR}' -o '${OUT_DIR}' -d ${DATASET_ID} -c ${CONFIG} -p ${PLANS} \
  -tr ${TRAINER} -f ${FOLD} -chk checkpoint_${CKPT}.pth --save_probabilities ${TTA_FLAG} -npp 4 -nps 4"
ls -la "${OUT_DIR}"
n_npz=$(ls "${OUT_DIR}"/*.npz | wc -l)
[[ "${n_npz}" -eq "${n_img}" ]] || { echo "Error: ${n_npz} npz for ${n_img} cases" >&2; exit 1; }
echo "=== done $(date) ==="
