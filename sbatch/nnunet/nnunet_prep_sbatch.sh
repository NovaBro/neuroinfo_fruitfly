#!/bin/bash
#SBATCH --job-name=nnunet-prep
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=16
#SBATCH --mem=160g
#SBATCH --time=08:00:00
#SBATCH --output=sbatch/nnunet/logs/%x-%j.out
#SBATCH --error=sbatch/nnunet/logs/%x-%j.err

# CPU job: FISBe completely zarr -> nnUNet_raw/Dataset501_FISBeCompletely (Tiff3DIO, region labels),
# then nnU-Net fingerprint + ResEnc-L planning + 3d_fullres preprocessing.
# Re-running is safe: converted cases are skipped (OVERWRITE=1 to redo), planning/preprocessing redone.
# Submit from repo root:
#   sbatch sbatch/nnunet/nnunet_prep_sbatch.sh
# Env overrides: CONVERT_WORKERS (8), PREP_NP (6), SKIP_CONVERT=1, OVERWRITE=1.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "${REPO_ROOT}"
source sbatch/nnunet/nnunet_common.sh

CONVERT_WORKERS="${CONVERT_WORKERS:-8}"
PREP_NP="${PREP_NP:-6}"
RAW_DS="${nnUNet_raw}/${DATASET_NAME}"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

if [[ "${SKIP_CONVERT:-0}" != 1 ]]; then
  echo "=== convert $(date) ==="
  nnunet_exec "python -u nnunet_work_folder/fisbe_to_nnunet.py --fisbe fisbe/completely --out '${RAW_DS}' \
    --workers ${CONVERT_WORKERS} ${OVERWRITE:+--overwrite}"
  du -sh "${RAW_DS}"
fi

echo "=== plan + preprocess $(date) ==="
# -c 3d_fullres only: 3d_lowres / cascade would downsample 1-3 voxel neurites away.
nnunet_exec "nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -pl ${PLANNER} -c ${CONFIG} \
  --verify_dataset_integrity -npfp 8 -np ${PREP_NP}"

PP="${nnUNet_preprocessed}/${DATASET_NAME}"
ls -la "${PP}"
du -sh "${PP}"
cmp "${RAW_DS}/splits_final.json" "${PP}/splits_final.json" && echo "splits_final.json copied (fold 0 = FISBe train/val)"
nnunet_exec "python - <<'PY'
import json
p = json.load(open('${PP}/${PLANS}.json'))
c = p['configurations']['${CONFIG}']
for k in ('patch_size', 'batch_size', 'spacing', 'median_image_size_in_voxels', 'normalization_schemes',
          'use_mask_for_norm'):
    print(k, '=', c.get(k))
a = c['architecture']['arch_kwargs']
print('n_stages =', a['n_stages'], 'features =', a['features_per_stage'], 'strides =', a['strides'])
print('transpose_forward =', p['transpose_forward'])
PY"
echo "=== done $(date) ==="
