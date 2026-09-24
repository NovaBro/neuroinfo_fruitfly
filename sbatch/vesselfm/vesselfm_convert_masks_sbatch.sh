#!/bin/bash
#SBATCH --job-name=vesselfm-convert-masks
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/vesselfm/%x-%j.out
#SBATCH --error=sbatch/vesselfm/%x-%j.err

# Stage 2: FISBe completely/{train,val,test} gt_fg_rm_5 → vesselfm-fisbe/{split}_masks/*.nii.gz
# Binary FG masks. Skips existing .nii.gz (no --overwrite).
# Submit from repo root:
#   sbatch sbatch/vesselfm/vesselfm_convert_masks_sbatch.sh

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/env/vesselfm.ext3" ]]; then
  echo "Error: expected overlay at ${REPO_ROOT}/env/vesselfm.ext3 (submit from repo root)" >&2
  exit 1
fi

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
SPLITS=(train val test)

echo "REPO_ROOT=${REPO_ROOT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

singularity exec \
  --overlay env/vesselfm.ext3:ro \
  "${SIF}" \
  /bin/bash -c "
    set -euo pipefail
    source /ext3/env.sh
    conda activate vesselfm
    export PYTHONUNBUFFERED=1
    cd \"${REPO_ROOT}\"
    for split in train val test; do
      echo \"=== converting masks: \${split} ===\"
      python -u vesselFM/convert_fisbe_zarr_to_nii.py \
        --input-dir \"fisbe/completely/\${split}\" \
        --output-dir \"fisbe/vesselfm-fisbe/\${split}_masks\" \
        --array-key volumes/gt_fg_rm_5 \
        --binarize
    done

    echo \"=== smoke: shape check one train stem ===\"
    python - <<'PY'
import SimpleITK as sitk
from pathlib import Path
img_dir = Path('fisbe/vesselfm-fisbe/train')
mask_dir = Path('fisbe/vesselfm-fisbe/train_masks')
stem = sorted(p.name for p in img_dir.glob('*.nii.gz'))[0]
img = sitk.ReadImage(str(img_dir / stem))
mask = sitk.ReadImage(str(mask_dir / stem))
print(stem, 'img', img.GetSize(), img.GetPixelIDTypeAsString(),
      'mask', mask.GetSize(), mask.GetPixelIDTypeAsString())
assert img.GetSize() == mask.GetSize(), (img.GetSize(), mask.GetSize())
print('shape OK')
PY
  "

echo "=== stem parity (images vs masks); empty = OK ==="
parity_fail=0
for split in "${SPLITS[@]}"; do
  echo "--- ${split} ---"
  if [[ ! -d "fisbe/vesselfm-fisbe/${split}" ]]; then
    echo "Error: missing image dir fisbe/vesselfm-fisbe/${split}" >&2
    parity_fail=1
    continue
  fi
  if [[ ! -d "fisbe/vesselfm-fisbe/${split}_masks" ]]; then
    echo "Error: missing mask dir fisbe/vesselfm-fisbe/${split}_masks" >&2
    parity_fail=1
    continue
  fi
  diff_out=$(comm -3 \
    <(ls "fisbe/vesselfm-fisbe/${split}/"*.nii.gz | xargs -n1 basename | sort) \
    <(ls "fisbe/vesselfm-fisbe/${split}_masks/"*.nii.gz | xargs -n1 basename | sort) \
    || true)
  if [[ -n "${diff_out}" ]]; then
    echo "${diff_out}"
    parity_fail=1
  else
    n=$(ls "fisbe/vesselfm-fisbe/${split}_masks/"*.nii.gz | wc -l)
    echo "OK (${n} masks)"
  fi
done

if [[ "${parity_fail}" -ne 0 ]]; then
  echo "Error: stem parity failed for one or more splits" >&2
  exit 1
fi
echo "Stage 2 convert masks: done"
