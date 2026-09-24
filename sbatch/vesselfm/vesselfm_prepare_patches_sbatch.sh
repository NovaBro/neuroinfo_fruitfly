#!/bin/bash
#SBATCH --job-name=vesselfm-prepare-patches
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=64g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/vesselfm/%x-%j.out
#SBATCH --error=sbatch/vesselfm/%x-%j.err

# Stage 4: FISBe completely → fisbe/vesselfm-fisbe-finetune patches.
# Smoke defaults (override for full):
#   PATCHES_TRAIN=4 PATCHES_VAL=2 PATCHES_TEST=2
# Full example:
#   PATCHES_TRAIN=32 PATCHES_VAL=8 PATCHES_TEST=8 sbatch ...
# Submit from repo root.

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/env/vesselfm.ext3" ]]; then
  echo "Error: expected overlay at ${REPO_ROOT}/env/vesselfm.ext3 (submit from repo root)" >&2
  exit 1
fi

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
PATCHES_TRAIN="${PATCHES_TRAIN:-4}"
PATCHES_VAL="${PATCHES_VAL:-2}"
PATCHES_TEST="${PATCHES_TEST:-2}"
MIN_FG="${MIN_FG:-0.001}"
CLEAN="${CLEAN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/fisbe/vesselfm-fisbe-finetune}"

CLEAN_FLAG=
if [[ "${CLEAN}" == "1" ]]; then
  CLEAN_FLAG="--clean"
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "PATCHES_TRAIN=${PATCHES_TRAIN} PATCHES_VAL=${PATCHES_VAL} PATCHES_TEST=${PATCHES_TEST}"
echo "MIN_FG=${MIN_FG} CLEAN=${CLEAN}"
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
    python -u vesselFM/prepare_fisbe_finetune_patches.py \
      --fisbe-root fisbe \
      --output-root \"${OUTPUT_ROOT}\" \
      --patches-per-volume-train ${PATCHES_TRAIN} \
      --patches-per-volume-val ${PATCHES_VAL} \
      --patches-per-volume-test ${PATCHES_TEST} \
      --min-fg-frac ${MIN_FG} \
      ${CLEAN_FLAG}

    echo '=== smoke load sample 000000 ==='
    python - <<'PY'
import numpy as np
from pathlib import Path
root = Path('${OUTPUT_ROOT}')
for split in ['train', 'val', 'test']:
    d = root / split
    n = len([p for p in d.iterdir() if p.is_dir()]) if d.exists() else 0
    print(split, 'n=', n)
p = root / 'train' / '000000'
img = np.load(p / 'img.npy')
mask = np.load(p / 'mask.npy')
print('train/000000', img.shape, img.dtype, mask.shape, mask.dtype, float(mask.mean()))
assert img.shape == mask.shape == (128, 128, 128)
assert float(mask.mean()) > 0
print('smoke OK')
PY
  "

echo "Stage 4 prepare patches: done"
