#!/bin/bash
#SBATCH --job-name=vesselfm-convert-images
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/vesselfm/%x-%j.out
#SBATCH --error=sbatch/vesselfm/%x-%j.err

# Stage 1: FISBe completely/{train,val,test} *.zarr → vesselfm-fisbe/{split}/*.nii.gz
# Channel-max of volumes/raw. Skips existing outputs (no --overwrite).
# Submit from repo root:
#   sbatch sbatch/vesselfm/vesselfm_convert_images_sbatch.sh

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
      echo \"=== converting images: \${split} ===\"
      python -u vesselFM/convert_fisbe_zarr_to_nii.py \
        --input-dir \"fisbe/completely/\${split}\" \
        --output-dir \"fisbe/vesselfm-fisbe/\${split}\"
    done
  "

echo "=== stem parity (zarr vs nii); empty = OK ==="
parity_fail=0
for split in "${SPLITS[@]}"; do
  echo "--- ${split} ---"
  diff_out=$(comm -3 \
    <(ls -d "fisbe/completely/${split}/"*.zarr | xargs -n1 basename | sed 's/\.zarr$//' | sort) \
    <(ls "fisbe/vesselfm-fisbe/${split}/"*.nii.gz | xargs -n1 basename | sed 's/\.nii\.gz$//' | sort) \
    || true)
  if [[ -n "${diff_out}" ]]; then
    echo "${diff_out}"
    parity_fail=1
  else
    n=$(ls "fisbe/vesselfm-fisbe/${split}/"*.nii.gz | wc -l)
    echo "OK (${n} volumes)"
  fi
done

if [[ "${parity_fail}" -ne 0 ]]; then
  echo "Error: stem parity failed for one or more splits" >&2
  exit 1
fi
echo "Stage 1 convert images: done"
