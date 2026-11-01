#!/bin/bash
# Remove all prepared himanshu zarrs under fisbe/himanshu/test/ (many small
# chunk files — run on a compute node, not a login shell).
#
# From repo root:
#   sbatch sbatch/ppp/prep/himanshu/delete_himanshu_zarrs_sbatch.sh
#   DEL_JID=$(sbatch --parsable sbatch/ppp/prep/himanshu/delete_himanshu_zarrs_sbatch.sh)
#   LIMIT=10 sbatch --dependency=afterok:${DEL_JID} \
#     sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh
#
#SBATCH --job-name=ppp_delete_himanshu
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=4:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p sbatch/ppp/logs

OUT="${OUT:-fisbe/himanshu/test}"

echo "============================================================"
echo "delete_himanshu_zarrs"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "OUT=${OUT}"
echo "============================================================"

if [[ ! -d "${OUT}" ]]; then
  echo "Nothing to delete: ${OUT} does not exist"
  exit 0
fi

echo "Contents before delete:"
ls -la "${OUT}" | head -50 || true
n_zarr=$(find "${OUT}" -maxdepth 1 -type d -name '*.zarr' | wc -l)
echo "zarr dirs: ${n_zarr}"

# Remove zarr trees and any sidecar files (e.g. manifest.csv); keep the
# parent directory so the prepare job can write into the same path.
find "${OUT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

echo "Contents after delete:"
ls -la "${OUT}" || true
echo "done"
