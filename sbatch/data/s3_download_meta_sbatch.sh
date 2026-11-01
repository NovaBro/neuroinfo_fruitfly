#!/bin/bash
# Download Split-GAL4 *metadata.json only + write downloads/hemidrivers.tsv.
# Uses host aws CLI (not Singularity). From repo root:
#
#   sbatch sbatch/data/s3_download_meta_sbatch.sh
#   FILELIST=data_download_methods/filelist.txt OUT_DIR=downloads \
#     sbatch sbatch/data/s3_download_meta_sbatch.sh
#
# Next: sbatch --dependency=afterok:<this_jid> sbatch/data/s3_download_gen1_sbatch.sh
#
#SBATCH --job-name=s3_meta
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=02:00:00
#SBATCH --partition=cs,cpu_short
#SBATCH --output=sbatch/data/logs/%x-%j.out
#SBATCH --error=sbatch/data/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p sbatch/data/logs

FILELIST="${FILELIST:-data_download_methods/filelist.txt}"
OUT_DIR="${OUT_DIR:-downloads}"

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v aws &>/dev/null; then
  echo "Error: aws CLI not found (expected ~/.local/bin/aws or on PATH)" >&2
  exit 1
fi

echo "============================================================"
echo "s3_download_fast --mode meta"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "FILELIST=${FILELIST}"
echo "OUT_DIR=${OUT_DIR}"
echo "aws=$(command -v aws)"
echo "============================================================"

chmod +x data_download_methods/s3_download_fast.sh
./data_download_methods/s3_download_fast.sh --mode meta --yes "${FILELIST}" "${OUT_DIR}"

echo "Done. Hemidrivers TSV: ${OUT_DIR}/hemidrivers.tsv"
echo "Next: sbatch --dependency=afterok:${SLURM_JOB_ID} sbatch/data/s3_download_gen1_sbatch.sh"
