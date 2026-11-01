#!/bin/bash
# Download Gen1 MCFO full stacks for hemidrivers listed in hemidrivers.tsv.
# Uses host aws CLI (not Singularity). From repo root:
#
#   sbatch sbatch/data/s3_download_gen1_sbatch.sh
#   TSV=downloads/hemidrivers.tsv OUT_DIR=downloads \
#     sbatch sbatch/data/s3_download_gen1_sbatch.sh
#
# After meta job:
#   sbatch --dependency=afterok:<meta_jid> sbatch/data/s3_download_gen1_sbatch.sh
#
# Then (once av is installed in ppp overlay):
#   sbatch --dependency=afterok:<gen1_jid>:<av_jid> \
#     sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh
#
#SBATCH --job-name=s3_gen1
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=24:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/data/logs/%x-%j.out
#SBATCH --error=sbatch/data/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p sbatch/data/logs

TSV="${TSV:-downloads/hemidrivers.tsv}"
OUT_DIR="${OUT_DIR:-downloads}"

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v aws &>/dev/null; then
  echo "Error: aws CLI not found (expected ~/.local/bin/aws or on PATH)" >&2
  exit 1
fi
if [[ ! -f "${TSV}" ]]; then
  echo "Error: TSV not found: ${TSV}" >&2
  echo "Run sbatch/data/s3_download_meta_sbatch.sh first." >&2
  exit 1
fi

echo "============================================================"
echo "s3_download_fast --mode gen1"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "TSV=${TSV}"
echo "OUT_DIR=${OUT_DIR}"
echo "aws=$(command -v aws)"
echo "============================================================"

chmod +x data_download_methods/s3_download_fast.sh
./data_download_methods/s3_download_fast.sh --mode gen1 --yes "${TSV}" "${OUT_DIR}"

echo "Done. Gen1 MCFO under: ${OUT_DIR}/Gen1 MCFO/"
echo "Next: sbatch sbatch/ppp/env/ppp_env_install_av_sbatch.sh  # if av not yet installed"
echo "Then: sbatch --dependency=afterok:${SLURM_JOB_ID} sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh"
