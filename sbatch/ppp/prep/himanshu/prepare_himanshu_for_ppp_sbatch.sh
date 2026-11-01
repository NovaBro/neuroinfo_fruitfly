#!/bin/bash
# Convert Gen1 MCFO aligned .h5j stacks → fisbe/himanshu/test/*.zarr for PPP
# inference. From repo root:
#
#   sbatch sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh
#   LIMIT=1 sbatch sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh   # smoke
#   OVERWRITE=1 sbatch sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh
#
# Prerequisites:
#   1. downloads/hemidrivers.tsv + Gen1 MCFO stacks
#      (sbatch/data/s3_download_{meta,gen1}_sbatch.sh)
#   2. PyAV in ppp overlay (sbatch/ppp/env/ppp_env_install_av_sbatch.sh)
#
# Example chain after gen1 + av job ids:
#   sbatch --dependency=afterok:<gen1_jid>:<av_jid> \
#     sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh
#
#SBATCH --job-name=ppp_prepare_himanshu
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p sbatch/ppp/logs

TSV="${TSV:-downloads/hemidrivers.tsv}"
DOWNLOADS="${DOWNLOADS:-downloads/Gen1 MCFO}"
OUT="${OUT:-fisbe/himanshu/test}"
CLIPMAX="${CLIPMAX:-1500}"
OVERWRITE="${OVERWRITE:-0}"
LIMIT="${LIMIT:-}"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

LIMIT_ARGS=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS="--limit ${LIMIT}"
fi

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${REPO_ROOT}/env/ppp.ext3"

echo "============================================================"
echo "prepare_himanshu_for_ppp"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "TSV=${TSV}"
echo "DOWNLOADS=${DOWNLOADS}"
echo "OUT=${OUT}"
echo "CLIPMAX=${CLIPMAX}"
echo "OVERWRITE=${OVERWRITE}"
echo "LIMIT=${LIMIT:-all}"
echo "============================================================"

singularity exec \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -c "
    set -eo pipefail
    source /ext3/env.sh
    conda activate ppp
    if ! python -c 'import av' 2>/dev/null; then
      echo \"Error: PyAV (av) not installed in ppp overlay.\" >&2
      echo \"Run: sbatch sbatch/ppp/env/ppp_env_install_av_sbatch.sh\" >&2
      echo \"(Do not run other ppp jobs while that install holds :rw.)\" >&2
      exit 1
    fi
    python -c 'import av; print(\"av\", av.__version__)'
    python -u data_download_methods/prepare_himanshu_for_ppp.py \
      --tsv \"${TSV}\" \
      --downloads \"${DOWNLOADS}\" \
      --out \"${OUT}\" \
      --clipmax ${CLIPMAX} \
      ${OVERWRITE_FLAG} \
      ${LIMIT_ARGS}
  "
