#!/bin/bash
# Resample fisbe/himanshu/test/*.zarr (63x, 0.19x0.19x0.38 um) to the FISBe voxel size
# (0.44 um isotropic) -> fisbe/himanshu_r044/test/*.zarr for PPP inference. From repo root:
#
#   sbatch sbatch/ppp/prep/himanshu/resample_himanshu_sbatch.sh
#   LIMIT=1 sbatch sbatch/ppp/prep/himanshu/resample_himanshu_sbatch.sh      # smoke
#   TARGET_UM=0.44 DST=fisbe/himanshu_r044/test sbatch ...                     # defaults
#
# Sources are read-only; existing outputs are skipped (delete a .zarr by hand to redo it).
# Prerequisite: fisbe/himanshu/test from himanshu_sbatch.sh (OBJECTIVE=63x NAME=himanshu).
#
#SBATCH --job-name=ppp_resample_himanshu
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p sbatch/ppp/logs

SRC="${SRC:-fisbe/himanshu/test}"
DST="${DST:-fisbe/himanshu_r044/test}"
TARGET_UM="${TARGET_UM:-0.44}"
CLIPMAX="${CLIPMAX:-1500}"
RAW_DTYPE="${RAW_DTYPE:-float16}"
LIMIT="${LIMIT:-}"

LIMIT_ARGS=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS="--limit ${LIMIT}"
fi

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${REPO_ROOT}/env/ppp.ext3"

echo "============================================================"
echo "resample_himanshu_zarr"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SRC=${SRC}"
echo "DST=${DST}"
echo "TARGET_UM=${TARGET_UM}"
echo "CLIPMAX=${CLIPMAX}"
echo "RAW_DTYPE=${RAW_DTYPE}"
echo "LIMIT=${LIMIT:-all}"
echo "============================================================"

singularity exec \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -c "
    set -eo pipefail
    source /ext3/env.sh
    conda activate ppp
    python -u data_download_methods/resample_himanshu_zarr.py \
      --src \"${SRC}\" \
      --dst \"${DST}\" \
      --target-um ${TARGET_UM} \
      --clipmax ${CLIPMAX} \
      --raw-dtype ${RAW_DTYPE} \
      ${LIMIT_ARGS}
  "
