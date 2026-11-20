#!/bin/bash
# Himanshu split-line MCFO -> PPP inference zarrs, in one job:
#   1. download (host python3 + aws CLI): data_download_methods/download_mcfo.py
#      -> downloads/mcfo_manifest_${OBJECTIVE}.tsv + .h5j stacks under downloads/<release>/<SS>/
#   2. convert (ppp container): data_download_methods/prepare_himanshu_for_ppp.py
#      -> fisbe/${NAME}/test/*.zarr at the native voxel size (no resampling)
#
# From repo root:
#   DRY_RUN=1 sbatch sbatch/ppp/prep/himanshu/himanshu_sbatch.sh   # manifest only, no stacks/zarrs
#   LIMIT=1   sbatch sbatch/ppp/prep/himanshu/himanshu_sbatch.sh   # smoke: convert 1 stack
#   sbatch sbatch/ppp/prep/himanshu/himanshu_sbatch.sh             # 20x -> fisbe/himanshu_20x/test
#   OBJECTIVE=63x NAME=himanshu sbatch ...                         # rebuild the 63x set
#
# Env: OBJECTIVE (20x) NAME (himanshu_${OBJECTIVE}) LINES (data_download_methods/himanshu_lines.tsv)
#      STACK (unaligned|gendered|unisex) STAGE (all|download|prep) DRY_RUN LIMIT
#      CLEAN=1 (delete existing zarrs in fisbe/${NAME}/test first) OVERWRITE=1 RAW_DTYPE (float16) CLIPMAX (1500)
# Needs PyAV in the ppp overlay (sbatch/ppp/env/ppp_env_install_av_sbatch.sh, once).
# New dataset names must be in KNOWN_DATASETS (PatchPerPix/experiments/run_ppp.py) for inference.
# Optional later step: resample_himanshu_sbatch.sh (e.g. to the FISBe 0.44 um grid).
#
#SBATCH --job-name=himanshu_prep
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p sbatch/ppp/logs

OBJECTIVE="${OBJECTIVE:-20x}"
NAME="${NAME:-himanshu_${OBJECTIVE}}"
LINES="${LINES:-data_download_methods/himanshu_lines.tsv}"
STACK="${STACK:-unaligned}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"
CLEAN="${CLEAN:-0}"
OVERWRITE="${OVERWRITE:-0}"
RAW_DTYPE="${RAW_DTYPE:-float16}"
CLIPMAX="${CLIPMAX:-1500}"
MANIFEST="downloads/mcfo_manifest_${OBJECTIVE}.tsv"
OUT="fisbe/${NAME}/test"

echo "============================================================"
echo "himanshu prep  SLURM_JOB_ID=${SLURM_JOB_ID:-unset}  host=$(hostname)"
echo "OBJECTIVE=${OBJECTIVE} STACK=${STACK} LINES=${LINES} STAGE=${STAGE} DRY_RUN=${DRY_RUN}"
echo "MANIFEST=${MANIFEST} OUT=${OUT} LIMIT=${LIMIT:-all} CLEAN=${CLEAN} OVERWRITE=${OVERWRITE} RAW_DTYPE=${RAW_DTYPE}"
echo "============================================================"

if [[ "${STAGE}" == "all" || "${STAGE}" == "download" || "${DRY_RUN}" == "1" ]]; then
  export PATH="${HOME}/.local/bin:/scratch/wmz2007/.local/bin:${PATH}"
  command -v aws >/dev/null || { echo "Error: aws CLI not on PATH" >&2; exit 1; }
  DL_ARGS=(--lines "${LINES}" --out downloads --manifest "${MANIFEST}"
           --objective "${OBJECTIVE}" --stack "${STACK}")
  [[ "${DRY_RUN}" == "1" ]] && DL_ARGS+=(--dry-run)
  python3 -u data_download_methods/download_mcfo.py "${DL_ARGS[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: manifest written, nothing downloaded or converted."
    exit 0
  fi
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "prep" ]]; then
  PREP_ARGS="--manifest '${MANIFEST}' --downloads downloads --out '${OUT}' --clipmax ${CLIPMAX} --raw-dtype ${RAW_DTYPE}"
  [[ "${CLEAN}" == "1" ]] && PREP_ARGS+=" --clean"
  [[ "${OVERWRITE}" == "1" ]] && PREP_ARGS+=" --overwrite"
  [[ -n "${LIMIT}" ]] && PREP_ARGS+=" --limit ${LIMIT}"
  singularity exec --overlay "${REPO_ROOT}/env/ppp.ext3:ro" \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
      set -eo pipefail
      source /ext3/env.sh
      conda activate ppp
      python -c 'import av; print(\"av\", av.__version__)' || {
        echo 'Error: PyAV missing in ppp overlay; run sbatch/ppp/env/ppp_env_install_av_sbatch.sh' >&2; exit 1; }
      python -u data_download_methods/prepare_himanshu_for_ppp.py ${PREP_ARGS}
    "
fi
