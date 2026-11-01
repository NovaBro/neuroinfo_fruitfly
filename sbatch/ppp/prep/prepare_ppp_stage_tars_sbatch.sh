#!/bin/bash
# Build per-dataset train_ppp.tar / val_ppp.tar for STAGE_DATA staging.
# From repo root:
#   sbatch sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh
#   DATASETS=completely,partly sbatch sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh
#   DATASETS=1k_v1 sbatch sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh
#SBATCH --job-name=ppp_prepare_stage_tars
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=08:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs

DATASETS="${DATASETS:-completely}"

# shellcheck source=sbatch/ppp/prep/sqf_overlays.sh
source sbatch/ppp/prep/sqf_overlays.sh
sqf_resolve_overlays

singularity exec \
  --overlay env/ppp.ext3:ro \
  "${SQF_OVERLAY_ARGS[@]}" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; ${SQF_MOUNT_CHECK}\
  python -u PatchPerPix/experiments/flylight/prepare_ppp_stage_tars.py \
    --fisbe-root fisbe --datasets ${DATASETS} --overwrite"
