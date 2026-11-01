#!/bin/bash
# Replace fisbe/synth_vh_u8 raw/gt_instances symlinks (into vh2308's loose v1_u8/) with real copies
# read from the v1_u8.sqf SquashFS overlay (mounted at /synth_MCFO/v1_u8). CPU only.
# From repo root:
#   sbatch sbatch/ppp/prep/synth/materialize_synth_vh_u8_from_sqf_sbatch.sh
#   SPLITS=val,test,train DRY_RUN=1 sbatch sbatch/ppp/prep/synth/materialize_synth_vh_u8_from_sqf_sbatch.sh
# Default SPLITS=val,test: train reads the self-contained train_ppp.tar, and copying train
# would add ~80k files (inodes) to /scratch.
#SBATCH --job-name=ppp_materialize_synth_u8
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs

SQF="${SQF:-/scratch/vh2308/data/synth_MCFO/v1_u8.sqf}"
DATASET_ROOT="${DATASET_ROOT:-fisbe/synth_vh_u8}"
SPLITS="${SPLITS:-val,test}"
DRY_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_FLAG="--dry-run"

singularity exec \
  --overlay env/ppp.ext3:ro \
  --overlay "${SQF}:ro" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; ls /synth_MCFO/v1_u8 | head -5; \
  python -u PatchPerPix/experiments/flylight/materialize_synth_links.py \
    --dataset-root ${DATASET_ROOT} --src-root /synth_MCFO/v1_u8/samples \
    --splits ${SPLITS} ${DRY_FLAG}"
