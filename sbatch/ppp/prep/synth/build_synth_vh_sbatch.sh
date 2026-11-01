#!/bin/bash
# Build fisbe/<dataset>/{train,val,test} from vh2308 synth_MCFO samples (CPU).
# From repo root:
#   sbatch sbatch/ppp/prep/synth/build_synth_vh_sbatch.sh
#   SRC=... OUT=... SAMPLE_LIST=... sbatch sbatch/ppp/prep/synth/build_synth_vh_sbatch.sh
#   COPY=1 ... sbatch sbatch/ppp/prep/synth/build_synth_vh_sbatch.sh
#SBATCH --job-name=ppp_build_synth_vh
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

SRC="${SRC:-/scratch/vh2308/data/synth_MCFO/v1_u8/samples}"
OUT="${OUT:-/scratch/wmz2007/neuroinfo_fruitfly/fisbe/synth_vh_u8}"
SAMPLE_LIST="${SAMPLE_LIST:-/scratch/vh2308/data/synth_MCFO/v1_u8/sample_list_per_split.txt}"
COPY="${COPY:-0}"

COPY_FLAG=""
if [[ "${COPY}" == "1" ]]; then
  COPY_FLAG="--copy"
fi

SAMPLE_LIST_FLAG=""
if [[ -n "${SAMPLE_LIST}" ]]; then
  SAMPLE_LIST_FLAG="--sample-list ${SAMPLE_LIST}"
fi

singularity exec \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; \
  python -u PatchPerPix/experiments/flylight/build_synth_vh.py \
    --src ${SRC} --out ${OUT} ${SAMPLE_LIST_FLAG} ${COPY_FLAG}"
