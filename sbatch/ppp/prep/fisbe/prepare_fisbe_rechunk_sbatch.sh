#!/bin/bash
# Re-prepare FISBe PPP arrays with 64^3 chunks. From repo root:
#   sbatch sbatch/ppp/prep/fisbe/prepare_fisbe_rechunk_sbatch.sh
#SBATCH --job-name=ppp_prepare_rechunk
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cs,cpu_short
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs

singularity exec \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate ppp; \
  python -u PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py \
    --fisbe-root fisbe --opening-radius 1 --overwrite'
