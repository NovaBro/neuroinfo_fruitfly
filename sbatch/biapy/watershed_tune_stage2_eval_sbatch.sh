#!/bin/bash
# Stage 2 watershed_tune eval_val exit gate (CPU only).
#SBATCH --job-name=ws-tune-stage2-eval
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=64g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/%x-%j.out
#SBATCH --error=sbatch/biapy/%x-%j.err

set -euo pipefail
cd /scratch/wmz2007/neuroinfo_fruitfly

singularity exec --overlay env/BiaPy_env.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \
    python biapy_work_folder/watershed_tune/eval_val.py \
      --setup biapy-aug-zarr-seunet-FDb-dice \
      --run biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred \
      --compare-baseline --strict-seed 0.7'
