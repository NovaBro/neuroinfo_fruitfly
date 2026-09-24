#!/bin/bash
# FDcDn-specific watershed probe (separate Dc / Dn seeds). CPU only.
# Submit from repo root:
#   sbatch sbatch/biapy/watershed_tune_fcdcdn_probe_sbatch.sh
#SBATCH --job-name=ws-tune-fcdcdn
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/%x-%j.out
#SBATCH --error=sbatch/biapy/%x-%j.err

set -euo pipefail
cd /scratch/wmz2007/neuroinfo_fruitfly

singularity exec --overlay env/BiaPy_env.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \
    python biapy_work_folder/watershed_tune/probe_fcdcdn.py \
      --setup biapy-aug-zarr-seunet-FDcDn-skel \
      --run biapy_fcdcdn_skel_l40s_c3_winner_valpred \
      --out biapy_work_folder/watershed_tune/probe_out/fcdcdn_skel_c3_valpred_fdccdn'
