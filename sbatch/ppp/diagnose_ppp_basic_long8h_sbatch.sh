#!/bin/bash
# Diagnose ppp_basic_long8h ckpt 30000 @ VI th=0.3 (HDF sizes + streamed aff/numinst).
# From repo root: sbatch sbatch/ppp/diagnose_ppp_basic_long8h_sbatch.sh
#SBATCH --job-name=ppp_diagnose_long8h
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/ppp_diagnose_long8h.out
#SBATCH --error=sbatch/ppp/ppp_diagnose_long8h.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

singularity exec \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate ppp; \
  python -u sbatch/ppp/diagnose_ppp_basic_instances.py'
