#!/bin/bash
# CPU diagnosis of ppp_basic_8h empty predictions. From repo root:
#   sbatch sbatch/ppp/diagnose_ppp_basic_sbatch.sh
#SBATCH --job-name=ppp_diagnose
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --partition=cs,cpu_short
#SBATCH --output=sbatch/ppp/ppp_diagnose.out
#SBATCH --error=sbatch/ppp/ppp_diagnose.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

singularity exec \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate ppp; \
  python -u sbatch/ppp/diagnose_ppp_basic_instances.py'
