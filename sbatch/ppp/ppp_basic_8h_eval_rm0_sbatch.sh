#!/bin/bash
# Re-eval ppp_basic_8h with remove_small_components=0 (no retrain/predict).
# From repo root: sbatch sbatch/ppp/ppp_basic_8h_eval_rm0_sbatch.sh
#SBATCH --job-name=ppp_basic_eval_rm0
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --output=sbatch/ppp/ppp_basic_eval_rm0.out
#SBATCH --error=sbatch/ppp/ppp_basic_eval_rm0.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

singularity exec \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_8h.toml \
    --config flylight/setups/setup01/eval_rm0.toml \
    -d evaluate \
    --app flylight --root ../../metrics/ppp -id ppp_basic_8h \
    --test-checkpoint last \
    --sample R22C03-20180918_66_J2'
