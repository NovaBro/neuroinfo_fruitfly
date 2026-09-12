#!/bin/bash
#SBATCH --job-name=prune_skeletons
#SBATCH --output=logs/prune_skeletons_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_61_general

cd /scratch/vh2308/GitHub/Repos/neuroinfo_fruitfly
mkdir -p logs

# Optionally restrict to one condition: sbatch --export=ALL,CONDITION=<tag> run_prune_skeletons.slurm
COND_ARGS=""
if [ -n "$CONDITION" ]; then
  COND_ARGS="--condition $CONDITION"
fi

singularity exec --fakeroot \
  --overlay /scratch/vh2308/conda-envs/test_env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh && conda activate /ext3/miniforge3/envs/neurofly_2 && python -u prune_skeletons.py $COND_ARGS"
