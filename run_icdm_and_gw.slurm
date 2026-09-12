#!/bin/bash
#SBATCH --job-name=icdm_and_gw
#SBATCH --output=logs/icdm_and_gw_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_61_general

cd /scratch/vh2308/GitHub/Repos/neuroinfo_fruitfly
mkdir -p logs

# Submit with: sbatch --export=ALL,SKEL_DIR=<dir>,TAG=<tag> run_icdm_and_gw.slurm
: "${SKEL_DIR:?Must set SKEL_DIR}"
: "${TAG:?Must set TAG}"

singularity exec --fakeroot \
  --overlay /scratch/vh2308/conda-envs/test_env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh && conda activate /ext3/miniforge3/envs/neurofly_2 && python -u compute_icdm_and_gw.py --skeleton-dir \"$SKEL_DIR\" --tag \"$TAG\""
