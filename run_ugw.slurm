#!/bin/bash
#SBATCH --job-name=UGW
#SBATCH --output=run_UGW.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --account=torch_pr_61_general

# 1. Navigate to your project directory
cd /scratch/vh2308/GitHub/Repos/neuroinfo_fruitfly

# 2. Launch Singularity and pass the environment setup and python command as a single execution string
# CPU-only multicore backend (no GPU requested) per HPC policy against under-utilized GPU jobs.
singularity exec --fakeroot \
  --overlay /scratch/vh2308/conda-envs/test_env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh && conda activate /ext3/miniforge3/envs/neurofly_2 && python ugw.py"
