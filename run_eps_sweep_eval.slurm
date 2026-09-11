#!/bin/bash
#SBATCH --job-name=eps_sweep_eval
#SBATCH --output=logs/eps_sweep_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_61_general

cd /scratch/vh2308/GitHub/Repos/neuroinfo_fruitfly

singularity exec --fakeroot \
  --overlay /scratch/vh2308/conda-envs/test_env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh && conda activate /ext3/miniforge3/envs/neurofly_2 && python -u evaluate_eps_sweep.py && python -u plot_eps_sweep.py"
