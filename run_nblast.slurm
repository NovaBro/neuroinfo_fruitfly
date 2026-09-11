#!/bin/bash
#SBATCH --job-name=nblast
#SBATCH --output=nblast.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_61_general

# 2. Initialize conda for this non-interactive bash script
eval "$(conda shell.bash hook)"

# 3. Activate your environment (Replace 'your_env_name' with your actual environment name)
conda activate /scratch/vh2308/conda-envs/neurofly

python -u run_nblast.py > nblast.txt 2>&1