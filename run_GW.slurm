#!/bin/bash
#SBATCH --job-name=GW
#SBATCH --output=run_GW.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --account=torch_pr_61_tandon_advanced

eval "$(conda shell.bash hook)"

conda activate /scratch/vh2308/conda-envs/neurofly

python -u calc_GW.py > run_GW.txt 2>&1