#!/bin/bash
# Parallel FISBe gt_instances 26-CC check. From repo root:
#   sbatch sbatch/ppp/prep/fisbe/check_gt_instance_connectivity_sbatch.sh
#SBATCH --job-name=gt_cc_check
#SBATCH --account=torch_pr_61_tandon_priority
#SBATCH --partition=cs
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs imaging_helpers_hpc/output

# Login-node python has zarr/scipy; keep the same env on the compute node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u imaging_helpers_hpc/check_gt_instance_connectivity.py \
  --fisbe-root fisbe/completely \
  --splits train val test \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --out imaging_helpers_hpc/output/gt_instance_connectivity.csv
