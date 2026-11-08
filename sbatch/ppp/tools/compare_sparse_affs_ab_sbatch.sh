#!/bin/bash
# CPU job: dense vs sparse pred_affs A/B comparison (compare_sparse_affs_ab.py).
# From the repo (or worktree) root:
#   sbatch --dependency=afterok:<dense_eval>:<sparse_eval> sbatch/ppp/tools/compare_sparse_affs_ab_sbatch.sh
# Env: DENSE_ROOT, SPARSE_ROOT, REF_ROOT (optional), SAMPLE, CKPT.
#SBATCH --job-name=ppp_sparse_ab_compare
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs,cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/ppp/logs/sparse_ab_compare-%j.out
#SBATCH --error=sbatch/ppp/logs/sparse_ab_compare-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
args=(--dense-root "${DENSE_ROOT:-metrics/ppp/abtest-dense}"
      --sparse-root "${SPARSE_ROOT:-metrics/ppp/abtest-sparse}"
      --sample "${SAMPLE:-1k_v1.1/brain_00005}" --ckpt "${CKPT:-10000}")
[[ -n "${REF_ROOT:-}" ]] && args+=(--ref-root "${REF_ROOT}")
singularity exec --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /ext3/miniforge3/envs/ppp/bin/python -u sbatch/ppp/tools/compare_sparse_affs_ab.py "${args[@]}"
