#!/bin/bash
#SBATCH --job-name=nnunet-bench
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250g
#SBATCH --time=02:00:00
#SBATCH --output=sbatch/nnunet/logs/%x-%j.out
#SBATCH --error=sbatch/nnunet/logs/%x-%j.err

# 5-epoch nnU-Net benchmark (nnUNetTrainerBenchmark_5epochs, real data-augmentation pipeline) on fold 0
# before committing to the 1000-epoch run. Read the fastest epoch time from benchmark_result.json and the
# GPU utilization from gpu_log/nnunet/gpu_usage_log_bench_<job>.csv: low utilization means data
# augmentation is the bottleneck -> raise NPROC_DA / CPUs (GPU-util floor on public partitions).
# (nnUNetTrainerBenchmark_5epochs_noDataLoading is skipped: its 1-channel dummy target does not fit the
#  3-region BCE loss of this dataset.) Results:
#   $nnUNet_results/Dataset501_FISBeCompletely/nnUNetTrainerBenchmark_5epochs*__.../fold_0/benchmark_result.json
# Submit from repo root:  sbatch sbatch/nnunet/nnunet_benchmark_sbatch.sh   (NPROC_DA=24 default)

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "${REPO_ROOT}"
source sbatch/nnunet/nnunet_common.sh
export nnUNet_n_proc_DA="${NPROC_DA:-24}"

start_gpu_logger "bench"
stage_preprocessed
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

for tr in nnUNetTrainerBenchmark_5epochs; do
  echo "=== ${tr} $(date) (nnUNet_n_proc_DA=${nnUNet_n_proc_DA}) ==="
  nnunet_exec "nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -p ${PLANS} -tr ${tr}"
  cat "${nnUNet_results}/${DATASET_NAME}/${tr}__${PLANS}__${CONFIG}/fold_${FOLD}/benchmark_result.json" || true
done
echo "=== done $(date) ==="
