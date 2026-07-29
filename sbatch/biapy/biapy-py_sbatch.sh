#!/bin/bash
#SBATCH --job-name=biapy-py
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --constraint='l40s|a100'
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/%x-%j.out
#SBATCH --error=sbatch/biapy/%x-%j.err

set -e
module purge
# >>>> Set Job Config >>>>
# Usage: sbatch sbatch/biapy/biapy-py_sbatch.sh [config.yaml] [train|test]
# Or use biapy-py_sbatch_chain.sh to submit train/test with dependencies.
config_file="${1:-biapy-py_v3.yaml}"
mode="${2:-train}"
# <<<< Set Job Config <<<<

config_name="${config_file%.*}"
echo "SBATCH Run: ${config_file}, mode: ${mode}"

# >>>> GPU Tracking >>>>
GPU_LOGGER_PID=
cleanup() {
  if [[ -n "$GPU_LOGGER_PID" ]]; then
    kill "$GPU_LOGGER_PID" 2>/dev/null || true
    wait "$GPU_LOGGER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nohup nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 3 > "gpu_biapy-py_${config_name}_${mode}.csv" &
GPU_LOGGER_PID=$!
# <<<< GPU Tracking <<<<

# Train Model
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    conda activate BiaPy_env; \
    echo \"running BiaPy/run_biapy-py.py\"; \
    python3 BiaPy/run_biapy-py.py \
        -c \"${config_file}\" \
        -m \"${mode}\" \
        --job-name \"${config_name}\" \
        --run-id 0 "
