#!/bin/bash
#SBATCH --job-name=BiaPy-py
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=256g
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/biapy-py-%j.out
#SBATCH --error=sbatch/biapy/biapy-py-%j.err

set -e
module purge
# >>>> Set Job Config >>>>
config_file="biapy-py_v3.yaml"
mode="test"
# sbatch --dependency=afterok:JOBID sbatch/biapy/biapy-py_sbatch.sh 
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
