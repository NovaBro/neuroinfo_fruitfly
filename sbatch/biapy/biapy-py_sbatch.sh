#!/bin/bash
# Resources (job-name, cpus, time, mem, GPU) are set by biapy-py_sbatch_chain.sh
# or must be passed on the sbatch CLI for direct submits.

#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/%x-%j.out
#SBATCH --error=sbatch/biapy/%x-%j.err

# >>>> Set Job Config >>>>
# Usage:
#   Prefer: ./sbatch/biapy/biapy-py_sbatch_chain.sh … (sets resources per mode)
#   Direct: sbatch --job-name=… --cpus-per-task=… --time=… --mem=… \
#             [--gres=gpu:1 --constraint='h100|h200'] \
#             sbatch/biapy/biapy-py_sbatch.sh \
#             [config.yaml] [train|test|preprocessing] [job_name] [run_id]
# Direct submits MUST pass job-name/cpus/time/mem; train/test also need GPU flags.
# Flat config e.g. biapy-v1-channel-rot.yaml; empty job_name ($3) derives from config path;
# run_id ($4) defaults to 0.
config_file="${1:-biapy-v1-channel-rot.yaml}"
mode="${2:-train}"
job_name_arg="${3:-}"
run_id="${4:-0}"
# <<<< Set Job Config <<<<

set -e
module purge

# BiaPy --job-name: override if provided; else use stem directory for nested paths
# so train/test share checkpoints.
if [[ -n "$job_name_arg" ]]; then
  job_name="$job_name_arg"
elif [[ "$config_file" == */* ]]; then
  job_name="${config_file%/*}"
  job_name="${job_name##*/}"
else
  job_name="${config_file%.*}"
fi
echo "SBATCH Run: ${config_file}, mode: ${mode}, job-name: ${job_name}, run-id: ${run_id}"

# >>>> GPU Tracking (train/test only) >>>>
GPU_LOGGER_PID=
cleanup() {
  if [[ -n "$GPU_LOGGER_PID" ]]; then
    kill "$GPU_LOGGER_PID" 2>/dev/null || true
    wait "$GPU_LOGGER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if [[ "$mode" == "train" || "$mode" == "test" ]]; then
  nohup nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 3 > "gpu_log/gpu_biapy-py_${job_name}_${mode}_${SLURM_JOB_ID}.csv" &
  GPU_LOGGER_PID=$!
fi
# <<<< GPU Tracking <<<<

# LD_PRELOAD conda libstdc++ so scipy's CXXABI_1.3.15 requirement is met
# without putting all of $CONDA_PREFIX/lib on LD_LIBRARY_PATH (breaks CUDA).
singularity_args=(exec)
if [[ "$mode" == "train" || "$mode" == "test" ]]; then
  singularity_args+=(--nv)
fi
singularity "${singularity_args[@]}" \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    conda activate BiaPy_env; \
    export LD_PRELOAD=\"\${CONDA_PREFIX}/lib/libstdc++.so.6\${LD_PRELOAD:+:\$LD_PRELOAD}\"; \
    export PYTHONUNBUFFERED=1; \
    echo \"running BiaPy/run_biapy-py.py\"; \
    python3 -u BiaPy/run_biapy-py.py \
        -c \"${config_file}\" \
        -m \"${mode}\" \
        --job-name \"${job_name}\" \
        --run-id \"${run_id}\" "
