#!/bin/bash
# Resources (job-name, cpus, time, mem, GPU) are set by biapy-py_sbatch_chain.sh
# or must be passed on the sbatch CLI for direct submits.

#SBATCH --account=torch_pr_61_tandon_priority
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
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

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

# >>>> Stage train/val data to $SLURM_TMPDIR (train only) >>>>
# Full raw + instance-channel GT so IN_MEMORY load hits node-local SSD.
# Channel GT basename must be a sibling of GT_PATH=…/label (BiaPy rewrite).
BIAPY_STAGE_ROOT=
if [[ "$mode" == "train" ]]; then
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "Error: SLURM_TMPDIR is unset; cannot stage training data" >&2
    exit 1
  fi

  SRC_ROOT="fisbe/biapy-no-aug-zarr"
  CONFIG_PATH="biapy_work_folder/configs/${config_file}"
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config not found: ${CONFIG_PATH}" >&2
    exit 1
  fi

  # Resolve exactly one label_F*_Sk*_I dir per split. Narrow by DATA_CHANNELS
  # when both FDb and FDcDn skel caches exist under the same parent.
  resolve_channel_gt() {
    local split="$1"
    local parent="${SRC_ROOT}/${split}"
    local -a matches=()
    local d

    if grep -qE "DATA_CHANNELS:[[:space:]]*\[.*'Dc'" "$CONFIG_PATH"; then
      for d in "${parent}"/label_F*_Dc*_Dn*_Sk*_I; do
        [[ -d "$d" ]] && matches+=("$d")
      done
    elif grep -qE "DATA_CHANNELS:[[:space:]]*\[.*'Db'" "$CONFIG_PATH"; then
      for d in "${parent}"/label_F*_Db*_Sk*_I; do
        [[ -d "$d" ]] && matches+=("$d")
      done
    else
      for d in "${parent}"/label_F*_Sk*_I; do
        [[ -d "$d" ]] && matches+=("$d")
      done
    fi

    if [[ ${#matches[@]} -ne 1 ]]; then
      echo "Error: expected exactly one channel GT under ${parent} for ${config_file}, got ${#matches[@]}:" >&2
      printf '  %s\n' "${matches[@]:-}" >&2
      exit 1
    fi
    printf '%s\n' "${matches[0]}"
  }

  TRAIN_CH_GT="$(resolve_channel_gt train)"
  VAL_CH_GT="$(resolve_channel_gt val)"
  BIAPY_STAGE_ROOT="${SLURM_TMPDIR}/biapy_data"

  echo "Staging train/val raw + channel GT -> ${BIAPY_STAGE_ROOT}"
  echo "  train channel GT: ${TRAIN_CH_GT}"
  echo "  val channel GT:   ${VAL_CH_GT}"
  df -h "$SLURM_TMPDIR"
  STAGE_T0=$(date +%s)

  mkdir -p "${BIAPY_STAGE_ROOT}/train" "${BIAPY_STAGE_ROOT}/val"
  rsync -a --delete "${SRC_ROOT}/train/raw/" "${BIAPY_STAGE_ROOT}/train/raw/"
  rsync -a --delete "${TRAIN_CH_GT}/" \
    "${BIAPY_STAGE_ROOT}/train/$(basename "$TRAIN_CH_GT")/"
  rsync -a --delete "${SRC_ROOT}/val/raw/" "${BIAPY_STAGE_ROOT}/val/raw/"
  rsync -a --delete "${VAL_CH_GT}/" \
    "${BIAPY_STAGE_ROOT}/val/$(basename "$VAL_CH_GT")/"

  # BiaPy check_configuration requires GT_PATH=…/label to exist before rewrite
  # to the label_F… sibling; relative targets stay valid under the bind mount.
  ln -sfn "$(basename "$TRAIN_CH_GT")" "${BIAPY_STAGE_ROOT}/train/label"
  ln -sfn "$(basename "$VAL_CH_GT")" "${BIAPY_STAGE_ROOT}/val/label"

  STAGE_T1=$(date +%s)
  echo "Staging finished in $((STAGE_T1 - STAGE_T0))s"
  df -h "$SLURM_TMPDIR"
  export BIAPY_STAGE_ROOT
fi
# <<<< Stage train/val data <<<<

# LD_PRELOAD conda libstdc++ so scipy's CXXABI_1.3.15 requirement is met
# without putting all of $CONDA_PREFIX/lib on LD_LIBRARY_PATH (breaks CUDA).
singularity_args=(exec)
if [[ "$mode" == "train" || "$mode" == "test" ]]; then
  singularity_args+=(--nv)
fi
if [[ -n "$BIAPY_STAGE_ROOT" ]]; then
  singularity_args+=(--bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
fi
# Forward BIAPY_STAGE_ROOT explicitly (some Singularity setups scrub the env).
stage_env_export=""
if [[ -n "$BIAPY_STAGE_ROOT" ]]; then
  stage_env_export="export BIAPY_STAGE_ROOT=\"${BIAPY_STAGE_ROOT}\"; "
fi
singularity "${singularity_args[@]}" \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    conda activate BiaPy_env; \
    export LD_PRELOAD=\"\${CONDA_PREFIX}/lib/libstdc++.so.6\${LD_PRELOAD:+:\$LD_PRELOAD}\"; \
    export PYTHONUNBUFFERED=1; \
    ${stage_env_export}\
    echo \"running biapy_work_folder/run_biapy-py.py\"; \
    python3 -u biapy_work_folder/run_biapy-py.py \
        -c \"${config_file}\" \
        -m \"${mode}\" \
        --job-name \"${job_name}\" \
        --run-id \"${run_id}\" "
