#!/bin/bash
# Submit BiaPy preprocessing/train/test sbatch jobs, chaining with afterok in order.
#
# Usage (from repo root):
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> preprocessing
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> train
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> test
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> train test
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> preprocessing train test
#
# Stem is the YAML basename under BiaPy/configs/ (e.g. biapy-v1-channel-rot).
# Resolves to ${stem}.yaml for all modes (mode is passed separately to the job).
# -j/--job-name overrides BiaPy --job-name and the SLURM job-name base (default: stem).
# -r/--run-id overrides BiaPy --run-id (default: 0).
# Chain order when multiple modes: preprocessing → train → test (each afterok on previous).
# Per-mode SLURM resources are set here (job-name, cpus, time, mem; GPU for train/test only).
# Preprocessing: 8 cpus, 5h, 128g, no GPU. Train/test: 8 cpus, 20h, 512g, GPU h100|h200.

# Fail on errors (-e), unset vars (-u), and failed pipeline stages (pipefail).
set -euo pipefail

# Print help to stderr and exit with failure.
usage() {
  echo "Usage: $0 [-j JOB_NAME] [-r RUN_ID] <stem> <preprocessing|train|test> [preprocessing|train|test]..." >&2
  echo "  Stem is the YAML basename under BiaPy/configs/ (e.g. biapy-v1-channel-rot)." >&2
  echo "  Resolves to BiaPy/configs/\${stem}.yaml for all modes." >&2
  echo "  Modes (1–3, no duplicates): preprocessing, train, test." >&2
  echo "  Chain order: preprocessing → train → test (afterok)." >&2
  echo "  -j, --job-name   BiaPy/SLURM job-name override (default: stem)" >&2
  echo "  -r, --run-id     BiaPy --run-id (default: 0)" >&2
  echo "  Examples:" >&2
  echo "    $0 biapy-v1-channel-rot preprocessing" >&2
  echo "    $0 biapy-v1-channel-rot train" >&2
  echo "    $0 biapy-v1-channel-rot test" >&2
  echo "    $0 biapy-v1-channel-rot train test" >&2
  echo "    $0 biapy-v1-channel-rot preprocessing train test" >&2
  echo "    $0 -j v1-something -r run-0 biapy-v1-channel-rot train test" >&2
  exit 1
}

# Optional overrides; empty job_name means derive from config stem in sbatch script.
job_name=""
run_id="0"

# Parse flags; remaining args are stem + modes.
positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--job-name)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires an argument" >&2
        usage
      fi
      job_name="$2"
      shift 2
      ;;
    -r|--run-id)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires an argument" >&2
        usage
      fi
      run_id="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "Error: unknown option '$1'" >&2
      usage
      ;;
    *)
      positionals+=("$1")
      shift
      ;;
  esac
done

# Require stem + 1 to 3 modes (e.g. train, or preprocessing train test).
if [[ ${#positionals[@]} -lt 2 || ${#positionals[@]} -gt 4 ]]; then
  usage
fi

# First positional is the config stem (YAML basename); rest are modes.
stem="${positionals[0]}"
stem="${stem%.yaml}"
stem="${stem##*/}"
modes_raw=("${positionals[@]:1}")

# Validate modes: preprocessing/train/test, no duplicates; set do_* flags.
do_preprocessing=0
do_train=0
do_test=0
for mode in "${modes_raw[@]}"; do
  case "$mode" in
    preprocessing)
      if [[ "$do_preprocessing" -eq 1 ]]; then
        echo "Error: duplicate mode 'preprocessing'" >&2
        exit 1
      fi
      do_preprocessing=1
      ;;
    train)
      if [[ "$do_train" -eq 1 ]]; then
        echo "Error: duplicate mode 'train'" >&2
        exit 1
      fi
      do_train=1
      ;;
    test)
      if [[ "$do_test" -eq 1 ]]; then
        echo "Error: duplicate mode 'test'" >&2
        exit 1
      fi
      do_test=1
      ;;
    *)
      echo "Error: unknown mode '$mode' (expected preprocessing, train, or test)" >&2
      usage
      ;;
  esac
done

# Absolute path to this script's directory, then the sibling sbatch job script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/biapy-py_sbatch.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/BiaPy/configs"

# Flat config shared by all modes: BiaPy/configs/${stem}.yaml
config_file="${stem}.yaml"
config_path="${CONFIG_DIR}/${config_file}"
if [[ ! -f "${config_path}" ]]; then
  echo "Error: config not found: ${config_path}" >&2
  exit 1
fi

# SLURM job-name base: -j override if set, else config stem.
slurm_name_base="${job_name:-$stem}"

# Submit in fixed order; each job depends on the previous one's afterok when chained.
prev_id=""

# Submit preprocessing (optional).
if [[ "$do_preprocessing" -eq 1 ]]; then
  sbatch_args=(
    --parsable
    --job-name="BiaPy-py-${slurm_name_base}-preprocessing"
    --cpus-per-task=8
    --time=8:00:00
    --mem=256g
  )
  job_id=$(sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT" "$config_file" "preprocessing" "$job_name" "$run_id")
  prev_id="$job_id"
  echo "Submitted preprocessing job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id})"
fi

# Submit train (optional); depends on prev_id when set.
if [[ "$do_train" -eq 1 ]]; then
  sbatch_args=(
    --parsable
    --job-name="BiaPy-py-${slurm_name_base}-train"
    --cpus-per-task=16
    --time=12:00:00
    --mem=256g
    --gres=gpu:1
    --constraint='h200'
    # --constraint='h100|h200'
  )
  if [[ -n "$prev_id" ]]; then
    sbatch_args+=(--dependency="afterok:${prev_id}")
  fi

  job_id=$(sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT" "$config_file" "train" "$job_name" "$run_id")
  if [[ -n "$prev_id" ]]; then
    echo "Submitted train job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id}, dependency=afterok:${prev_id})"
  else
    echo "Submitted train job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id})"
  fi
  prev_id="$job_id"
fi

# Submit test (optional); depends on prev_id when set.
if [[ "$do_test" -eq 1 ]]; then
  sbatch_args=(
    --parsable
    --job-name="BiaPy-py-${slurm_name_base}-test"
    --cpus-per-task=6
    --time=3:00:00
    --mem=128g
    --gres=gpu:1
    --constraint='l40s|h100'
    # --constraint='h100|h200'
  )
  if [[ -n "$prev_id" ]]; then
    sbatch_args+=(--dependency="afterok:${prev_id}")
  fi

  job_id=$(sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT" "$config_file" "test" "$job_name" "$run_id")
  if [[ -n "$prev_id" ]]; then
    echo "Submitted test job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id}, dependency=afterok:${prev_id})"
  else
    echo "Submitted test job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id})"
  fi
fi
