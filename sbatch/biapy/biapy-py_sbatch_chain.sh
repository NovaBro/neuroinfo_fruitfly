#!/bin/bash
# Submit BiaPy train and/or test sbatch jobs, chaining test after train when both are requested.
#
# Usage (from repo root):
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> train
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> test
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh [-j JOB_NAME] [-r RUN_ID] <stem> train test
#
# Stem is the folder name under BiaPy/configs/ (e.g. biapy-py_v3).
# Resolves to ${stem}/${stem}-${mode}.yaml for each mode.
# -j/--job-name overrides BiaPy --job-name and the SLURM job-name base (default: stem).
# -r/--run-id overrides BiaPy --run-id (default: 0).

# Fail on errors (-e), unset vars (-u), and failed pipeline stages (pipefail).
set -euo pipefail

# Print help to stderr and exit with failure.
usage() {
  echo "Usage: $0 [-j JOB_NAME] [-r RUN_ID] <stem> <train|test> [train|test]" >&2
  echo "  Stem is the config folder under BiaPy/configs/ (e.g. biapy-py_v3)." >&2
  echo "  -j, --job-name   BiaPy/SLURM job-name override (default: stem)" >&2
  echo "  -r, --run-id     BiaPy --run-id (default: 0)" >&2
  echo "  Examples:" >&2
  echo "    $0 biapy-py_v3 train" >&2
  echo "    $0 biapy-py_v3 test" >&2
  echo "    $0 biapy-py_v3 train test" >&2
  echo "    $0 -j v1-something -r run-0 biapy-py_v3 train test" >&2
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

# Require stem + 1 or 2 modes (e.g. train, or train test).
if [[ ${#positionals[@]} -lt 2 || ${#positionals[@]} -gt 3 ]]; then
  usage
fi

# First positional is the config stem (folder name); rest are modes.
stem="${positionals[0]}"
stem="${stem%.yaml}"
stem="${stem##*/}"
modes_raw=("${positionals[@]:1}")

# Validate modes: only train/test, no duplicates; keep order in modes[].
modes=()
seen_train=0
seen_test=0
for mode in "${modes_raw[@]}"; do
  case "$mode" in
    train)
      if [[ "$seen_train" -eq 1 ]]; then
        echo "Error: duplicate mode 'train'" >&2
        exit 1
      fi
      seen_train=1
      modes+=("train")
      ;;
    test)
      if [[ "$seen_test" -eq 1 ]]; then
        echo "Error: duplicate mode 'test'" >&2
        exit 1
      fi
      seen_test=1
      modes+=("test")
      ;;
    *)
      echo "Error: unknown mode '$mode' (expected train or test)" >&2
      usage
      ;;
  esac
done

# Absolute path to this script's directory, then the sibling sbatch job script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/biapy-py_sbatch.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/BiaPy/configs"

# SLURM job-name base: -j override if set, else config stem.
slurm_name_base="${job_name:-$stem}"

# Submit each mode; if train then test, attach --dependency=afterok:<train_id>.
train_id=""
for mode in "${modes[@]}"; do
  config_file="${stem}/${stem}-${mode}.yaml"
  config_path="${CONFIG_DIR}/${config_file}"
  if [[ ! -f "${config_path}" ]]; then
    echo "Error: config not found: ${config_path}" >&2
    exit 1
  fi

  # --parsable: sbatch prints only the job ID (for capture below).
  sbatch_args=(--parsable --job-name="BiaPy-py-${slurm_name_base}-${mode}")
  if [[ "$mode" == "test" && -n "$train_id" ]]; then
    sbatch_args+=(--dependency="afterok:${train_id}")
  fi

  # Pass config, mode, optional job-name ($3), and run-id ($4) into biapy-py_sbatch.sh.
  job_id=$(sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT" "$config_file" "$mode" "$job_name" "$run_id")
  if [[ "$mode" == "train" ]]; then
    train_id="$job_id"
    echo "Submitted train job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id})"
  else
    if [[ -n "$train_id" ]]; then
      echo "Submitted test job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id}, dependency=afterok:${train_id})"
    else
      echo "Submitted test job ${job_id} (config=${config_file}, job-name=${slurm_name_base}, run-id=${run_id})"
    fi
  fi
done
