#!/bin/bash
# Submit BiaPy train and/or test sbatch jobs, chaining test after train when both are requested.
#
# Usage (from repo root):
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh <config.yaml> train
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh <config.yaml> test
#   ./sbatch/biapy/biapy-py_sbatch_chain.sh <config.yaml> train test

# Fail on errors (-e), unset vars (-u), and failed pipeline stages (pipefail).
set -euo pipefail

# Print help to stderr and exit with failure.
usage() {
  echo "Usage: $0 <config.yaml> <train|test> [train|test]" >&2
  echo "  Examples:" >&2
  echo "    $0 biapy-py_v3.yaml train" >&2
  echo "    $0 biapy-py_v3.yaml test" >&2
  echo "    $0 biapy-py_v3.yaml train test" >&2
  exit 1
}

# Require config + 1 or 2 modes (e.g. train, or train test).
if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
fi

# First arg is the BiaPy YAML; shift so "$@" is only the modes.
config_file="$1"
shift

# Validate modes: only train/test, no duplicates; keep order in modes[].
modes=()
seen_train=0
seen_test=0
for mode in "$@"; do
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
# Job-name stem: strip extension, then any directory prefix (e.g. biapy-py_v3).
stem="${config_file%.*}"
stem="${stem##*/}"

# Submit each mode; if train then test, attach --dependency=afterok:<train_id>.
train_id=""
for mode in "${modes[@]}"; do
  # --parsable: sbatch prints only the job ID (for capture below).
  sbatch_args=(--parsable --job-name="BiaPy-py-${stem}-${mode}")
  if [[ "$mode" == "test" && -n "$train_id" ]]; then
    sbatch_args+=(--dependency="afterok:${train_id}")
  fi

  # Pass config and mode as $1/$2 into biapy-py_sbatch.sh.
  job_id=$(sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT" "$config_file" "$mode")
  if [[ "$mode" == "train" ]]; then
    train_id="$job_id"
    echo "Submitted train job ${job_id} (config=${config_file})"
  else
    if [[ -n "$train_id" ]]; then
      echo "Submitted test job ${job_id} (config=${config_file}, dependency=afterok:${train_id})"
    else
      echo "Submitted test job ${job_id} (config=${config_file})"
    fi
  fi
done
