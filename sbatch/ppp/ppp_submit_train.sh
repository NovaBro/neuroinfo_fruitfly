#!/bin/bash
# PatchPerPix training submitter. Sets SLURM resources and exports env for
# sbatch/ppp/ppp_job.sh. From repo root:
#
#   ./sbatch/ppp/ppp_submit_train.sh our-noenc-cs \
#     --base original_default.toml --id our-noenc-cs --time 24:00:00
#   ./sbatch/ppp/ppp_submit_train.sh --dry-run basic_fmaps24
#
# Stages fisbe tars (STAGE_DATA=1) and runs -d train.
# Inference is ./sbatch/ppp/ppp_submit_test.sh. To run inference after this
# job, pass the printed job id as --dependency on the test command.
#
# Flow:
#   1. Parse CLI flags; leftover arg → PRESET.
#   2. Source ppp_submit_common.sh and ppp_submit_train_lib.sh.
#   3. Submit one train job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/ppp_job.sh"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ppp_submit_train.sh [--dry-run] <preset> [flags…]

Starts one training job: stage data + -d train.

Presets (= TOML stem): basic_8h | basic_long8h | basic_fmaps24 |
         basic_fmaps24_400k | basic_fmaps24_400k_bs5_aug |
         basic_fmaps24_400k_aug_cheap | basic-ppp-noenc | basic-ppp-noenc-cp | basic-ppp-noenc-cps |
         basic-ppp-decoder | base1-noenc-c | our-noenc-cs | our-noenc-synth_vh_u8

Flags:
  --id EXP_ID              Override experiment id (default from preset)
  --datasets A,B           Override dataset folders under fisbe/. Default: from
                           the TOMLs (train_datasets + val_datasets).
  --base FILE.toml         Base config
  --config FILE.toml       Extra overlay (repeatable)
  --time HH:MM:SS          Override walltime
  --partition NAME         Override partition
  --cpus N                 Override cpus-per-task
  --mem SIZE               Override mem (e.g. 96g)
  --account NAME           Override account, e.g. default torch_pr_61_general
                           torch_pr_61_tandon_priority or torch_pr_61_tandon_advanced
  --dependency SPEC        sbatch --dependency (job id → afterok:ID; or full SPEC)
  --dry-run                Print sbatch lines; do not submit

Inference: ./sbatch/ppp/ppp_submit_test.sh
EOF
  exit 1
}

DRY_RUN=0
KEEP_PRED_AFFS=0
KEEP_PRED_CODE=0
EVAL_BOTH_SPLITS=0
EXP_ID=""
DATASETS=""
BASE_CONFIG=""
EXTRA_CONFIGS=()
TIME_OVR=""
PART_OVR=""
CPUS_OVR=""
MEM_OVR=""
ACCOUNT="torch_pr_61_general"
SBATCH_DEPENDENCY=""
DEP_ID=""
LAST_ID=""
NEEDS_DECODE=0

positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --dry-run) DRY_RUN=1; shift ;;
    --id)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      EXP_ID="$2"; shift 2 ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      # Normalize to '+' (SLURM --export is comma-separated; commas in values break).
      DATASETS="${2//,/+}"; shift 2 ;;
    --base)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      BASE_CONFIG="$(basename "$2")"; shift 2 ;;
    --config)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      EXTRA_CONFIGS+=("$2"); shift 2 ;;
    --time)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      TIME_OVR="$2"; shift 2 ;;
    --partition)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      PART_OVR="$2"; shift 2 ;;
    --cpus)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      CPUS_OVR="$2"; shift 2 ;;
    --mem)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      MEM_OVR="$2"; shift 2 ;;
    --account)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      ACCOUNT="$2"; shift 2 ;;
    --dependency)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      if [[ "$2" == *:* ]]; then
        SBATCH_DEPENDENCY="$2"
      else
        SBATCH_DEPENDENCY="afterok:$2"
      fi
      shift 2 ;;
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

if (( ${#positionals[@]} != 1 )); then
  usage
fi

PRESET="${positionals[0]}"

# shellcheck source=ppp_submit_common.sh
source "${SCRIPT_DIR}/ppp_submit_common.sh"
# shellcheck source=ppp_submit_train_lib.sh
source "${SCRIPT_DIR}/ppp_submit_train_lib.sh"

# Validate preset in this shell (not $(…)): unknown names must abort, not empty-export.
preset_row "${PRESET}" >/dev/null

OVERLAY="$(preset_overlay "${PRESET}")"
if [[ -z "${EXP_ID}" ]]; then
  EXP_ID="$(preset_exp_id "${PRESET}")"
fi

# Exported for ppp_job.sh. Decode itself runs only on the test submitter.
NEEDS_DECODE="$(needs_decode_from_configs)"
if [[ "${NEEDS_DECODE}" != "0" && "${NEEDS_DECODE}" != "1" ]]; then
  echo "Error: needs_decode.py returned '${NEEDS_DECODE}' (expected 0 or 1)" >&2
  exit 1
fi

submit_train
