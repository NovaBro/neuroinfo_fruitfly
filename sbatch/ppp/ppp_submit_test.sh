#!/bin/bash
# PatchPerPix inference submitter. Sets SLURM resources and exports env for
# sbatch/ppp/ppp_job.sh. From repo root:
#
#   ./sbatch/ppp/ppp_submit_test.sh array our-noenc-cs --ckpt 400000 --split test
#   ./sbatch/ppp/ppp_submit_test.sh run base1-noenc-c --ckpt 400000 \
#     --sample completely/R22C03-20180918_66_J2 --split val
#   ./sbatch/ppp/ppp_submit_test.sh evaluate our-noenc-cs --split test --ckpt 400000
#   ./sbatch/ppp/ppp_submit_test.sh ckpt-sweep basic_fmaps24 --ckpts 50000,60000,70000,80000
#   ./sbatch/ppp/ppp_submit_test.sh vi-sweep basic_long8h --ckpt 30000
#   ./sbatch/ppp/ppp_submit_test.sh --dry-run array basic_fmaps24 --ckpt 80000
#
# Outputs under metrics/ppp/<exp>/{val,test}/ matching --split (comma-capable).
# array --stage all (default): GPU predict → [GPU decode if train_code] →
#   GPU label graph (vi_split_gpu) → CPU label (vi_split_cpu, aftercorr) →
#   CPU evaluate.
# Training is ./sbatch/ppp/ppp_submit_train.sh. Pass its job id as --dependency
# to wait for training first.
#
# Flow:
#   1. Parse CLI flags; leftover args → MODE, PRESET.
#   2. Source ppp_submit_common.sh and ppp_submit_test_lib.sh.
#   3. Dispatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/ppp_job.sh"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ppp_submit_test.sh [--dry-run] <mode> <preset> [flags…]

Modes:
  run            One sample: --do TASKS (default predict[+decode]+label+evaluate;
                 decode included when merged configs have train_code)
  evaluate       -d evaluate (CPU). --split val|test → that split;
                 omit or --split val,test → both + consolidated summary.csv
  ckpt-sweep     For each ckpt: predict → [decode] → label+evaluate (afterok)
  array          Per-volume pipeline; --stage all|predict|decode|label|legacy
                 (default all = GPU predict → [decode] → GPU graph → CPU label
                 → evaluate; decode inserted when train_code)
  vi-sweep       afterok chain over --overlays (default vi_th_0_3..0_7)

Presets (= TOML stem): basic_8h | basic_long8h | basic_fmaps24 |
         basic_fmaps24_400k | basic_fmaps24_400k_bs5_aug |
         basic_fmaps24_400k_aug_cheap | basic-ppp-noenc | basic-ppp-noenc-cp | basic-ppp-noenc-cps |
         basic-ppp-decoder | base1-noenc-c | our-noenc-cs | our-noenc-synth_vh_u8

Flags:
  --id EXP_ID              Override experiment id (default from preset)
  --ckpt N                 Checkpoint number
  --ckpts A,B,C            ckpt-sweep list (default 50000,60000,70000,80000)
  --sample NAME            Sample id as {dataset}/{stem}
                           (default completely/R22C03-20180918_66_J2)
  --split val|test[,…]     FISBe split(s) + output dir (comma-capable).
                           Infer jobs omit --test-data; run_ppp gets --split.
  --datasets A,B           Override dataset folders under fisbe/. Default: from
                           the TOMLs (<split>_datasets). Applies to both val and
                           test arrays; run_ppp errors if a listed dir is missing.
  --stage all|predict|decode|label|legacy   array stage (default all)
  --pct K                  Array concurrency %K (default 2)
  --agg                    After --stage legacy: CPU evaluate afterok
  --vi-gpu                 array label stage: one GPU job (no GPU/CPU split)
  --vi FILE.toml           VI overlay (default vi_th_0_3.toml)
  --eval FILE.toml         Eval overlay (default eval_rm200.toml)
  --overlays A.toml,B.toml vi-sweep list
  --base FILE.toml         Base config (default default_l40s.toml; under setup01)
  --config FILE.toml       Extra overlay (repeatable)
  --do TASKS               Override -d for run (space or + separated)
  --time HH:MM:SS          Override walltime
  --partition NAME         Override partition
  --cpus N                 Override cpus-per-task
  --mem SIZE               Override mem (e.g. 96g)
  --account NAME           Override account, e.g. default torch_pr_61_general
                           or torch_pr_61_tandon_advanced
  --dependency SPEC        sbatch --dependency for the first job (job id →
                           afterok:ID; or full SPEC e.g. afterok:1:2)
  --force-predict          ckpt-sweep: always re-predict
  --keep-pred-affs         After evaluate, keep volumes/pred_affs (default: strip)
  --keep-pred-code         After evaluate, keep volumes/pred_code (default: strip)
  --dry-run                Print sbatch lines; do not submit

Training: ./sbatch/ppp/ppp_submit_train.sh
EOF
  exit 1
}

DRY_RUN=0
FORCE_PREDICT=0
KEEP_PRED_AFFS=0
KEEP_PRED_CODE=0
DO_AGG=0
VI_GPU=0
SAMPLE_EXPLICIT=0
SPLIT_EXPLICIT=0
EVAL_BOTH_SPLITS=0
EXP_ID=""
CKPT=""
CKPTS=""
SAMPLE=""
SPLIT=""
DATASETS=""
STAGE=""
PCT=2
VI_OVERLAY="vi_th_0_3.toml"
EVAL_OVERLAY="eval_rm200.toml"
OVERLAYS=""
BASE_CONFIG="default_l40s.toml"
EXTRA_CONFIGS=()
DO_OVERRIDE=""
TIME_OVR=""
PART_OVR=""
CPUS_OVR=""
MEM_OVR=""
ACCOUNT="torch_pr_61_general"
DEFAULT_SAMPLE="completely/R22C03-20180918_66_J2"
SBATCH_DEPENDENCY=""
DEP_ID=""
DEP_TYPE=""
LAST_ID=""
NEEDS_DECODE=0

positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force-predict) FORCE_PREDICT=1; shift ;;
    --keep-pred-affs) KEEP_PRED_AFFS=1; shift ;;
    --keep-pred-code) KEEP_PRED_CODE=1; shift ;;
    --agg) DO_AGG=1; shift ;;
    --vi-gpu) VI_GPU=1; shift ;;
    --id)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      EXP_ID="$2"; shift 2 ;;
    --ckpt)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      CKPT="$2"; shift 2 ;;
    --ckpts)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      CKPTS="$2"; shift 2 ;;
    --sample)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      SAMPLE="$2"; SAMPLE_EXPLICIT=1; shift 2 ;;
    --split)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      SPLIT="$2"; SPLIT_EXPLICIT=1; shift 2 ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      # Normalize to '+' (SLURM --export is comma-separated; commas in values break).
      DATASETS="${2//,/+}"; shift 2 ;;
    --stage)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      STAGE="$2"; shift 2 ;;
    --pct)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      PCT="$2"; shift 2 ;;
    --vi)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      VI_OVERLAY="$2"; shift 2 ;;
    --eval)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      EVAL_OVERLAY="$2"; shift 2 ;;
    --overlays)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      OVERLAYS="$2"; shift 2 ;;
    --base)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      BASE_CONFIG="$(basename "$2")"; shift 2 ;;
    --config)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      EXTRA_CONFIGS+=("$2"); shift 2 ;;
    --do)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      DO_OVERRIDE="$2"; shift 2 ;;
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

if (( ${#positionals[@]} < 2 )); then
  usage
fi

MODE="${positionals[0]}"
PRESET="${positionals[1]}"

# shellcheck source=ppp_submit_common.sh
source "${SCRIPT_DIR}/ppp_submit_common.sh"
# shellcheck source=ppp_submit_test_lib.sh
source "${SCRIPT_DIR}/ppp_submit_test_lib.sh"

# Validate preset in this shell (not $(…)): unknown names must abort, not empty-export.
preset_row "${PRESET}" >/dev/null

OVERLAY="$(preset_overlay "${PRESET}")"
if [[ -z "${EXP_ID}" ]]; then
  EXP_ID="$(preset_exp_id "${PRESET}")"
fi
if [[ "${SAMPLE_EXPLICIT}" -eq 0 ]]; then
  SAMPLE="${DEFAULT_SAMPLE}"
fi
if [[ -n "${SAMPLE}" && "${SAMPLE}" != */* ]]; then
  echo "Error: --sample must be '{dataset}/{stem}' (got '${SAMPLE}'); flat stems are not supported" >&2
  exit 1
fi

NEEDS_DECODE="$(needs_decode_from_configs)"
if [[ "${NEEDS_DECODE}" != "0" && "${NEEDS_DECODE}" != "1" ]]; then
  echo "Error: needs_decode.py returned '${NEEDS_DECODE}' (expected 0 or 1)" >&2
  exit 1
fi

# Bare array with no --stage → all (preferred pipeline).
if [[ "${MODE}" == "array" && -z "${STAGE}" ]]; then
  STAGE="all"
fi

# --agg only meaningful for legacy GPU predict+label array.
if [[ "${DO_AGG}" -eq 1 && "${MODE}" == "array" && "${STAGE}" != "legacy" ]]; then
  echo "Warning: --agg only applies to --stage legacy (ignored for stage=${STAGE})" >&2
fi

case "${MODE}" in
  run) submit_run ;;
  evaluate) submit_evaluate ;;
  ckpt-sweep) submit_ckpt_sweep ;;
  array) submit_array_mode ;;
  vi-sweep) submit_vi_sweep ;;
  *)
    echo "Error: unknown mode '${MODE}'" >&2
    usage
    ;;
esac
