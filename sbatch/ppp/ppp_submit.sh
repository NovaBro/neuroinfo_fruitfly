#!/bin/bash
# PatchPerPix consolidated submitter. Sets SLURM resources and exports env for
# sbatch/ppp/ppp_job.sh. From repo root:
#
#   ./sbatch/ppp/ppp_submit.sh train fmaps24
#   ./sbatch/ppp/ppp_submit.sh infer long8h
#   ./sbatch/ppp/ppp_submit.sh chain fmaps24 train infer
#   ./sbatch/ppp/ppp_submit.sh ckpt-sweep fmaps24 --ckpts 50000,60000,70000,80000
#   ./sbatch/ppp/ppp_submit.sh array fmaps24 --split val --ckpt 80000 --pct 2 --agg
#   ./sbatch/ppp/ppp_submit.sh label-array long8h --split val --ckpt 70000 --pct 2 --agg
#   ./sbatch/ppp/ppp_submit.sh vi-sweep long8h --ckpt 30000
#   ./sbatch/ppp/ppp_submit.sh evaluate fmaps24 --split val --ckpt 80000
#   ./sbatch/ppp/ppp_submit.sh --dry-run train fmaps24
#
# Inference outputs land under metrics/ppp/<exp>/{val,test}/ matching --split
# (FISBe completely/val or .../test). Existing per-experiment scripts remain
# valid until you validate and archive them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/ppp_job.sh"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ppp_submit.sh [--dry-run] <mode> <preset> [flags…]
  ppp_submit.sh [--dry-run] chain <preset> <mode> [<mode>…] [flags…]

Modes:
  train          Stage data + -d train
  infer          One sample: predict label evaluate (VI + eval_rm0)
  predict        One sample: -d predict (optional --ckpt)
  label          One sample: -d label [+evaluate via --do]
  evaluate       -d evaluate. --split val|test → that split under metrics/.../{val|test}/
                 + summary_<split>.csv; no --split → val then test, summary_val/test.csv
                 + consolidated summary.csv under test/evaluated/
  chain          afterok chain of modes (e.g. chain fmaps24 train infer)
  ckpt-sweep     For each ckpt: predict → label+evaluate (afterok)
  array          Per-volume predict+label+evaluate into matching val|test; optional --agg
  label-array    Per-volume label only (+ label_resume); optional --agg
  vi-sweep       afterok chain over --overlays (default vi_th_0_3..0_7)

Presets: 8h | long8h | fmaps24 | fmaps24_400k | fmaps24_400k_bs5_aug | fmaps24_400k_aug_cheap

Flags:
  --id EXP_ID              Override experiment id (default ppp_basic_<preset>)
  --ckpt N                 Checkpoint number
  --ckpts A,B,C            ckpt-sweep list (default 50000,60000,70000,80000)
  --sample NAME            Sample id (default R22C03-20180918_66_J2)
  --split val|test         FISBe split + output dir (array / evaluate / infer)
  --splits val,test        array/label-array: multiple splits
  --pct K                  Array concurrency %K (default 2)
  --agg                    After array: submit evaluate aggregate (afterok)
  --vi FILE.toml           VI overlay (default vi_th_0_3.toml)
  --eval FILE.toml         Eval overlay (default eval_rm0.toml)
  --overlays A.toml,B.toml vi-sweep list
  --config FILE.toml       Extra overlay (repeatable)
  --do TASKS               Override -d (space or + separated)
  --time HH:MM:SS          Override walltime
  --partition NAME         Override partition
  --cpus N                 Override cpus-per-task
  --mem SIZE               Override mem (e.g. 96g)
  --account NAME           Override account
  --force-predict          ckpt-sweep: always re-predict
  --keep-pred-affs         After evaluate, keep volumes/pred_affs (default: strip)
  --dry-run                Print sbatch lines; do not submit
EOF
  exit 1
}

# ---- defaults / flag state ----
DRY_RUN=0
FORCE_PREDICT=0
KEEP_PRED_AFFS=0
DO_AGG=0
SAMPLE_EXPLICIT=0
SPLIT_EXPLICIT=0
EVAL_BOTH_SPLITS=0
EXP_ID=""
CKPT=""
CKPTS=""
SAMPLE=""
SPLIT=""
SPLITS=""
PCT=2
VI_OVERLAY="vi_th_0_3.toml"
EVAL_OVERLAY="eval_rm0.toml"
OVERLAYS=""
EXTRA_CONFIGS=()
DO_OVERRIDE=""
TIME_OVR=""
PART_OVR=""
CPUS_OVR=""
MEM_OVR=""
ACCOUNT="torch_pr_61_general"
DEFAULT_SAMPLE="R22C03-20180918_66_J2"
DEP_ID=""
LAST_ID=""

positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force-predict) FORCE_PREDICT=1; shift ;;
    --keep-pred-affs) KEEP_PRED_AFFS=1; shift ;;
    --agg) DO_AGG=1; shift ;;
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
    --splits)
      [[ $# -ge 2 ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      SPLITS="$2"; shift 2 ;;
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
CHAIN_MODES=("${positionals[@]:2}")

# ---- preset resolution ----
preset_overlay() {
  case "$1" in
    8h) echo "basic_8h.toml" ;;
    long8h) echo "basic_long8h.toml" ;;
    fmaps24) echo "basic_fmaps24.toml" ;;
    fmaps24_400k) echo "basic_fmaps24_400k.toml" ;;
    fmaps24_400k_bs5_aug) echo "basic_fmaps24_400k_bs5_aug.toml" ;;
    fmaps24_400k_aug_cheap) echo "basic_fmaps24_400k_aug_cheap.toml" ;;
    *)
      echo "Error: unknown preset '$1' (8h|long8h|fmaps24|fmaps24_400k|fmaps24_400k_bs5_aug|fmaps24_400k_aug_cheap)" >&2
      exit 1
      ;;
  esac
}

preset_exp_id() {
  case "$1" in
    8h) echo "ppp_basic_8h" ;;
    long8h) echo "ppp_basic_long8h" ;;
    fmaps24) echo "ppp_basic_fmaps24" ;;
    fmaps24_400k) echo "ppp_basic_fmaps24_400k" ;;
    fmaps24_400k_bs5_aug) echo "ppp_basic_fmaps24_400k_bs5_aug" ;;
    fmaps24_400k_aug_cheap) echo "ppp_basic_fmaps24_400k_aug_cheap" ;;
  esac
}

preset_train_cpus() {
  case "$1" in
    fmaps24|fmaps24_400k|fmaps24_400k_aug_cheap) echo 32 ;;
    fmaps24_400k_bs5_aug) echo 16 ;;
    *) echo 16 ;;
  esac
}

preset_train_time() {
  case "$1" in
    fmaps24_400k|fmaps24_400k_bs5_aug|fmaps24_400k_aug_cheap) echo "48:00:00" ;;
    *) echo "08:00:00" ;;
  esac
}

preset_partition() {
  case "$1" in
    fmaps24_400k_bs5_aug) echo "h200_public" ;;
    *) echo "l40s_public" ;;
  esac
}

preset_train_mem() {
  case "$1" in
    fmaps24_400k_bs5_aug) echo "500g" ;;
    *) echo "192g" ;;
  esac
}

OVERLAY="$(preset_overlay "${PRESET}")"
if [[ -z "${EXP_ID}" ]]; then
  EXP_ID="$(preset_exp_id "${PRESET}")"
fi
if [[ "${SAMPLE_EXPLICIT}" -eq 0 ]]; then
  SAMPLE="${DEFAULT_SAMPLE}"
fi

# ---- helpers ----
configs_join() {
  # Join with '+' — SLURM --export cannot carry spaces.
  local out="" c
  for c in "$@"; do
    [[ -n "${c}" ]] || continue
    if [[ -n "${out}" ]]; then
      out+="+${c}"
    else
      out="${c}"
    fi
  done
  echo "${out}"
}

export_base() {
  local parts=(
    "ALL"
    "EXP_ID=${EXP_ID}"
    "KEEP_PRED_AFFS=${KEEP_PRED_AFFS}"
    "EVAL_BOTH_SPLITS=${EVAL_BOTH_SPLITS}"
  )
  local kv
  for kv in "$@"; do
    parts+=("${kv}")
  done
  local IFS=,
  echo "${parts[*]}"
}

sbatch_submit() {
  local job_name="$1" cpus="$2" wall="$3" mem="$4" part="$5" gres="$6" export_csv="$7"
  shift 7

  [[ -n "${TIME_OVR}" ]] && wall="${TIME_OVR}"
  [[ -n "${PART_OVR}" ]] && part="${PART_OVR}"
  [[ -n "${CPUS_OVR}" ]] && cpus="${CPUS_OVR}"
  [[ -n "${MEM_OVR}" ]] && mem="${MEM_OVR}"

  local args=(
    --parsable
    --account="${ACCOUNT}"
    --job-name="${job_name}"
    --cpus-per-task="${cpus}"
    --time="${wall}"
    --mem="${mem}"
    --partition="${part}"
    --output="sbatch/ppp/logs/${job_name}-%j.out"
    --error="sbatch/ppp/logs/${job_name}-%j.err"
    --export="${export_csv}"
  )
  if [[ -n "${gres}" ]]; then
    args+=(--gres="${gres}")
  fi
  if [[ -n "${DEP_ID}" ]]; then
    args+=(--dependency="afterok:${DEP_ID}")
  fi
  args+=("$@" "${JOB_SCRIPT}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY_RUN: sbatch ${args[*]}" >&2
    echo "DRY_RUN_EXPORT: ${export_csv}" >&2
    echo "dryrun_job"
    return 0
  fi
  sbatch "${args[@]}"
}

# ---- mode submitters ----
submit_train() {
  local cpus wall part mem cfgs export_csv jid
  cpus="$(preset_train_cpus "${PRESET}")"
  wall="$(preset_train_time "${PRESET}")"
  part="$(preset_partition "${PRESET}")"
  mem="$(preset_train_mem "${PRESET}")"
  cfgs="$(configs_join "${OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  export_csv="$(export_base \
    "DO_TASKS=train" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=1" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=0" \
    "LABEL_RESUME=0")"
  jid=$(sbatch_submit "ppp_${PRESET}_train" "${cpus}" "${wall}" "${mem}" "${part}" "gpu:1" "${export_csv}")
  echo "train=${jid} exp=${EXP_ID} overlay=${OVERLAY}"
  LAST_ID="${jid}"
}

submit_infer() {
  local do_tasks="${DO_OVERRIDE:-predict+label+evaluate}"
  local cfgs wall export_csv jid
  cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  wall="04:00:00"
  [[ "${PRESET}" == "long8h" || "${PRESET}" == "8h" ]] && wall="02:00:00"
  export_csv="$(export_base \
    "DO_TASKS=${do_tasks}" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=0" \
    "LABEL_RESUME=0" \
    "SAMPLE=${SAMPLE}" \
    "SPLIT=${SPLIT:-val}" \
    "CKPT=${CKPT}")"
  jid=$(sbatch_submit "ppp_${PRESET}_infer" "16" "${wall}" "96g" "l40s_public" "gpu:1" "${export_csv}")
  echo "infer=${jid} exp=${EXP_ID} sample=${SAMPLE} ckpt=${CKPT:-last}"
  LAST_ID="${jid}"
}

submit_predict() {
  local cfgs export_csv jid name
  cfgs="$(configs_join "${OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  export_csv="$(export_base \
    "DO_TASKS=predict" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=1" \
    "LABEL_RESUME=0" \
    "SAMPLE=${SAMPLE}" \
    "SPLIT=${SPLIT:-val}" \
    "CKPT=${CKPT}")"
  name="ppp_${PRESET}_predict"
  [[ -n "${CKPT}" ]] && name="ppp_${PRESET}_pred_${CKPT}"
  jid=$(sbatch_submit "${name}" "16" "02:00:00" "96g" "l40s_public" "gpu:1" "${export_csv}")
  echo "predict=${jid} exp=${EXP_ID} ckpt=${CKPT:-last} sample=${SAMPLE}"
  LAST_ID="${jid}"
}

submit_label() {
  local do_tasks="${DO_OVERRIDE:-label+evaluate}"
  local cfgs export_csv jid name
  cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  export_csv="$(export_base \
    "DO_TASKS=${do_tasks}" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=1" \
    "LABEL_RESUME=0" \
    "SAMPLE=${SAMPLE}" \
    "SPLIT=${SPLIT:-val}" \
    "CKPT=${CKPT}")"
  name="ppp_${PRESET}_label"
  [[ -n "${CKPT}" ]] && name="ppp_${PRESET}_vi_${CKPT}"
  jid=$(sbatch_submit "${name}" "16" "04:00:00" "96g" "l40s_public" "gpu:1" "${export_csv}")
  echo "label=${jid} exp=${EXP_ID} ckpt=${CKPT:-last} do=${do_tasks}"
  LAST_ID="${jid}"
}

submit_evaluate() {
  local cfgs export_csv jid sample_kv split_kv job_name
  cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"

  # Full-split evaluate: clear SAMPLE unless user passed --sample.
  if [[ "${SAMPLE_EXPLICIT}" -eq 0 ]]; then
    sample_kv="SAMPLE="
  else
    sample_kv="SAMPLE=${SAMPLE}"
  fi

  EVAL_BOTH_SPLITS=0
  if [[ -n "${SPLIT}" ]]; then
    # Explicit --split, or SPLIT set by array --agg for one fisbe split.
    split_kv="SPLIT=${SPLIT}"
    job_name="ppp_${PRESET}_eval_${SPLIT}"
  else
    # No split → evaluate val then test in one job; consolidate summary.csv.
    EVAL_BOTH_SPLITS=1
    split_kv="SPLIT="
    job_name="ppp_${PRESET}_eval_both"
  fi

  export_csv="$(export_base \
    "DO_TASKS=evaluate" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=0" \
    "LABEL_RESUME=0" \
    "${sample_kv}" \
    "${split_kv}" \
    "CKPT=${CKPT}")"
  jid=$(sbatch_submit "${job_name}" "16" "01:00:00" "64g" "l40s_public" "gpu:1" "${export_csv}")
  echo "evaluate=${jid} exp=${EXP_ID} split=${SPLIT:-both} both=${EVAL_BOTH_SPLITS} ckpt=${CKPT:-last}"
  LAST_ID="${jid}"
  EVAL_BOTH_SPLITS=0
}

submit_one_mode() {
  case "$1" in
    train) submit_train ;;
    infer) submit_infer ;;
    predict) submit_predict ;;
    label) submit_label ;;
    evaluate) submit_evaluate ;;
    *)
      echo "Error: unknown mode '$1'" >&2
      exit 1
      ;;
  esac
}

submit_chain() {
  if (( ${#CHAIN_MODES[@]} < 1 )); then
    echo "Error: chain requires at least one mode after preset" >&2
    usage
  fi
  DEP_ID=""
  local m
  for m in "${CHAIN_MODES[@]}"; do
    submit_one_mode "${m}"
    DEP_ID="${LAST_ID}"
    [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_PREV"
  done
  echo "chain done exp=${EXP_ID} modes=${CHAIN_MODES[*]}"
}

submit_ckpt_sweep() {
  local ckpt_list
  if [[ -n "${CKPTS}" ]]; then
    IFS=',' read -r -a ckpt_list <<< "${CKPTS}"
  else
    ckpt_list=(50000 60000 70000 80000)
  fi
  DEP_ID=""
  local ckpt need_predict proc_zarr
  for ckpt in "${ckpt_list[@]}"; do
    CKPT="${ckpt}"
    proc_zarr="metrics/ppp/${EXP_ID}/${SPLIT:-val}/processed/${ckpt}/${SAMPLE}.zarr"
    need_predict=1
    # Skip predict only when pred_affs is complete (numinst-only zarrs must re-predict).
    if [[ "${FORCE_PREDICT}" -ne 1 ]] \
      && [[ -d "${proc_zarr}/volumes/pred_affs" ]] \
      && python3 sbatch/ppp/check_prediction_complete.py -q "${proc_zarr}"; then
      need_predict=0
      echo "ckpt=${ckpt}: pred_affs complete, skip predict"
    fi
    if [[ "${need_predict}" -eq 1 ]]; then
      submit_predict
      DEP_ID="${LAST_ID}"
      [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_PREV"
    fi
    submit_label
    DEP_ID="${LAST_ID}"
    [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_PREV"
  done
  echo "ckpt_sweep done exp=${EXP_ID} ckpts=${ckpt_list[*]} vi=${VI_OVERLAY}"
}

count_zarrs() {
  find "${REPO_ROOT}/fisbe/completely/$1" -maxdepth 1 -name '*.zarr' 2>/dev/null | wc -l
}

submit_array_split() {
  local split="$1" do_tasks="$2" label_resume="$3" discard="$4"
  local n last arr_id cfgs export_csv wall job_name
  n=$(count_zarrs "${split}")
  if (( n < 1 )); then
    echo "Error: no .zarr under fisbe/completely/${split}" >&2
    exit 1
  fi
  last=$((n - 1))
  if [[ -z "${CKPT}" ]]; then
    echo "Error: array modes require --ckpt" >&2
    exit 1
  fi
  cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  export_csv="$(export_base \
    "DO_TASKS=${do_tasks}" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=1" \
    "DISCARD_PARTIAL=${discard}" \
    "RUN_PRED_METRICS=0" \
    "LABEL_RESUME=${label_resume}" \
    "SAMPLE=" \
    "SPLIT=${split}" \
    "CKPT=${CKPT}")"

  mkdir -p sbatch/ppp/logs/array
  wall="04:00:00"
  job_name="ppp_${PRESET}_arr_${split}"
  if [[ "${label_resume}" == "1" ]]; then
    wall="01:30:00"
    job_name="ppp_${PRESET}_label_${split}"
  fi

  local args=(
    --parsable
    --account="${ACCOUNT}"
    --job-name="${job_name}"
    --cpus-per-task="${CPUS_OVR:-16}"
    --time="${TIME_OVR:-${wall}}"
    --mem="${MEM_OVR:-96g}"
    --partition="${PART_OVR:-l40s_public}"
    --gres=gpu:1
    --array="0-${last}%${PCT}"
    --output="sbatch/ppp/logs/array/${PRESET}_${split}_%A_%a.out"
    --error="sbatch/ppp/logs/array/${PRESET}_${split}_%A_%a.err"
    --export="${export_csv}"
    "${JOB_SCRIPT}"
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY_RUN: sbatch ${args[*]}" >&2
    echo "DRY_RUN_EXPORT: ${export_csv}" >&2
    arr_id="dryrun_array_${split}"
  else
    arr_id=$(sbatch "${args[@]}")
  fi
  echo "${split}: n=${n} array=0-${last}%${PCT} array_job=${arr_id}"
  LAST_ID="${arr_id}"

  if [[ "${DO_AGG}" -eq 1 ]]; then
    DEP_ID="${arr_id}"
    [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_ARR"
    local saved_split="${SPLIT}"
    SPLIT="${split}"
    SAMPLE_EXPLICIT=0
    submit_evaluate
    SPLIT="${saved_split}"
    DEP_ID=""
  fi
}

submit_array() {
  local splits_list
  if [[ -n "${SPLITS}" ]]; then
    IFS=',' read -r -a splits_list <<< "${SPLITS}"
  elif [[ -n "${SPLIT}" ]]; then
    splits_list=("${SPLIT}")
  else
    splits_list=(val test)
  fi
  local sp
  for sp in "${splits_list[@]}"; do
    submit_array_split "${sp}" "predict+label+evaluate" "0" "1"
  done
  echo "array done exp=${EXP_ID} ckpt=${CKPT} vi=${VI_OVERLAY}"
}

submit_label_array() {
  local splits_list
  if [[ -n "${SPLITS}" ]]; then
    IFS=',' read -r -a splits_list <<< "${SPLITS}"
  elif [[ -n "${SPLIT}" ]]; then
    splits_list=("${SPLIT}")
  else
    splits_list=(val test)
  fi
  local sp
  for sp in "${splits_list[@]}"; do
    submit_array_split "${sp}" "label" "1" "0"
  done
  echo "label-array done exp=${EXP_ID} ckpt=${CKPT}"
}

submit_vi_sweep() {
  local ov_list
  if [[ -n "${OVERLAYS}" ]]; then
    IFS=',' read -r -a ov_list <<< "${OVERLAYS}"
  else
    ov_list=(vi_th_0_3.toml vi_th_0_4.toml vi_th_0_5.toml vi_th_0_6.toml vi_th_0_7.toml)
  fi
  [[ -n "${CKPT}" ]] || CKPT=30000
  DEP_ID=""
  local ov tag cfgs export_csv jid do_tasks
  do_tasks="${DO_OVERRIDE:-label+evaluate}"
  for ov in "${ov_list[@]}"; do
    tag="${ov%.toml}"
    cfgs="$(configs_join "${OVERLAY}" "${ov}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
    export_csv="$(export_base \
      "DO_TASKS=${do_tasks}" \
      "CONFIGS=${cfgs}" \
      "STAGE_DATA=0" \
      "ARRAY_MODE=0" \
      "DISCARD_PARTIAL=0" \
      "RUN_PRED_METRICS=0" \
      "LABEL_RESUME=0" \
      "SAMPLE=${SAMPLE}" \
      "SPLIT=${SPLIT:-val}" \
      "CKPT=${CKPT}")"
    jid=$(sbatch_submit "ppp_vi_${tag}" "16" "04:00:00" "96g" "l40s_public" "gpu:1" "${export_csv}")
    echo "${tag}=${jid}"
    DEP_ID="${jid}"
    [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_PREV"
  done
  echo "vi_sweep afterok chain exp=${EXP_ID} ckpt=${CKPT}"
}

# ---- dispatch ----
case "${MODE}" in
  train) submit_train ;;
  infer) submit_infer ;;
  predict) submit_predict ;;
  label) submit_label ;;
  evaluate) submit_evaluate ;;
  chain) submit_chain ;;
  ckpt-sweep) submit_ckpt_sweep ;;
  array) submit_array ;;
  label-array) submit_label_array ;;
  vi-sweep) submit_vi_sweep ;;
  *)
    echo "Error: unknown mode '${MODE}'" >&2
    usage
    ;;
esac
