#!/bin/bash
# Sourced by ppp_submit_train.sh and ppp_submit_test.sh.
# Presets, config helpers, and the sbatch wrapper. Expects SCRIPT_DIR,
# REPO_ROOT, JOB_SCRIPT, and CLI globals already set.

# ---- presets (name|overlay|exp_id|train_cpus|train_mem|train_time|partition) ----
PRESET_TABLE=(
  "basic_8h|basic_8h.toml|ppp_basic_8h|16|192g|08:00:00|l40s_public"
  "basic_long8h|basic_long8h.toml|ppp_basic_long8h|16|192g|08:00:00|l40s_public"
  "basic_fmaps24|basic_fmaps24.toml|ppp_basic_fmaps24|32|192g|08:00:00|l40s_public"
  "basic_fmaps24_400k|basic_fmaps24_400k.toml|ppp_basic_fmaps24_400k|32|192g|48:00:00|l40s_public"
  "basic_fmaps24_400k_bs5_aug|basic_fmaps24_400k_bs5_aug.toml|ppp_basic_fmaps24_400k_bs5_aug|16|500g|48:00:00|h200_public"
  "basic_fmaps24_400k_aug_cheap|basic_fmaps24_400k_aug_cheap.toml|ppp_basic_fmaps24_400k_aug_cheap|32|192g|48:00:00|l40s_public"
  "basic-ppp-noenc|basic-ppp-noenc.toml|basic-ppp-noenc|32|250g|08:00:00|l40s_public"
  "basic-ppp-noenc-cp|basic-ppp-noenc-cp.toml|basic-ppp-noenc-cp|32|250g|08:00:00|l40s_public"
  "basic-ppp-noenc-cps|basic-ppp-noenc-cps.toml|basic-ppp-noenc-cps|32|250g|08:00:00|l40s_public"
  "basic-ppp-decoder|basic-ppp-decoder.toml|basic-ppp-decoder|32|250g|08:00:00|l40s_public"
  "base1-noenc-c|base1-noenc-c.toml|base1-noenc-c|32|250g|08:00:00|l40s_public"
  # "base1-noenc-c|base1-noenc-c.toml|base1-noenc-c|16|500g|08:00:00|h200_public"
  "base1-decoder-c|base1-decoder-c.toml|base1-decoder-c|32|250g|08:00:00|l40s_public"
  # "base1-decoder-c|base1-decoder-c.toml|base1-decoder-c|16|500g|08:00:00|h200_public"
  "our-noenc-cs|our-noenc-cs.toml|our-noenc-cs|32|250g|08:00:00|l40s_public"
  # "our-noenc-cs|our-noenc-cs.toml|our-noenc-cs|16|500g|08:00:00|h200_public"
  # "our-decoder-c1k_v1|our-decoder-c1k_v1.toml|our-decoder-c1k_v1|32|500g|08:00:00|l40s_public"
  "our-decoder-c1k_v1|our-decoder-c1k_v1.toml|our-decoder-c1k_v1|16|500g|08:00:00|h200_public"
  # "our-decoder-c1k_v1_u8|our-decoder-c1k_v1_u8.toml|our-decoder-c1k_v1_u8|32|250g|08:00:00|l40s_public"
  "our-decoder-c1k_v1_u8|our-decoder-c1k_v1_u8.toml|our-decoder-c1k_v1_u8|16|500g|08:00:00|h200_public"
  "our-noenc-c1k_v1|our-noenc-c1k_v1.toml|our-noenc-c1k_v1|32|250g|08:00:00|l40s_public"
  # "our-noenc-c1k_v1|our-noenc-c1k_v1.toml|our-noenc-c1k_v1|16|500g|08:00:00|h200_public"
  "our-noenc-c1k_v1_u8|our-noenc-c1k_v1_u8.toml|our-noenc-c1k_v1_u8|32|250g|08:00:00|l40s_public"
  # "our-noenc-c1k_v1_u8|our-noenc-c1k_v1_u8.toml|our-noenc-c1k_v1_u8|16|500g|08:00:00|h200_public"
)

preset_row() {
  # Args: $1 = preset name. Echo matching PRESET_TABLE row or exit 1.
  local name="$1" row
  for row in "${PRESET_TABLE[@]}"; do
    if [[ "${row%%|*}" == "${name}" ]]; then
      echo "${row}"
      return 0
    fi
  done
  echo "Error: unknown preset '${name}' (basic_8h|basic_long8h|basic_fmaps24|basic_fmaps24_400k|basic_fmaps24_400k_bs5_aug|basic_fmaps24_400k_aug_cheap|basic-ppp-noenc|basic-ppp-noenc-cp|basic-ppp-noenc-cps|basic-ppp-decoder)" >&2
  exit 1
}

preset_field() {
  # Args: $1 = preset name, $2 = 1-based field index (1=name … 7=partition).
  local row="$1" idx="$2" f1 f2 f3 f4 f5 f6 f7
  row="$(preset_row "$row")"
  IFS='|' read -r f1 f2 f3 f4 f5 f6 f7 <<< "${row}"
  case "${idx}" in
    1) echo "${f1}" ;;
    2) echo "${f2}" ;;
    3) echo "${f3}" ;;
    4) echo "${f4}" ;;
    5) echo "${f5}" ;;
    6) echo "${f6}" ;;
    7) echo "${f7}" ;;
    *) echo "Error: preset_field index ${idx}" >&2; exit 1 ;;
  esac
}

preset_overlay() { preset_field "$1" 2; }
preset_exp_id() { preset_field "$1" 3; }
preset_train_cpus() { preset_field "$1" 4; }
preset_train_mem() { preset_field "$1" 5; }
preset_train_time() { preset_field "$1" 6; }
preset_partition() { preset_field "$1" 7; }

# ---- helpers ----

configs_join() {
  # Join TOML basenames with '+' into a single CONFIGS= value.
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

datasets_from_configs() {
  # Echo '+'-joined dataset names for split $1 from merged BASE+overlay+extras.
  local args=("${BASE_CONFIG}" "${OVERLAY}")
  if (( ${#EXTRA_CONFIGS[@]} > 0 )); then
    args+=("${EXTRA_CONFIGS[@]}")
  fi
  python3 "${SCRIPT_DIR}/ppp_datasets.py" --split "$1" "${args[@]}"
}

needs_decode_from_configs() {
  # Echo 1/0: merged BASE+overlay+extras have train_code (ppp+dec path).
  local args=("${BASE_CONFIG}" "${OVERLAY}")
  if (( ${#EXTRA_CONFIGS[@]} > 0 )); then
    args+=("${EXTRA_CONFIGS[@]}")
  fi
  python3 "${SCRIPT_DIR}/needs_decode.py" "${args[@]}"
}

export_base() {
  # Build the --export CSV passed to sbatch / read by ppp_job.sh.
  local parts=(
    "ALL"
    "EXP_ID=${EXP_ID}"
    "KEEP_PRED_AFFS=${KEEP_PRED_AFFS:-0}"
    "KEEP_PRED_CODE=${KEEP_PRED_CODE:-0}"
    "EVAL_BOTH_SPLITS=${EVAL_BOTH_SPLITS}"
    "DEFAULT_CONFIG=${BASE_CONFIG}"
    "DATASETS=${DATASETS}"
    "NEEDS_DECODE=${NEEDS_DECODE:-0}"
  )
  local kv
  for kv in "$@"; do
    parts+=("${kv}")
  done
  local IFS=,
  echo "${parts[*]}"
}

sbatch_submit() {
  # Args: $1=job_name $2=cpus $3=wall $4=mem $5=partition $6=gres ("" = none)
  #       $7=export_csv  $8…=extra sbatch args before the job script.
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
  elif [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
    args+=(--dependency="${SBATCH_DEPENDENCY}")
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
