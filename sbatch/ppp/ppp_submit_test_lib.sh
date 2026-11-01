#!/bin/bash
# Sourced by ppp_submit_test.sh. Expects ppp_submit_common.sh already sourced.

primary_split() {
  # First split from --split (comma-capable), default val. For single-sample jobs.
  local s="${SPLIT:-val}"
  echo "${s%%,*}"
}

do_tasks_has() {
  # Args: $1=needle $2=do_tasks (+ or space separated). Return 0 if present.
  local needle="$1" tasks=" $2 "
  tasks="${tasks//+/ }"
  tasks="${tasks//,/ }"
  [[ "${tasks}" == *" ${needle} "* ]]
}

# ---- mode submitters ----

submit_run() {
  # One-sample job driven by --do (default predict[+decode]+label+evaluate).
  local default_do="predict+label+evaluate"
  if [[ "${NEEDS_DECODE:-0}" == "1" ]]; then
    default_do="predict+decode+label+evaluate"
  fi
  local do_tasks="${DO_OVERRIDE:-${default_do}}"
  local split cfgs wall export_csv jid name run_pred=0
  split="$(primary_split)"

  if do_tasks_has label "${do_tasks}" || do_tasks_has evaluate "${do_tasks}"; then
    cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  else
    cfgs="$(configs_join "${OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  fi

  # Walltime / metrics: predict-only shorter; label or full infer longer.
  wall="04:00:00"
  if do_tasks_has predict "${do_tasks}" \
    && ! do_tasks_has label "${do_tasks}" \
    && ! do_tasks_has evaluate "${do_tasks}" \
    && ! do_tasks_has decode "${do_tasks}"; then
    wall="02:00:00"
    run_pred=1
  elif do_tasks_has decode "${do_tasks}" \
    && ! do_tasks_has label "${do_tasks}" \
    && ! do_tasks_has evaluate "${do_tasks}" \
    && ! do_tasks_has predict "${do_tasks}"; then
    wall="02:00:00"
  elif [[ "${PRESET}" == "basic_long8h" || "${PRESET}" == "basic_8h" ]]; then
    wall="02:00:00"
  fi
  if do_tasks_has predict "${do_tasks}" && ! do_tasks_has label "${do_tasks}"; then
    run_pred=1
  fi
  if do_tasks_has label "${do_tasks}" && ! do_tasks_has predict "${do_tasks}"; then
    run_pred=1
  fi

  export_csv="$(export_base \
    "DO_TASKS=${do_tasks}" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=${run_pred}" \
    "LABEL_RESUME=0" \
    "SAMPLE=${SAMPLE}" \
    "SPLIT=${split}" \
    "CKPT=${CKPT}")"

  name="ppp_${PRESET}_run"
  if do_tasks_has predict "${do_tasks}" \
    && do_tasks_has label "${do_tasks}"; then
    name="ppp_${PRESET}_infer"
  elif do_tasks_has predict "${do_tasks}" \
    && ! do_tasks_has label "${do_tasks}"; then
    name="ppp_${PRESET}_predict"
    [[ -n "${CKPT}" ]] && name="ppp_${PRESET}_pred_${CKPT}"
  elif do_tasks_has decode "${do_tasks}" \
    && ! do_tasks_has label "${do_tasks}" \
    && ! do_tasks_has predict "${do_tasks}"; then
    name="ppp_${PRESET}_decode"
    [[ -n "${CKPT}" ]] && name="ppp_${PRESET}_decode_${CKPT}"
  elif do_tasks_has label "${do_tasks}"; then
    name="ppp_${PRESET}_label"
    [[ -n "${CKPT}" ]] && name="ppp_${PRESET}_vi_${CKPT}"
  fi

  jid=$(sbatch_submit "${name}" "16" "${wall}" "96g" "l40s_public" "gpu:1" "${export_csv}")
  echo "run=${jid} exp=${EXP_ID} do=${do_tasks} sample=${SAMPLE} ckpt=${CKPT:-last} split=${split} needs_decode=${NEEDS_DECODE:-0}"
  LAST_ID="${jid}"
}

# ckpt-sweep helpers: fixed --do via submit_run.
submit_predict() {
  local saved="${DO_OVERRIDE}"
  DO_OVERRIDE="predict"
  submit_run
  DO_OVERRIDE="${saved}"
}

submit_decode() {
  local saved="${DO_OVERRIDE}"
  DO_OVERRIDE="decode"
  submit_run
  DO_OVERRIDE="${saved}"
}

submit_label() {
  local saved="${DO_OVERRIDE}"
  [[ -z "${DO_OVERRIDE}" ]] && DO_OVERRIDE="label+evaluate"
  submit_run
  DO_OVERRIDE="${saved}"
}

submit_evaluate() {
  # Aggregate -d evaluate over a split (or both when --split omitted or comma list).
  local cfgs export_csv jid sample_kv split_kv job_name
  cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"

  if [[ "${SAMPLE_EXPLICIT}" -eq 0 ]]; then
    sample_kv="SAMPLE="
  else
    sample_kv="SAMPLE=${SAMPLE}"
  fi

  EVAL_BOTH_SPLITS=0
  if [[ -z "${SPLIT}" || "${SPLIT}" == *","* ]]; then
    EVAL_BOTH_SPLITS=1
    split_kv="SPLIT="
    job_name="ppp_${PRESET}_eval_both"
  else
    split_kv="SPLIT=${SPLIT}"
    job_name="ppp_${PRESET}_eval_${SPLIT}"
  fi

  export_csv="$(export_base \
    "DO_TASKS=evaluate" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=0" \
    "LABEL_RESUME=0" \
    "NEED_GPU=0" \
    "${sample_kv}" \
    "${split_kv}" \
    "CKPT=${CKPT}")"
  jid=$(sbatch_submit "${job_name}" "16" "04:00:00" "112g" "cs,cpu_short" "" "${export_csv}")
  echo "evaluate=${jid} exp=${EXP_ID} split=${SPLIT:-both} both=${EVAL_BOTH_SPLITS} ckpt=${CKPT:-last}"
  LAST_ID="${jid}"
  EVAL_BOTH_SPLITS=0
}

submit_ckpt_sweep() {
  local ckpt_list
  if [[ -n "${CKPTS}" ]]; then
    IFS=',' read -r -a ckpt_list <<< "${CKPTS}"
  else
    ckpt_list=(50000 60000 70000 80000)
  fi
  DEP_ID=""
  local ckpt need_predict need_decode proc_zarr split
  split="$(primary_split)"
  for ckpt in "${ckpt_list[@]}"; do
    CKPT="${ckpt}"
    proc_zarr="metrics/ppp/${EXP_ID}/${split}/processed/${ckpt}/${SAMPLE}.zarr"
    need_predict=1
    need_decode=0
    if [[ "${NEEDS_DECODE:-0}" == "1" ]]; then
      need_decode=1
      if [[ "${FORCE_PREDICT}" -ne 1 ]] \
        && [[ -d "${proc_zarr}/volumes/pred_affs" ]] \
        && python3 sbatch/ppp/check_prediction_complete.py -q "${proc_zarr}"; then
        need_predict=0
        need_decode=0
        echo "ckpt=${ckpt}: pred_affs complete, skip predict+decode"
      elif [[ "${FORCE_PREDICT}" -ne 1 ]] \
        && [[ -d "${proc_zarr}/volumes/pred_code" ]] \
        && python3 sbatch/ppp/check_prediction_complete.py -q \
          --aff-key volumes/pred_code "${proc_zarr}"; then
        need_predict=0
        echo "ckpt=${ckpt}: pred_code complete, skip predict (still decode)"
      fi
    elif [[ "${FORCE_PREDICT}" -ne 1 ]] \
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
    if [[ "${need_decode}" -eq 1 ]]; then
      submit_decode
      DEP_ID="${LAST_ID}"
      [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_PREV"
    fi
    submit_label
    DEP_ID="${LAST_ID}"
    [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_PREV"
  done
  echo "ckpt_sweep done exp=${EXP_ID} ckpts=${ckpt_list[*]} vi=${VI_OVERLAY} needs_decode=${NEEDS_DECODE:-0}"
}

datasets_list() {
  # Echo dataset names for split $1 one per line: DATASETS (+ or comma
  # separated) if set, else from the TOMLs.
  local raw="${DATASETS:-$(datasets_from_configs "$1")}"
  raw="${raw//+/ }"
  raw="${raw//,/ }"
  local ds
  for ds in ${raw}; do
    ds="${ds// /}"
    [[ -n "${ds}" ]] || continue
    echo "${ds}"
  done
}

count_zarrs() {
  # Count samples for split $1 via ppp_datasets.py (honors max_test_samples).
  # Uses DATASETS override when set; else TOML *_datasets. Same first-N as
  # list_array_samples / run_ppp.list_dataset_samples.
  local split="$1"
  local args=("--split" "${split}" "--count" "--fisbe-root" "${REPO_ROOT}/fisbe")
  if [[ -n "${DATASETS}" ]]; then
    args+=(--datasets "${DATASETS}")
  fi
  args+=("${BASE_CONFIG}" "${OVERLAY}")
  if (( ${#EXTRA_CONFIGS[@]} > 0 )); then
    args+=("${EXTRA_CONFIGS[@]}")
  fi
  python3 "${SCRIPT_DIR}/ppp_datasets.py" "${args[@]}"
}

array_splits_list() {
  # Echo split names one per line. --split accepts comma lists; omit → val+test.
  if [[ -n "${SPLIT}" ]]; then
    local s
    IFS=',' read -r -a _splits_tmp <<< "${SPLIT}"
    for s in "${_splits_tmp[@]}"; do
      s="${s// /}"
      [[ -n "${s}" ]] || continue
      echo "${s}"
    done
  else
    echo "val"
    echo "test"
  fi
}

submit_array_split() {
  # Args: $1=split $2=DO_TASKS $3=LABEL_RESUME $4=DISCARD_PARTIAL
  #       $5=cfgs $6=wall $7=partition $8=gres $9=job_name $10=need_gpu
  local split="$1" do_tasks="$2" label_resume="$3" discard="$4"
  local cfgs="$5" wall="$6" part="$7" gres="$8" job_name="$9" need_gpu="${10}"
  local n last arr_id export_csv
  local ds
  ds="${DATASETS:-$(datasets_from_configs "${split}")}"
  n=$(count_zarrs "${split}")
  if (( n < 1 )); then
    echo "Error: no .zarr under fisbe/{${ds//+/,}}/${split}" >&2
    exit 1
  fi
  last=$((n - 1))
  if [[ -z "${CKPT}" ]]; then
    echo "Error: array modes require --ckpt" >&2
    exit 1
  fi
  export_csv="$(export_base \
    "DO_TASKS=${do_tasks}" \
    "CONFIGS=${cfgs}" \
    "STAGE_DATA=0" \
    "ARRAY_MODE=1" \
    "DISCARD_PARTIAL=${discard}" \
    "RUN_PRED_METRICS=0" \
    "LABEL_RESUME=${label_resume}" \
    "NEED_GPU=${need_gpu}" \
    "SAMPLE=" \
    "SPLIT=${split}" \
    "CKPT=${CKPT}")"

  mkdir -p sbatch/ppp/logs/array
  [[ -n "${PART_OVR}" ]] && part="${PART_OVR}"
  [[ -n "${TIME_OVR}" ]] && wall="${TIME_OVR}"

  local args=(
    --parsable
    --account="${ACCOUNT}"
    --job-name="${job_name}"
    --cpus-per-task="${CPUS_OVR:-16}"
    --time="${wall}"
    --mem="${MEM_OVR:-96g}"
    --partition="${part}"
    --array="0-${last}%${PCT}"
    --output="sbatch/ppp/logs/array/${PRESET}_${split}_%A_%a.out"
    --error="sbatch/ppp/logs/array/${PRESET}_${split}_%A_%a.err"
    --export="${export_csv}"
  )
  if [[ -n "${gres}" ]]; then
    args+=(--gres="${gres}")
  fi
  if [[ -n "${DEP_ID}" ]]; then
    args+=(--dependency="${DEP_TYPE:-afterok}:${DEP_ID}")
  elif [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
    args+=(--dependency="${SBATCH_DEPENDENCY}")
  fi
  args+=("${JOB_SCRIPT}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY_RUN: sbatch ${args[*]}" >&2
    echo "DRY_RUN_EXPORT: ${export_csv}" >&2
    arr_id="dryrun_array_${split}"
  else
    arr_id=$(sbatch "${args[@]}")
  fi
  echo "${split}: n=${n} datasets=${ds} array=0-${last}%${PCT} array_job=${arr_id} do=${do_tasks} part=${part} gres=${gres:-none}"
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

label_array_resources() {
  # Sets label_cfgs, wall, part, gres, need_gpu for --vi-gpu (single GPU label job).
  label_cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  wall="01:30:00"
  part="l40s_public"
  gres="gpu:1"
  need_gpu="1"
}

submit_label_split() {
  # Default array label: GPU patch-graph phase → CPU labelling phase (+ evaluate).
  # Blockwise vote_instances needs CUDA only for the per-/inter-block patch graph
  # (cached in <instanced>/<sample>.zarr); affGraphToInstances is CPU-only and
  # would idle the GPU (Greene low-util cancels). Task i of the CPU array waits
  # only for GPU task i (aftercorr).
  # Args: $1=split $2=dependency job id for the GPU phase ("" = none, afterok).
  local sp="$1" dep="$2" base_cfgs gpu_cfgs cpu_cfgs saved_agg="${DO_AGG}"
  base_cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
  gpu_cfgs="$(configs_join "${base_cfgs}" "vi_split_gpu.toml")"
  cpu_cfgs="$(configs_join "${base_cfgs}" "vi_split_cpu.toml")"

  DO_AGG=0
  DEP_ID="${dep}"
  DEP_TYPE="afterok"
  submit_array_split "${sp}" "label" "1" "0" \
    "${gpu_cfgs}" "04:00:00" "l40s_public" "gpu:1" "ppp_${PRESET}_labgraph_${sp}" "1"

  DO_AGG="${saved_agg}"
  DEP_ID="${LAST_ID}"
  [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_LABGRAPH_${sp}"
  DEP_TYPE="aftercorr"
  submit_array_split "${sp}" "label" "1" "0" \
    "${cpu_cfgs}" "04:00:00" "cs,cpu_short" "" "ppp_${PRESET}_label_${sp}" "0"
  DEP_TYPE=""
  DEP_ID=""
}

submit_array_mode() {
  # Unified array: --stage all|predict|decode|label|legacy (default all).
  local stage="${STAGE:-all}"
  local splits_list sp pred_cfgs decode_cfgs label_cfgs wall part gres need_gpu saved_agg
  local pred_id decode_id label_dep
  case "${stage}" in
    all|predict|decode|label|legacy) ;;
    *)
      echo "Error: --stage must be all|predict|decode|label|legacy (got '${stage}')" >&2
      exit 1
      ;;
  esac
  if [[ "${stage}" == "decode" && "${NEEDS_DECODE:-0}" != "1" ]]; then
    echo "Error: --stage decode requires train_code configs (NEEDS_DECODE=0 for this preset/base)" >&2
    exit 1
  fi
  mapfile -t splits_list < <(array_splits_list)
  saved_agg="${DO_AGG}"

  case "${stage}" in
    predict)
      pred_cfgs="$(configs_join "${OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
      DO_AGG=0
      for sp in "${splits_list[@]}"; do
        DEP_ID=""
        submit_array_split "${sp}" "predict" "0" "1" \
          "${pred_cfgs}" "02:00:00" "l40s_public" "gpu:1" "ppp_${PRESET}_pred_${sp}" "1"
      done
      ;;
    decode)
      decode_cfgs="$(configs_join "${OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
      DO_AGG=0
      for sp in "${splits_list[@]}"; do
        DEP_ID=""
        submit_array_split "${sp}" "decode" "0" "0" \
          "${decode_cfgs}" "02:00:00" "l40s_public" "gpu:1" "ppp_${PRESET}_decode_${sp}" "1"
      done
      ;;
    label)
      DO_AGG=1
      for sp in "${splits_list[@]}"; do
        if [[ "${VI_GPU}" -eq 1 ]]; then
          label_array_resources
          DEP_ID=""
          submit_array_split "${sp}" "label" "1" "0" \
            "${label_cfgs}" "${wall}" "${part}" "${gres}" "ppp_${PRESET}_label_${sp}" "${need_gpu}"
        else
          submit_label_split "${sp}" ""
        fi
      done
      ;;
    all)
      pred_cfgs="$(configs_join "${OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
      decode_cfgs="${pred_cfgs}"
      for sp in "${splits_list[@]}"; do
        DO_AGG=0
        DEP_ID=""
        submit_array_split "${sp}" "predict" "0" "1" \
          "${pred_cfgs}" "02:00:00" "l40s_public" "gpu:1" "ppp_${PRESET}_pred_${sp}" "1"
        pred_id="${LAST_ID}"
        [[ "${DRY_RUN}" -eq 1 ]] && pred_id="DRYRUN_PRED_${sp}"
        label_dep="${pred_id}"
        if [[ "${NEEDS_DECODE:-0}" == "1" ]]; then
          DEP_ID="${pred_id}"
          DEP_TYPE="afterok"
          submit_array_split "${sp}" "decode" "0" "0" \
            "${decode_cfgs}" "02:00:00" "l40s_public" "gpu:1" "ppp_${PRESET}_decode_${sp}" "1"
          decode_id="${LAST_ID}"
          [[ "${DRY_RUN}" -eq 1 ]] && decode_id="DRYRUN_DECODE_${sp}"
          label_dep="${decode_id}"
        fi
        DO_AGG=1
        if [[ "${VI_GPU}" -eq 1 ]]; then
          label_array_resources
          DEP_ID="${label_dep}"
          DEP_TYPE="afterok"
          submit_array_split "${sp}" "label" "1" "0" \
            "${label_cfgs}" "${wall}" "${part}" "${gres}" "ppp_${PRESET}_label_${sp}" "${need_gpu}"
        else
          submit_label_split "${sp}" "${label_dep}"
        fi
      done
      DEP_ID=""
      DEP_TYPE=""
      ;;
    legacy)
      # GPU predict[+decode]+label in one task; --agg optionally chains evaluate.
      local cfgs do_legacy="predict+label"
      if [[ "${NEEDS_DECODE:-0}" == "1" ]]; then
        do_legacy="predict+decode+label"
      fi
      cfgs="$(configs_join "${OVERLAY}" "${VI_OVERLAY}" "${EVAL_OVERLAY}" "${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}")"
      for sp in "${splits_list[@]}"; do
        submit_array_split "${sp}" "${do_legacy}" "0" "1" \
          "${cfgs}" "04:00:00" "l40s_public" "gpu:1" "ppp_${PRESET}_arr_${sp}" "1"
      done
      ;;
  esac

  DO_AGG="${saved_agg}"
  echo "array done stage=${stage} exp=${EXP_ID} ckpt=${CKPT} vi=${VI_OVERLAY} vi_gpu=${VI_GPU} needs_decode=${NEEDS_DECODE:-0}"
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
  local ov tag cfgs export_csv jid do_tasks split
  do_tasks="${DO_OVERRIDE:-label+evaluate}"
  split="$(primary_split)"
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
      "SPLIT=${split}" \
      "CKPT=${CKPT}")"
    jid=$(sbatch_submit "ppp_vi_${tag}" "16" "04:00:00" "96g" "l40s_public" "gpu:1" "${export_csv}")
    echo "${tag}=${jid}"
    DEP_ID="${jid}"
    [[ "${DRY_RUN}" -eq 1 ]] && DEP_ID="DRYRUN_PREV"
  done
  echo "vi_sweep afterok chain exp=${EXP_ID} ckpt=${CKPT}"
}
