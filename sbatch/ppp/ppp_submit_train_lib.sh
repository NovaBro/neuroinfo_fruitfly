#!/bin/bash
# Sourced by ppp_submit_train.sh. Expects ppp_submit_common.sh already sourced.

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
    "STAGE_DATA=0" \
    "ARRAY_MODE=0" \
    "DISCARD_PARTIAL=0" \
    "RUN_PRED_METRICS=0" \
    "LABEL_RESUME=0")"
  # DATASETS via export_base; empty → ppp_job.sh derives it from the TOMLs.
  jid=$(sbatch_submit "ppp_${PRESET}_train" "${cpus}" "${wall}" "${mem}" "${part}" "gpu:1" "${export_csv}")
  echo "train=${jid} exp=${EXP_ID} overlay=${OVERLAY} datasets=${DATASETS:-$(datasets_from_configs train)}"
  LAST_ID="${jid}"
}
