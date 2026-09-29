#!/bin/bash
# PatchPerPix generic SLURM job body. Resources and job-name come from
# ppp_submit.sh (or sbatch CLI). Behavior is driven by exported env vars.
#
# Prefer: ./sbatch/ppp/ppp_submit.sh <mode> <preset> …
# Direct:  sbatch --export=ALL,EXP_ID=…,DO_TASKS=…,CONFIGS=…,STAGE_DATA=1 \
#            --job-name=… --cpus-per-task=… --time=… --mem=… --gres=gpu:1 \
#            --partition=l40s_public sbatch/ppp/ppp_job.sh
#
# Env (required unless noted):
#   EXP_ID          -id under metrics/ppp/
#   DO_TASKS        space- or +-separated -d tasks (train|predict|label|evaluate|…)
#   CONFIGS         +-separated overlay basenames (e.g. basic_fmaps24.toml+vi_th_0_3.toml)
#                   (spaces break SLURM --export; '+' matches DO_TASKS convention)
# Optional:
#   CKPT            --checkpoint N  (empty + non-train → --test-checkpoint last)
#   SAMPLE          --sample NAME   (single-sample modes; ignored when ARRAY_MODE=1)
#   SPLIT           val|test → --test-data fisbe/completely/$SPLIT
#   STAGE_DATA      1 → rsync train/val to $SLURM_TMPDIR
#   ARRAY_MODE      1 → pick sample by SLURM_ARRAY_TASK_ID from SPLIT
#   DISCARD_PARTIAL 1 → drop incomplete volumes/pred_affs before predict (keeps pred_numinst)
#   RUN_PRED_METRICS 1 → run diagnose_ppp_pred_metrics.py after predict / before label
#   LABEL_RESUME    1 → append label_resume.toml
#   KEEP_PRED_AFFS  1 → after evaluate, do not strip volumes/pred_affs (default: strip
#                   all but keep-sample R22C03-20180918_66_J2.zarr via strip_pred_affs.py)
#   EVAL_BOTH_SPLITS 1 → evaluate val then test; write summary_val/test.csv and merge summary.csv
#   PPP_ROOT        metrics root relative to experiments/ (default ../../metrics/ppp)
#   DEFAULT_CONFIG  always prepended (default default_l40s.toml)
#   SETUP_DIR       config dir under experiments/ (default flylight/setups/setup01)

#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="/scratch/wmz2007/neuroinfo_fruitfly"
cd "${REPO_ROOT}"

EXP_ID="${EXP_ID:?Error: set EXP_ID}"
DO_TASKS="${DO_TASKS:?Error: set DO_TASKS}"
CONFIGS="${CONFIGS:?Error: set CONFIGS}"

CKPT="${CKPT:-}"
SAMPLE="${SAMPLE:-}"
SPLIT="${SPLIT:-}"
STAGE_DATA="${STAGE_DATA:-0}"
ARRAY_MODE="${ARRAY_MODE:-0}"
DISCARD_PARTIAL="${DISCARD_PARTIAL:-0}"
RUN_PRED_METRICS="${RUN_PRED_METRICS:-0}"
LABEL_RESUME="${LABEL_RESUME:-0}"
KEEP_PRED_AFFS="${KEEP_PRED_AFFS:-0}"
EVAL_BOTH_SPLITS="${EVAL_BOTH_SPLITS:-0}"
PPP_ROOT="${PPP_ROOT:-../../metrics/ppp}"
DEFAULT_CONFIG="${DEFAULT_CONFIG:-default_l40s.toml}"
SETUP_DIR="${SETUP_DIR:-flylight/setups/setup01}"
FISBE_ROOT="${FISBE_ROOT:-${REPO_ROOT}/fisbe/completely}"
DEFAULT_SAMPLE="${DEFAULT_SAMPLE:-R22C03-20180918_66_J2}"
KEEP_AFFS_SAMPLE="${KEEP_AFFS_SAMPLE:-R22C03-20180918_66_J2.zarr,JRC_SS04989-20160318_24_A2.zarr}"

# Route inference outputs to val/ or test/ to match --test-data (run_ppp.py).
if [[ "${SPLIT}" == "val" || "${SPLIT}" == "test" ]]; then
  PROCESSED_SPLIT="${SPLIT}"
else
  PROCESSED_SPLIT="val"
fi

processed_sample_zarr() {
  local ckpt="$1" sample="$2"
  echo "metrics/ppp/${EXP_ID}/${PROCESSED_SPLIT}/processed/${ckpt}/${sample}.zarr"
}

resolve_ckpt_for_strip() {
  # Prefer explicit CKPT; else newest numeric processed dir containing SAMPLE (or any).
  if [[ -n "${CKPT}" ]]; then
    echo "${CKPT}"
    return 0
  fi
  local base="metrics/ppp/${EXP_ID}/${PROCESSED_SPLIT}/processed"
  [[ -d "${base}" ]] || return 1
  local best="" d
  for d in $(find "${base}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n); do
    if [[ -n "${SAMPLE}" ]]; then
      [[ -d "${base}/${d}/${SAMPLE}.zarr" ]] || continue
    fi
    best="${d}"
  done
  [[ -n "${best}" ]] || return 1
  echo "${best}"
}

# Normalize DO_TASKS: allow '+' or commas (commas unsafe in SLURM --export).
DO_TASKS_NORM="${DO_TASKS//+/ }"
DO_TASKS_NORM="${DO_TASKS_NORM//,/ }"
# shellcheck disable=SC2206
DO_ARR=(${DO_TASKS_NORM})
if (( ${#DO_ARR[@]} < 1 )); then
  echo "Error: DO_TASKS empty after normalize: '${DO_TASKS}'" >&2
  exit 1
fi

has_task() {
  local t
  for t in "${DO_ARR[@]}"; do
    [[ "${t}" == "$1" ]] && return 0
  done
  return 1
}

IS_TRAIN=0
has_task train && IS_TRAIN=1

# Resolve sample for array tasks.
if [[ "${ARRAY_MODE}" == "1" ]]; then
  if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
    echo "Error: ARRAY_MODE=1 requires SPLIT=val or SPLIT=test" >&2
    exit 1
  fi
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Error: ARRAY_MODE=1 requires a SLURM array job" >&2
    exit 1
  fi
  DATA_DIR="${FISBE_ROOT}/${SPLIT}"
  mapfile -t SAMPLES < <(find "${DATA_DIR}" -maxdepth 1 -name '*.zarr' -printf '%f\n' \
    | sed 's/\.zarr$//' | sort)
  if (( SLURM_ARRAY_TASK_ID >= ${#SAMPLES[@]} )); then
    echo "task ${SLURM_ARRAY_TASK_ID} >= ${#SAMPLES[@]} samples; nothing to do"
    exit 0
  fi
  SAMPLE="${SAMPLES[${SLURM_ARRAY_TASK_ID}]}"
  echo "[array ${SLURM_ARRAY_TASK_ID}] split=${SPLIT} sample=${SAMPLE}"
fi

# Default single-sample for predict/label when SAMPLE unset (not array, not eval-only).
if [[ "${IS_TRAIN}" -eq 0 && -z "${SAMPLE}" && "${ARRAY_MODE}" != "1" ]]; then
  if has_task predict || has_task label; then
    SAMPLE="${DEFAULT_SAMPLE}"
  fi
fi

# test-data path
TEST_DATA=""
if [[ "${IS_TRAIN}" -eq 0 ]]; then
  if [[ -n "${SPLIT}" ]]; then
    TEST_DATA="${FISBE_ROOT}/${SPLIT}"
  else
    TEST_DATA="${FISBE_ROOT}/val"
  fi
fi

# Build config path list (always prepend default_l40s unless already first).
# CONFIGS uses '+' (SLURM --export cannot carry spaces).
CONFIGS_NORM="${CONFIGS//+/ }"
CONFIGS_NORM="${CONFIGS_NORM//,/ }"
# shellcheck disable=SC2206
CFG_NAMES=(${CONFIGS_NORM})
if [[ "${LABEL_RESUME}" == "1" ]]; then
  CFG_NAMES+=("label_resume.toml")
fi
CONFIG_ARGS=()
have_default=0
for c in "${CFG_NAMES[@]}"; do
  [[ "${c}" == "${DEFAULT_CONFIG}" ]] && have_default=1
done
if [[ "${have_default}" -eq 0 ]]; then
  CONFIG_ARGS+=(--config "${SETUP_DIR}/${DEFAULT_CONFIG}")
fi
for c in "${CFG_NAMES[@]}"; do
  CONFIG_ARGS+=(--config "${SETUP_DIR}/${c}")
done

# Checkpoint args
CKPT_ARGS=()
if [[ "${IS_TRAIN}" -eq 0 ]]; then
  if [[ -n "${CKPT}" ]]; then
    CKPT_ARGS+=(--checkpoint "${CKPT}")
  else
    CKPT_ARGS+=(--test-checkpoint last)
  fi
fi

SAMPLE_ARGS=()
if [[ -n "${SAMPLE}" ]]; then
  SAMPLE_ARGS+=(--sample "${SAMPLE}")
fi

echo "ppp_job exp=${EXP_ID} do=${DO_ARR[*]} configs=${CFG_NAMES[*]} ckpt=${CKPT:-last} sample=${SAMPLE:-none} split=${SPLIT:-none} stage=${STAGE_DATA} array=${ARRAY_MODE} job=${SLURM_JOB_ID:-local}"

# ---- GPU logger (skip on pure CPU evaluate if no nvidia-smi / no GPU) ----
GPU_LOGGER_PID=
cleanup() {
  if [[ -n "${GPU_LOGGER_PID}" ]]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

LOG_TAG="${EXP_ID}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  LOG_TAG="${EXP_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LOG_TAG="${EXP_ID}_${SLURM_JOB_ID}"
fi
mkdir -p gpu_log/ppp
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 3 \
    > "gpu_log/ppp/gpu_usage_log_${LOG_TAG}.csv" &
  GPU_LOGGER_PID=$!
fi

# ---- Optional train/val staging ----
STAGE_ROOT=""
BIND_ARGS=()
TRAIN_ARGS=()
if [[ "${STAGE_DATA}" == "1" ]]; then
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "Error: SLURM_TMPDIR is unset; cannot stage training data" >&2
    exit 1
  fi
  STAGE_ROOT="${SLURM_TMPDIR}/ppp_data"
  echo "Staging train/val -> ${STAGE_ROOT}"
  df -h "${SLURM_TMPDIR}"
  STAGE_T0=$(date +%s)
  mkdir -p "${STAGE_ROOT}"
  rsync -a --delete "${FISBE_ROOT}/train/" "${STAGE_ROOT}/train/"
  rsync -a --delete "${FISBE_ROOT}/val/" "${STAGE_ROOT}/val/"
  STAGE_T1=$(date +%s)
  echo "Staging finished in $((STAGE_T1 - STAGE_T0))s"
  df -h "${SLURM_TMPDIR}"
  BIND_ARGS+=(--bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
  TRAIN_ARGS+=(--train-data "${STAGE_ROOT}/train" --val-data "${STAGE_ROOT}/val")
fi

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="env/ppp.ext3:ro"

run_singularity() {
  # Runs a bash -c string inside the ppp overlay.
  local inner="$1"
  singularity exec --nv \
    --overlay "${OVERLAY}" \
    "${BIND_ARGS[@]}" \
    "${SIF}" \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; ${inner}"
}

# Discard incomplete pred_affs before predict (array / resume).
# Keep pred_numinst: only remove volumes/pred_affs (or the whole zarr if nothing useful left).
if [[ "${DISCARD_PARTIAL}" == "1" && -n "${SAMPLE}" && -n "${CKPT}" ]]; then
  SAMPLE_ZARR="$(processed_sample_zarr "${CKPT}" "${SAMPLE}")"
  if [[ -d "${SAMPLE_ZARR}" ]] && ! python3 sbatch/ppp/check_prediction_complete.py -q "${SAMPLE_ZARR}"; then
    AFF_DIR="${SAMPLE_ZARR}/volumes/pred_affs"
    NUMINST_DIR="${SAMPLE_ZARR}/volumes/pred_numinst"
    if [[ -d "${AFF_DIR}" ]]; then
      echo "discarding incomplete pred_affs: ${AFF_DIR}"
      rm -rf "${AFF_DIR}"
    fi
    if [[ ! -d "${NUMINST_DIR}" && ! -d "${AFF_DIR}" ]]; then
      echo "discarding empty prediction zarr: ${SAMPLE_ZARR}"
      rm -rf "${SAMPLE_ZARR}"
    fi
  fi
fi

# Optional FG/numinst diagnostics on processed zarr.
run_pred_metrics_if_requested() {
  if [[ "${RUN_PRED_METRICS}" != "1" || -z "${SAMPLE}" || -z "${CKPT}" ]]; then
    return 0
  fi
  local zarr
  zarr="$(processed_sample_zarr "${CKPT}" "${SAMPLE}")"
  if [[ -d "${zarr}" ]]; then
    run_singularity "python -u sbatch/ppp/diagnose_ppp_pred_metrics.py --pred-zarr '${zarr}'"
  fi
}

if has_task predict && [[ "${RUN_PRED_METRICS}" == "1" ]]; then
  : # metrics after predict
elif has_task label && [[ "${RUN_PRED_METRICS}" == "1" ]] && ! has_task predict; then
  run_pred_metrics_if_requested
fi

# Require complete pred_affs for label-only tasks (array or single-sample).
if has_task label && ! has_task predict; then
  if [[ -z "${CKPT}" ]]; then
    echo "Error: label without predict requires CKPT" >&2
    exit 1
  fi
  if [[ -z "${SAMPLE}" ]]; then
    echo "Error: label without predict requires SAMPLE" >&2
    exit 1
  fi
  PRED_ZARR="$(processed_sample_zarr "${CKPT}" "${SAMPLE}")"
  if [[ ! -d "${PRED_ZARR}/volumes/pred_affs" ]]; then
    echo "Error: missing pred_affs for label: ${PRED_ZARR}/volumes/pred_affs" >&2
    exit 1
  fi
  if ! python3 sbatch/ppp/check_prediction_complete.py -q "${PRED_ZARR}"; then
    echo "Error: incomplete pred_affs for label: ${PRED_ZARR}" >&2
    exit 1
  fi
fi

# Build -d args and optional --test-data.
DO_ARGS=()
for t in "${DO_ARR[@]}"; do
  DO_ARGS+=("${t}")
done

CONFIG_STR=""
for a in "${CONFIG_ARGS[@]}"; do
  CONFIG_STR+=" ${a}"
done
TRAIN_STR=""
for a in "${TRAIN_ARGS[@]}"; do
  TRAIN_STR+=" ${a}"
done
CKPT_STR=""
for a in "${CKPT_ARGS[@]}"; do
  CKPT_STR+=" ${a}"
done
SAMPLE_STR=""
for a in "${SAMPLE_ARGS[@]}"; do
  SAMPLE_STR+=" ${a}"
done
DO_STR=""
for a in "${DO_ARGS[@]}"; do
  DO_STR+=" ${a}"
done

run_ppp_once() {
  # Args: optional --test-data path as $1 (empty → omit).
  local test_data="${1:-}"
  local test_str=""
  if [[ -n "${test_data}" ]]; then
    test_str=" --test-data ${test_data}"
  fi
  run_singularity \
    "cd PatchPerPix/experiments/; \
     env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python3 -u run_ppp.py --setup setup01 \
       ${CONFIG_STR} \
       ${TRAIN_STR} \
       ${test_str} \
       -d${DO_STR} \
       --app flylight --root ${PPP_ROOT} -id ${EXP_ID} \
       ${CKPT_STR} \
       ${SAMPLE_STR}"
}

find_summary_csv() {
  local ckpt="$1"
  local split="${2:-${PROCESSED_SPLIT}}"
  local base="metrics/ppp/${EXP_ID}/${split}/evaluated/${ckpt}"
  if [[ ! -d "${base}" ]]; then
    return 1
  fi
  # Prefer the newest summary.csv under this ckpt (VI param tree).
  find "${base}" -name summary.csv -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

copy_summary_for_split() {
  local split="$1" ckpt="$2"
  local sum dir
  sum="$(find_summary_csv "${ckpt}" "${split}" || true)"
  if [[ -z "${sum}" || ! -f "${sum}" ]]; then
    echo "Error: no summary.csv under metrics/ppp/${EXP_ID}/${split}/evaluated/${ckpt} after evaluate (${split})" >&2
    exit 1
  fi
  dir="$(dirname "${sum}")"
  cp -f "${sum}" "${dir}/summary_${split}.csv"
  echo "copied ${sum} → ${dir}/summary_${split}.csv"
  # Export last summary dir for merge.
  LAST_SUMMARY_DIR="${dir}"
}

# ---- run_ppp ----
LAST_SUMMARY_DIR=""
VAL_SUMMARY_DIR=""
TEST_SUMMARY_DIR=""
if [[ "${EVAL_BOTH_SPLITS}" == "1" ]]; then
  if ! has_task evaluate; then
    echo "Error: EVAL_BOTH_SPLITS=1 requires evaluate in DO_TASKS" >&2
    exit 1
  fi
  # Dual-split evaluate only (submit sets DO_TASKS=evaluate).
  DO_STR=" evaluate"
  for _split in val test; do
    echo "evaluate both: split=${_split}"
    PROCESSED_SPLIT="${_split}"
    run_ppp_once "${FISBE_ROOT}/${_split}"
    _ckpt="$(resolve_ckpt_for_strip || true)"
    if [[ -z "${_ckpt}" ]]; then
      echo "Error: could not resolve ckpt for summary copy" >&2
      exit 1
    fi
    copy_summary_for_split "${_split}" "${_ckpt}"
    if [[ "${_split}" == "val" ]]; then
      VAL_SUMMARY_DIR="${LAST_SUMMARY_DIR}"
    else
      TEST_SUMMARY_DIR="${LAST_SUMMARY_DIR}"
    fi
  done
  if [[ -z "${VAL_SUMMARY_DIR}" || -z "${TEST_SUMMARY_DIR}" ]]; then
    echo "Error: no summary dir for merge (val='${VAL_SUMMARY_DIR}' test='${TEST_SUMMARY_DIR}')" >&2
    exit 1
  fi
  python3 sbatch/ppp/merge_ppp_summary_csvs.py \
    --val  "${VAL_SUMMARY_DIR}/summary_val.csv" \
    --test "${TEST_SUMMARY_DIR}/summary_test.csv" \
    --out  "${TEST_SUMMARY_DIR}/summary.csv"
else
  # Single run (existing behavior). TEST_DATA already set from SPLIT / default val.
  run_ppp_once "${TEST_DATA}"
  if has_task evaluate && [[ -n "${SPLIT}" ]]; then
    _ckpt="$(resolve_ckpt_for_strip || true)"
    if [[ -n "${_ckpt}" ]]; then
      copy_summary_for_split "${SPLIT}" "${_ckpt}"
    else
      echo "Warning: could not resolve ckpt; skipped summary_${SPLIT}.csv copy" >&2
    fi
  fi
fi

if has_task predict && [[ "${RUN_PRED_METRICS}" == "1" ]]; then
  run_pred_metrics_if_requested
fi

# After evaluate: drop pred_affs (keep pred_numinst) except keep-sample, unless opted out.
if has_task evaluate && [[ "${KEEP_PRED_AFFS}" != "1" ]]; then
  STRIP_CKPT="$(resolve_ckpt_for_strip || true)"
  if [[ -z "${STRIP_CKPT}" ]]; then
    # When EVAL_BOTH just finished, PROCESSED_SPLIT is test; try resolving with CKPT alone.
    if [[ -n "${CKPT}" ]]; then
      STRIP_CKPT="${CKPT}"
    else
      echo "strip_pred_affs: skip (could not resolve ckpt under metrics/ppp/${EXP_ID}/${PROCESSED_SPLIT}/processed)"
      STRIP_CKPT=""
    fi
  fi
  if [[ -n "${STRIP_CKPT}" ]]; then
    if [[ "${EVAL_BOTH_SPLITS}" == "1" ]]; then
      _strip_splits=(val test)
    else
      _strip_splits=("${PROCESSED_SPLIT}")
    fi
    for _sp in "${_strip_splits[@]}"; do
      STRIP_ARGS=(
        python3 sbatch/ppp/strip_pred_affs.py
        --root metrics/ppp
        --exp "${EXP_ID}"
        --split "${_sp}"
        --ckpt "${STRIP_CKPT}"
        --keep-name "${KEEP_AFFS_SAMPLE}"
      )
      # Single-sample jobs only strip that sample (keep-sample is still retained).
      # Full-split evaluate (no SAMPLE) strips the whole ckpt dir except keep-sample.
      if [[ -n "${SAMPLE}" ]]; then
        STRIP_ARGS+=(--sample "${SAMPLE}")
      fi
      echo "strip_pred_affs: ${STRIP_ARGS[*]}"
      "${STRIP_ARGS[@]}"
    done
  fi
fi

echo "ppp_job finished exp=${EXP_ID} sample=${SAMPLE:-none}"
