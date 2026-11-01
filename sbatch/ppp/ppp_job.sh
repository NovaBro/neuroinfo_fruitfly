#!/bin/bash
# PatchPerPix generic SLURM job body. Resources and job-name come from
# ppp_submit_train.sh / ppp_submit_test.sh (or sbatch CLI). Behavior is
# driven by exported env vars.
#
# Prefer:
#   ./sbatch/ppp/ppp_submit_train.sh <preset> …
#   ./sbatch/ppp/ppp_submit_test.sh <mode> <preset> …
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
#   SAMPLE          --sample {dataset}/{stem}  (required namespaced when set;
#                   ignored when ARRAY_MODE=1 — array picks namespaced IDs)
#   SPLIT           val|test → passed as run_ppp --split; output under
#                   metrics/.../$SPLIT/; omit --test-data (list mode)
#   DATASETS        +/CSV of fisbe/ folder names. Default (unset/empty): derived
#                   from the merged TOMLs via ppp_datasets.py (train: train_datasets
#                   + val_datasets; val/test: <split>_datasets). Set to override.
#                   Array tasks union *.zarr under each; sample IDs are {dataset}/{stem}.
#                   Infer jobs omit --test-data so list mode stays active.
#   STAGE_DATA      1 → extract fisbe/<DATASETS>/{train,val}_ppp.tar into
#                   $SLURM_TMPDIR/ppp_data/<dataset>/{train,val}/ and pass
#                   --fisbe-root (list mode). Build tars via
#                   sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh
#                   Leave this off for sqf-packed datasets (1k_v1): extracting
#                   the tar writes the chunk files out again. Those jobs mount
#                   fisbe/sqf/<dataset>/*.sqf and the source image automatically.
#   ARRAY_MODE      1 → pick sample by SLURM_ARRAY_TASK_ID from DATASETS×SPLIT
#   DISCARD_PARTIAL 1 → drop incomplete volumes/pred_affs before predict (keeps pred_numinst)
#   RUN_PRED_METRICS 1 → run diagnose_ppp_pred_metrics.py after predict / before label
#   LABEL_RESUME    1 → append label_resume.toml
#   KEEP_PRED_AFFS  1 → after evaluate, do not strip volumes/pred_affs (default: strip
#                   every sample via strip_pred_affs.py)
#   KEEP_PRED_CODE  1 → after evaluate, do not strip volumes/pred_code (default: strip
#                   every sample via strip_pred_affs.py)
#   KEEP_AFFS_SAMPLE comma-separated zarr names to retain (default empty: strip all)
#   EVAL_BOTH_SPLITS 1 → evaluate val then test; write summary_val/test.csv and merge summary.csv
#   NEED_GPU        0 → CPU job: skip --nv, GPU logger, CUDA_VISIBLE_DEVICES (default 1).
#                   Also implied for evaluate-only tasks.
#   PPP_ROOT        metrics root relative to experiments/ (default ../../metrics/ppp)
#   DEFAULT_CONFIG  always prepended (default default_l40s.toml);
#                   set by --base FILE.toml on either submitter
#   SETUP_DIR       config dir under experiments/ (default flylight/setups/setup01)
#
# Flow:
#   1. Validate env; normalize DO_TASKS / CONFIGS; resolve SAMPLE (array or default).
#   2. Assemble run_ppp.py argv pieces (configs, ckpt, sample, train/test data).
#   3. Start nvidia-smi GPU logger (skipped when NEED_GPU=0 / evaluate-only); optionally stage tars.
#   4. Pre-flight: discard partial pred_affs; require complete affinities for label-only.
#   5. Run run_ppp.py inside Singularity (no --nv when NEED_GPU=0; or val+test when
#      EVAL_BOTH_SPLITS=1).
#   6. Post: optional pred metrics, summary_<split>.csv copy/merge, strip pred_affs/pred_code.

#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="/scratch/wmz2007/neuroinfo_fruitfly"
cd "${REPO_ROOT}"

# ---- env / defaults ----
# Required vars come from the submitter --export; optionals default to "off" / empty.
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
KEEP_PRED_CODE="${KEEP_PRED_CODE:-0}"
EVAL_BOTH_SPLITS="${EVAL_BOTH_SPLITS:-0}"
NEED_GPU="${NEED_GPU:-1}"
NEEDS_DECODE="${NEEDS_DECODE:-0}"
PPP_ROOT="${PPP_ROOT:-../../metrics/ppp}"
DEFAULT_CONFIG="${DEFAULT_CONFIG:-default_l40s.toml}"
SETUP_DIR="${SETUP_DIR:-flylight/setups/setup01}"
FISBE_PARENT="${FISBE_PARENT:-${REPO_ROOT}/fisbe}"
# Completely root kept for legacy scripts; staging uses FISBE_PARENT/<dataset>/.
FISBE_ROOT="${FISBE_ROOT:-${FISBE_PARENT}/completely}"
# Empty → derive per split from the TOMLs (datasets_for).
DATASETS="${DATASETS:-}"
DEFAULT_SAMPLE="${DEFAULT_SAMPLE:-completely/R22C03-20180918_66_J2}"
KEEP_AFFS_SAMPLE="${KEEP_AFFS_SAMPLE:-}"

# Route inference outputs / strip paths to val/ or test/.
if [[ "${SPLIT}" == "val" || "${SPLIT}" == "test" ]]; then
  PROCESSED_SPLIT="${SPLIT}"
else
  PROCESSED_SPLIT="val"
fi

# ---- helpers ----

datasets_for() {
  # Echo '+'-joined dataset names for split $1: DATASETS if set, else from the
  # merged TOMLs (DEFAULT_CONFIG first unless listed, then CONFIGS), same order
  # as the run_ppp.py --config args.
  if [[ -n "${DATASETS}" ]]; then
    echo "${DATASETS}"
    return 0
  fi
  local names="${CONFIGS//+/ }" c cfgs=()
  names="${names//,/ }"
  [[ " ${names} " == *" ${DEFAULT_CONFIG} "* ]] \
    || cfgs+=("PatchPerPix/experiments/${SETUP_DIR}/${DEFAULT_CONFIG}")
  for c in ${names}; do
    cfgs+=("PatchPerPix/experiments/${SETUP_DIR}/${c}")
  done
  python3 sbatch/ppp/ppp_datasets.py --split "$1" "${cfgs[@]}"
}

list_array_samples() {
  # Echo sorted {dataset}/{stem} for split $1 (datasets $2 or DATASETS/TOML).
  # Honors data/max_test_samples via ppp_datasets.py (same order as run_ppp).
  local split="$1"
  local ds_override="${2:-}"
  local names="${CONFIGS//+/ }" c cfgs=()
  names="${names//,/ }"
  [[ " ${names} " == *" ${DEFAULT_CONFIG} "* ]] \
    || cfgs+=("PatchPerPix/experiments/${SETUP_DIR}/${DEFAULT_CONFIG}")
  for c in ${names}; do
    cfgs+=("PatchPerPix/experiments/${SETUP_DIR}/${c}")
  done
  local args=(--split "${split}" --list-samples --fisbe-root "${FISBE_PARENT}")
  local ds="${DATASETS:-${ds_override}}"
  if [[ -n "${ds}" ]]; then
    args+=(--datasets "${ds}")
  fi
  python3 sbatch/ppp/ppp_datasets.py "${args[@]}" "${cfgs[@]}"
}

processed_sample_zarr() {
  # Path to the processed prediction zarr for one sample at one checkpoint.
  # Args: $1 = ckpt number, $2 = sample id (flat stem or dataset/stem).
  # Prefers nested list-mode path; falls back to flat stem for older outputs.
  local ckpt="$1" sample="$2"
  local nested="metrics/ppp/${EXP_ID}/${PROCESSED_SPLIT}/processed/${ckpt}/${sample}.zarr"
  if [[ -d "${nested}" ]]; then
    echo "${nested}"
    return 0
  fi
  local flat="metrics/ppp/${EXP_ID}/${PROCESSED_SPLIT}/processed/${ckpt}/${sample##*/}.zarr"
  if [[ -d "${flat}" ]]; then
    echo "${flat}"
    return 0
  fi
  echo "${nested}"
}

resolve_ckpt_for_strip() {
  # Pick the checkpoint dir used for summary copy / pred_affs strip.
  # Prefer explicit CKPT; else newest numeric processed/ subdir that contains
  # SAMPLE (flat or nested) or any sample if SAMPLE unset. Echoes ckpt id.
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
  # Return 0 if $1 appears in DO_ARR (the normalized -d task list).
  local t
  for t in "${DO_ARR[@]}"; do
    [[ "${t}" == "$1" ]] && return 0
  done
  return 1
}

IS_TRAIN=0
has_task train && IS_TRAIN=1

# CPU jobs (NEED_GPU=0 or evaluate-only): skip --nv, GPU logger, CUDA_VISIBLE_DEVICES.
EVALUATE_ONLY=0
if has_task evaluate \
  && ! has_task train && ! has_task predict && ! has_task label && ! has_task decode; then
  EVALUATE_ONLY=1
fi
CPU_JOB=0
if [[ "${NEED_GPU}" == "0" || "${EVALUATE_ONLY}" -eq 1 ]]; then
  CPU_JOB=1
fi

# ---- sample / test-data resolution ----
# Array tasks index sorted {dataset}/{stem} under fisbe/<DATASETS>/<SPLIT>.
# Infer (val and test): list mode — namespaced SAMPLE, omit --test-data, pass
# --split so run_ppp retargets list-mode discovery + output folder.
# Single-sample predict/label: default SAMPLE when unset (evaluate may leave empty).

# Non-train with empty SPLIT defaults to val (historical single-sample infer).
if [[ "${IS_TRAIN}" -eq 0 && -z "${SPLIT}" ]]; then
  SPLIT="val"
  PROCESSED_SPLIT="val"
fi

# Datasets for this job's split (train → train+val datasets); aborts on helper error.
if [[ "${IS_TRAIN}" -eq 1 ]]; then
  JOB_DATASETS="$(datasets_for train)"
else
  JOB_DATASETS="$(datasets_for "${SPLIT}")"
fi

USE_LIST_MODE=0
if [[ "${IS_TRAIN}" -eq 0 && ( "${SPLIT}" == "val" || "${SPLIT}" == "test" ) ]]; then
  USE_LIST_MODE=1
fi

if [[ "${ARRAY_MODE}" == "1" ]]; then
  if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
    echo "Error: ARRAY_MODE=1 requires SPLIT=val or SPLIT=test" >&2
    exit 1
  fi
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Error: ARRAY_MODE=1 requires a SLURM array job" >&2
    exit 1
  fi
  mapfile -t SAMPLES < <(list_array_samples "${SPLIT}" "${JOB_DATASETS}")
  if (( SLURM_ARRAY_TASK_ID >= ${#SAMPLES[@]} )); then
    echo "task ${SLURM_ARRAY_TASK_ID} >= ${#SAMPLES[@]} samples; nothing to do"
    exit 0
  fi
  SAMPLE="${SAMPLES[${SLURM_ARRAY_TASK_ID}]}"
  echo "[array ${SLURM_ARRAY_TASK_ID}] split=${SPLIT} sample=${SAMPLE} datasets=${JOB_DATASETS}"
fi

# Default single-sample for predict/label when SAMPLE unset (not array, not eval-only).
if [[ "${IS_TRAIN}" -eq 0 && -z "${SAMPLE}" && "${ARRAY_MODE}" != "1" ]]; then
  if has_task predict || has_task label; then
    SAMPLE="${DEFAULT_SAMPLE}"
  fi
fi

# Namespaced sample IDs only ({dataset}/{stem}); no flat stems / silent prefix.
if [[ -n "${SAMPLE}" && "${SAMPLE}" != */* ]]; then
  echo "Error: SAMPLE must be '{dataset}/{stem}' (got '${SAMPLE}'); flat stems are not supported" >&2
  exit 1
fi

# List-mode infer: never pass --test-data (TOML val_datasets/test_datasets + --split).
TEST_DATA=""

# ---- assemble run_ppp.py argv fragments ----
# CONFIGS uses '+' (SLURM --export cannot carry spaces). Always prepend
# DEFAULT_CONFIG unless the caller already listed it first among overlays.

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

# Non-train: explicit --checkpoint N, else --test-checkpoint last.
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

SPLIT_ARGS=()
if [[ "${SPLIT}" == "val" || "${SPLIT}" == "test" ]]; then
  SPLIT_ARGS+=(--split "${SPLIT}")
fi

echo "ppp_job exp=${EXP_ID} do=${DO_ARR[*]} configs=${CFG_NAMES[*]} ckpt=${CKPT:-last} sample=${SAMPLE:-none} split=${SPLIT:-none} datasets=${JOB_DATASETS} list_mode=${USE_LIST_MODE} stage=${STAGE_DATA} array=${ARRAY_MODE} job=${SLURM_JOB_ID:-local}"

# ---- GPU logger ----
# Background nvidia-smi CSV for Greene util audits; killed on EXIT via trap.
# Skipped for CPU jobs (NEED_GPU=0 / evaluate-only) and when nvidia-smi is missing.

GPU_LOGGER_PID=
cleanup() {
  # Stop the background GPU logger on job exit (success or failure).
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
if [[ "${CPU_JOB}" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 3 \
    > "gpu_log/ppp/gpu_usage_log_${LOG_TAG}.csv" &
  GPU_LOGGER_PID=$!
fi

# ---- optional train/val staging ----
# Extract per-dataset tars onto node-local SSD as:
#   $SLURM_TMPDIR/ppp_data/<dataset>/{train,val}/
# then pass --fisbe-root so list mode stays on (no --train-data/--val-data).

STAGE_ROOT=""
BIND_ARGS=()
TRAIN_ARGS=()
if [[ "${STAGE_DATA}" == "1" ]]; then
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "Error: SLURM_TMPDIR is unset; cannot stage training data" >&2
    exit 1
  fi
  STAGE_ROOT="${SLURM_TMPDIR}/ppp_data"
  STAGE_DATASETS="$(datasets_for train)"
  echo "Staging train/val from tars -> ${STAGE_ROOT} datasets=${STAGE_DATASETS}"
  df -h "${SLURM_TMPDIR}"
  STAGE_T0=$(date +%s)
  mkdir -p "${STAGE_ROOT}"

  _stage_pids=()
  _ds_raw="${STAGE_DATASETS//+/ }"
  _ds_raw="${_ds_raw//,/ }"
  for _ds in ${_ds_raw}; do
    _ds="${_ds// /}"
    [[ -n "${_ds}" ]] || continue
    TRAIN_TAR="${FISBE_PARENT}/${_ds}/train_ppp.tar"
    VAL_TAR="${FISBE_PARENT}/${_ds}/val_ppp.tar"
    if [[ ! -f "${TRAIN_TAR}" || ! -f "${VAL_TAR}" ]]; then
      echo "Error: missing ${TRAIN_TAR} and/or ${VAL_TAR}" >&2
      echo "Build with: DATASETS=${STAGE_DATASETS} sbatch sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh" >&2
      exit 1
    fi
    mkdir -p "${STAGE_ROOT}/${_ds}/train" "${STAGE_ROOT}/${_ds}/val"
    tar -xf "${TRAIN_TAR}" -C "${STAGE_ROOT}/${_ds}/train" &
    _stage_pids+=($!)
    tar -xf "${VAL_TAR}" -C "${STAGE_ROOT}/${_ds}/val" &
    _stage_pids+=($!)
  done
  for _pid in "${_stage_pids[@]}"; do
    wait "${_pid}"
  done
  STAGE_T1=$(date +%s)
  echo "Staging finished in $((STAGE_T1 - STAGE_T0))s"
  df -h "${SLURM_TMPDIR}"
  BIND_ARGS+=(--bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}")
  TRAIN_ARGS+=(--fisbe-root "${STAGE_ROOT}")
fi

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="env/ppp.ext3:ro"
SQF_ROOT="/scratch/vh2308/data/synth_MCFO"
PPP_LINK_ARRAYS=(raw_normalized gt_instances_rm_5 gt_numinst gt_fg_rm_5)
DATA_OVERLAY_ARGS=()

resolve_data_overlays() {
  # Mount source SquashFS images and packed PPP shards for dataset list $1
  # (+ or comma separated). Source images come from volumes/raw symlinks
  # into /synth_MCFO/<name>/. Packed arrays are symlinks into
  # /synth_MCFO/<dataset>_ppp/ and live in fisbe/sqf/<dataset>/*.sqf.
  # completely/partly have real files and add nothing.
  local ds_raw="$1"
  local -A mounted=()
  local ds split raw target name sqf shard shard_dir arr link need_ppp shard_found
  DATA_OVERLAY_ARGS=()
  ds_raw="${ds_raw//+/ }"
  ds_raw="${ds_raw//,/ }"
  for ds in ${ds_raw}; do
    ds="${ds// /}"
    [[ -n "${ds}" ]] || continue
    for split in train val test; do
      for raw in "${FISBE_PARENT}/${ds}/${split}/"*.zarr/volumes/raw; do
        [[ -L "${raw}" ]] || continue
        target="$(readlink "${raw}")"
        [[ "${target}" =~ ^/synth_MCFO/([^/]+)/ ]] || continue
        name="${BASH_REMATCH[1]}"
        sqf="${SQF_ROOT}/${name}.sqf"
        if [[ ! -f "${sqf}" ]]; then
          echo "error: ${raw} -> ${target}" >&2
          echo "not mounted: ${sqf}" >&2
          exit 1
        fi
        sqf="$(realpath -e "${sqf}")"
        if [[ -n "${mounted[${sqf}]:-}" ]]; then
          continue
        fi
        mounted["${sqf}"]=1
        DATA_OVERLAY_ARGS+=(--overlay "${sqf}:ro")
        echo "SQF=${sqf} mount=/synth_MCFO/${name}"
      done
    done
    need_ppp=0
    for split in train val test; do
      for arr in "${PPP_LINK_ARRAYS[@]}"; do
        for link in "${FISBE_PARENT}/${ds}/${split}/"*.zarr/volumes/"${arr}"; do
          [[ -L "${link}" ]] || continue
          target="$(readlink "${link}")"
          if [[ "${target}" == /synth_MCFO/* ]]; then
            need_ppp=1
          fi
        done
      done
    done
    shard_dir="${FISBE_PARENT}/sqf/${ds}"
    shard_found=0
    for shard in "${shard_dir}/"*.sqf; do
      [[ -f "${shard}" ]] || continue
      shard_found=1
      shard="$(realpath -e "${shard}")"
      if [[ -n "${mounted[${shard}]:-}" ]]; then
        continue
      fi
      mounted["${shard}"]=1
      DATA_OVERLAY_ARGS+=(--overlay "${shard}:ro")
      echo "PPP_SQF=${shard}"
    done
    if [[ "${need_ppp}" -eq 1 && "${shard_found}" -eq 0 ]]; then
      echo "error: ${ds} prepared-array symlink points at /synth_MCFO/ but no image matched ${shard_dir}/*.sqf" >&2
      exit 1
    fi
  done
}

_overlay_ds="${JOB_DATASETS}"
if [[ "${EVAL_BOTH_SPLITS}" == "1" ]]; then
  _overlay_ds="$(datasets_for val)+$(datasets_for test)"
fi
resolve_data_overlays "${_overlay_ds}"

run_singularity() {
  # Run a bash -c command string inside the ppp Singularity + conda overlay.
  # Activates env ppp; forwards BIND_ARGS (e.g. staged $SLURM_TMPDIR).
  # CPU jobs (NEED_GPU=0 / evaluate-only): omit --nv.
  # Args: $1 = shell command to execute after conda activate.
  local inner="$1"
  local nv_args=()
  if [[ "${CPU_JOB}" -eq 0 ]]; then
    nv_args=(--nv)
  fi
  singularity exec "${nv_args[@]}" \
    --overlay "${OVERLAY}" \
    "${DATA_OVERLAY_ARGS[@]}" \
    "${BIND_ARGS[@]}" \
    "${SIF}" \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; ${inner}"
}

# ---- pre-flight: partial predictions / label / decode prerequisites ----
# Array resume: drop incomplete primary prediction arrays so predict can rewrite
# them; keep pred_numinst. train_code (NEEDS_DECODE=1) uses pred_code as the
# primary predict output; otherwise pred_affs. Label-only jobs must already have
# complete pred_affs. Decode-only jobs must already have complete pred_code.

PRED_PRIMARY_KEY="volumes/pred_affs"
PRED_PRIMARY_NAME="pred_affs"
if [[ "${NEEDS_DECODE}" == "1" ]]; then
  PRED_PRIMARY_KEY="volumes/pred_code"
  PRED_PRIMARY_NAME="pred_code"
fi

if [[ "${DISCARD_PARTIAL}" == "1" && -n "${SAMPLE}" && -n "${CKPT}" ]]; then
  SAMPLE_ZARR="$(processed_sample_zarr "${CKPT}" "${SAMPLE}")"
  if [[ -d "${SAMPLE_ZARR}" ]] \
    && ! python3 sbatch/ppp/check_prediction_complete.py -q \
      --aff-key "${PRED_PRIMARY_KEY}" "${SAMPLE_ZARR}"; then
    PRIMARY_DIR="${SAMPLE_ZARR}/${PRED_PRIMARY_KEY}"
    NUMINST_DIR="${SAMPLE_ZARR}/volumes/pred_numinst"
    if [[ -d "${PRIMARY_DIR}" ]]; then
      echo "discarding incomplete ${PRED_PRIMARY_NAME}: ${PRIMARY_DIR}"
      rm -rf "${PRIMARY_DIR}"
    fi
    if [[ ! -d "${NUMINST_DIR}" && ! -d "${PRIMARY_DIR}" ]]; then
      echo "discarding empty prediction zarr: ${SAMPLE_ZARR}"
      rm -rf "${SAMPLE_ZARR}"
    fi
  fi
fi

run_pred_metrics_if_requested() {
  # Optionally run diagnose_ppp_pred_metrics.py on the processed sample zarr.
  # No-op unless RUN_PRED_METRICS=1 and SAMPLE+CKPT are set and the zarr exists.
  if [[ "${RUN_PRED_METRICS}" != "1" || -z "${SAMPLE}" || -z "${CKPT}" ]]; then
    return 0
  fi
  local zarr
  zarr="$(processed_sample_zarr "${CKPT}" "${SAMPLE}")"
  if [[ -d "${zarr}" ]]; then
    run_singularity "python -u sbatch/ppp/tools/diagnose_ppp_pred_metrics.py --pred-zarr '${zarr}'"
  fi
}

# Metrics timing: after predict (handled post-run below); for label-only, run now
# so FG/numinst stats are logged before vote_instances starts.
if has_task predict && [[ "${RUN_PRED_METRICS}" == "1" ]]; then
  : # metrics after predict
elif has_task label && [[ "${RUN_PRED_METRICS}" == "1" ]] && ! has_task predict; then
  run_pred_metrics_if_requested
fi

# Require complete pred_code for decode-only tasks (no predict in the same job).
if has_task decode && ! has_task predict; then
  if [[ -z "${CKPT}" ]]; then
    echo "Error: decode without predict requires CKPT" >&2
    exit 1
  fi
  if [[ -z "${SAMPLE}" ]]; then
    echo "Error: decode without predict requires SAMPLE" >&2
    exit 1
  fi
  PRED_ZARR="$(processed_sample_zarr "${CKPT}" "${SAMPLE}")"
  if [[ ! -d "${PRED_ZARR}/volumes/pred_code" ]]; then
    echo "Error: missing pred_code for decode: ${PRED_ZARR}/volumes/pred_code" >&2
    exit 1
  fi
  if ! python3 sbatch/ppp/check_prediction_complete.py -q \
    --aff-key volumes/pred_code "${PRED_ZARR}"; then
    echo "Error: incomplete pred_code for decode: ${PRED_ZARR}" >&2
    exit 1
  fi
fi

# Require complete pred_affs for label-only tasks (array or single-sample).
if has_task label && ! has_task predict && ! has_task decode; then
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

# Flatten argv arrays into space-separated strings for the singularity bash -c.
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
SPLIT_STR=""
for a in "${SPLIT_ARGS[@]}"; do
  SPLIT_STR+=" ${a}"
done
DO_STR=""
for a in "${DO_ARGS[@]}"; do
  DO_STR+=" ${a}"
done

run_ppp_once() {
  # Invoke PatchPerPix/experiments/run_ppp.py inside the container once.
  # Uses prebuilt CONFIG_STR / TRAIN_STR / CKPT_STR / SAMPLE_STR / DO_STR.
  # Args: $1 = optional --test-data path (empty → omit the flag).
  #       $2 = optional --split override (empty → use prebuilt SPLIT_STR).
  local test_data="${1:-}"
  local split_override="${2:-}"
  local test_str=""
  if [[ -n "${test_data}" ]]; then
    test_str=" --test-data ${test_data}"
  fi
  local split_str="${SPLIT_STR}"
  if [[ -n "${split_override}" ]]; then
    split_str=" --split ${split_override}"
  fi
  local cuda_env=""
  if [[ "${CPU_JOB}" -eq 0 ]]; then
    cuda_env="CUDA_VISIBLE_DEVICES=0 "
  fi
  run_singularity \
    "cd PatchPerPix/experiments/; \
     env ${cuda_env}PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python3 -u run_ppp.py --setup setup01 \
       ${CONFIG_STR} \
       ${TRAIN_STR} \
       ${test_str} \
       ${split_str} \
       -d${DO_STR} \
       --app flylight --root ${PPP_ROOT} -id ${EXP_ID} \
       ${CKPT_STR} \
       ${SAMPLE_STR}"
}

find_summary_csv() {
  # Locate the newest summary.csv under evaluated/<ckpt>/ (VI param tree).
  # Args: $1 = ckpt, $2 = optional split (default PROCESSED_SPLIT).
  # Echoes the path, or returns 1 if none found.
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
  # Copy evaluated summary.csv → summary_<split>.csv beside it; set LAST_SUMMARY_DIR.
  # Args: $1 = split (val|test), $2 = ckpt. Exits 1 if summary.csv is missing.
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
# EVAL_BOTH_SPLITS: two evaluate-only passes (val then test), copy summary_*.csv,
# then merge into test/.../summary.csv. Otherwise one run_ppp_once (list mode).

LAST_SUMMARY_DIR=""
VAL_SUMMARY_DIR=""
TEST_SUMMARY_DIR=""
if [[ "${EVAL_BOTH_SPLITS}" == "1" ]]; then
  if ! has_task evaluate; then
    echo "Error: EVAL_BOTH_SPLITS=1 requires evaluate in DO_TASKS" >&2
    exit 1
  fi
  # Dual-split evaluate only (submit sets DO_TASKS=evaluate).
  # Both splits: omit --test-data; pass --split for list-mode routing.
  DO_STR=" evaluate"
  for _split in val test; do
    echo "evaluate both: split=${_split} datasets=$(datasets_for "${_split}")"
    PROCESSED_SPLIT="${_split}"
    run_ppp_once "" "${_split}"
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
  # Single run. TEST_DATA empty → omit --test-data (list mode + --split).
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

# Post-predict diagnostics (deferred until affinities exist on disk).
if has_task predict && [[ "${RUN_PRED_METRICS}" == "1" ]]; then
  run_pred_metrics_if_requested
fi

# ---- post: strip pred_affs / pred_code ----
# After evaluate, drop large prediction arrays to save space (keep pred_numinst).
# KEEP_AFFS_SAMPLE, if set, retains those arrays on the named samples only.
# Single-sample jobs only strip that sample. Full-split evaluate strips the
# whole ckpt dir.
# Opt out: KEEP_PRED_AFFS=1 and/or KEEP_PRED_CODE=1.

STRIP_ARRAYS=()
if [[ "${KEEP_PRED_AFFS}" != "1" ]]; then
  STRIP_ARRAYS+=("volumes/pred_affs")
fi
if [[ "${KEEP_PRED_CODE}" != "1" ]]; then
  STRIP_ARRAYS+=("volumes/pred_code")
fi

if has_task evaluate && (( ${#STRIP_ARRAYS[@]} > 0 )); then
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
    # Comma-join for --arrays (e.g. volumes/pred_affs,volumes/pred_code).
    _strip_arrays_csv="$(IFS=,; echo "${STRIP_ARRAYS[*]}")"
    for _sp in "${_strip_splits[@]}"; do
      STRIP_ARGS=(
        python3 sbatch/ppp/strip_pred_affs.py
        --root metrics/ppp
        --exp "${EXP_ID}"
        --split "${_sp}"
        --ckpt "${STRIP_CKPT}"
        --keep-name "${KEEP_AFFS_SAMPLE}"
        --arrays "${_strip_arrays_csv}"
      )
      # Single-sample jobs only strip that sample (named KEEP_AFFS_SAMPLE is retained).
      # Full-split evaluate (no SAMPLE) strips the whole ckpt dir.
      if [[ -n "${SAMPLE}" ]]; then
        STRIP_ARGS+=(--sample "${SAMPLE}")
      fi
      echo "strip_pred_affs: ${STRIP_ARGS[*]}"
      "${STRIP_ARGS[@]}"
    done
  fi
fi

echo "ppp_job finished exp=${EXP_ID} sample=${SAMPLE:-none}"
