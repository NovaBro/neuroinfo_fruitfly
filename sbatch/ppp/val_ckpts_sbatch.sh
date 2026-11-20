#!/bin/bash
# Offline train-time validation of saved checkpoints on other datasets' val splits
# (PatchPerPix/experiments/flylight/setups/setup01/val_ckpts.py).
#
# Each argument is one run:  EXP_ID:DATASET[+DATASET...][:OUT_SUBDIR]
#   EXP_ID      dir under metrics/ppp/
#   DATASET     fisbe/<dataset> whose val split to evaluate -> val_metrics_<dataset>.csv
#   OUT_SUBDIR  optional, under <exp>/train/ (default: <exp>/train itself)
#
# Submit from the repo root, e.g.:
#   sbatch sbatch/ppp/val_ckpts_sbatch.sh \
#     expv2-e1.1-pre-n1000:completely \
#     expv2-e1.3-ft-n1000-completely:1k_v1.1
# Optional env: CKPTS="2000 4000" (default: every checkpoint), SBATCH overrides on the CLI.

#SBATCH --job-name=ppp_val_ckpts
#SBATCH --account=torch_pr_61_tandon_priority
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge

REPO_ROOT="/scratch/wmz2007/neuroinfo_fruitfly"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/sbatch/ppp/val_ckpts_sbatch.sh" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
fi
cd "${REPO_ROOT}"

if (( $# < 1 )); then
  echo "usage: sbatch sbatch/ppp/val_ckpts_sbatch.sh EXP_ID:DATASET[+DATASET][:OUT_SUBDIR] ..." >&2
  exit 1
fi

FISBE_PARENT="${FISBE_PARENT:-${REPO_ROOT}/fisbe}"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="env/ppp.ext3:ro"
SQF_ROOT="/scratch/vh2308/data/synth_MCFO"
PPP_LINK_ARRAYS=(raw_normalized gt_instances_rm_5 gt_numinst gt_fg_rm_5)
DATA_OVERLAY_ARGS=()
CKPTS="${CKPTS:-}"

# ---- copy of ppp_job.sh resolve_data_overlays (keep in sync) ----
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
    src_list="${FISBE_PARENT}/${ds}/source_sqf.txt"
    if [[ -f "${src_list}" ]]; then
      while read -r sqf; do
        [[ -z "${sqf}" || "${sqf}" == \#* ]] && continue
        if ! sqf="$(realpath -e "${sqf}")"; then
          echo "error: source image listed in ${src_list} not found" >&2
          exit 1
        fi
        [[ -n "${mounted[${sqf}]:-}" ]] && continue
        mounted["${sqf}"]=1
        DATA_OVERLAY_ARGS+=(--overlay "${sqf}:ro")
        echo "SRC_SQF=${sqf} (${src_list})"
      done < "${src_list}"
    fi
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

# ---- parse runs ----
CMDS=""
ALL_DS=()
for spec in "$@"; do
  IFS=':' read -r exp ds_list outsub <<< "${spec}"
  if [[ -z "${exp}" || -z "${ds_list}" ]]; then
    echo "error: bad run spec '${spec}' (want EXP_ID:DATASET[+DATASET][:OUT_SUBDIR])" >&2
    exit 1
  fi
  if [[ ! -f "metrics/ppp/${exp}/config.toml" ]]; then
    echo "error: metrics/ppp/${exp}/config.toml not found" >&2
    exit 1
  fi
  ds_words="${ds_list//+/ }"
  for d in ${ds_words}; do
    ALL_DS+=("${d}")
  done
  cmd="python -u -m flylight.setups.setup01.val_ckpts --exp ../../metrics/ppp/${exp} --datasets ${ds_words}"
  if [[ -n "${outsub:-}" ]]; then
    cmd+=" --out-dir ../../metrics/ppp/${exp}/train/${outsub}"
  fi
  if [[ -n "${CKPTS}" ]]; then
    cmd+=" --checkpoints ${CKPTS}"
  fi
  CMDS+="echo '>>> ${spec}'; ${cmd}; "
done

MOUNT_DS="$(printf '%s\n' "${ALL_DS[@]}" | sort -u | paste -sd+)"
echo "val_ckpts job=${SLURM_JOB_ID:-local} runs=$* mount_datasets=${MOUNT_DS}"
resolve_data_overlays "${MOUNT_DS}"
echo "overlays: $(( ${#DATA_OVERLAY_ARGS[@]} / 2 ))"

# ---- GPU logger (public partitions cancel low-utilization jobs; keep for audits) ----
GPU_LOGGER_PID=
cleanup() {
  if [[ -n "${GPU_LOGGER_PID}" ]]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
mkdir -p gpu_log/ppp
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 3 > "gpu_log/ppp/gpu_usage_log_val_ckpts_${SLURM_JOB_ID:-local}.csv" &
  GPU_LOGGER_PID=$!
fi

singularity exec --nv \
  --overlay "${OVERLAY}" \
  "${DATA_OVERLAY_ARGS[@]}" \
  "${SIF}" \
  /bin/bash -c "set -e; source /ext3/env.sh; conda activate ppp; \
    cd PatchPerPix/experiments/; export CUDA_VISIBLE_DEVICES=0; ${CMDS}"

echo "val_ckpts done"
