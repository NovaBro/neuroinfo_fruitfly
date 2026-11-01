#!/bin/bash
# One-time FG point CSVs for sample_fg_points training (written to fisbe/fg_points/<dataset>/). From repo root:
#   sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh
# Override splits via env, e.g.:
#   SPLITS=synth_vh/train,synth_vh/val,synth_vh/test sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh
#SBATCH --job-name=ppp_prepare_fg_points
#SBATCH --account=torch_pr_61_tandon_advanced
# --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs fisbe/fg_points

# Override splits via env, e.g.:
#   SPLITS=synth_vh/train,synth_vh/val,synth_vh/test sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh
SPLITS="${SPLITS:-partly/train,partly/val,partly/test}"

# Packed arrays (1k_v1, 1k_v1_u8) are symlinks into /synth_MCFO/<dataset>_ppp/
# and live in fisbe/sqf/<dataset>/*.sqf. Loose datasets have real directories.
# Same shard mount as resolve_data_overlays in ppp_job.sh; source images are
# not needed because this job only reads gt_fg_rm_5.
PPP_OVERLAY_ARGS=()
declare -A _ppp_mounted=()
declare -A _ppp_seen_ds=()
IFS=',' read -r -a _ppp_splits <<< "${SPLITS}"
for _split in "${_ppp_splits[@]}"; do
  _split="${_split// /}"
  [[ -n "${_split}" ]] || continue
  _ds="${_split%%/*}"
  [[ -n "${_ds}" ]] || continue
  [[ -n "${_ppp_seen_ds[${_ds}]:-}" ]] && continue
  _ppp_seen_ds["${_ds}"]=1
  _need_ppp=0
  for _sp in train val test; do
    for _link in "fisbe/${_ds}/${_sp}/"*.zarr/volumes/gt_fg_rm_5; do
      [[ -L "${_link}" ]] || continue
      _target="$(readlink "${_link}")"
      if [[ "${_target}" == /synth_MCFO/* ]]; then
        _need_ppp=1
      fi
    done
  done
  if [[ "${_need_ppp}" -eq 0 ]]; then
    continue
  fi
  _shard_dir="fisbe/sqf/${_ds}"
  _shard_found=0
  for _shard in "${_shard_dir}/"*.sqf; do
    [[ -f "${_shard}" ]] || continue
    _shard_found=1
    _shard="$(realpath -e "${_shard}")"
    if [[ -n "${_ppp_mounted[${_shard}]:-}" ]]; then
      continue
    fi
    _ppp_mounted["${_shard}"]=1
    PPP_OVERLAY_ARGS+=(--overlay "${_shard}:ro")
    echo "PPP_SQF=${_shard}"
  done
  if [[ "${_shard_found}" -eq 0 ]]; then
    echo "error: ${_ds} prepared-array symlink points at /synth_MCFO/ but no image matched ${_shard_dir}/*.sqf" >&2
    exit 1
  fi
done

singularity exec \
  --overlay env/ppp.ext3:ro \
  "${PPP_OVERLAY_ARGS[@]}" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; \
  python -u PatchPerPix/experiments/flylight/prepare_fg_points_csv.py \
    --fisbe-root fisbe \
    --out-dir fisbe/fg_points \
    --splits ${SPLITS} \
    --stride 4 \
    --overwrite"
