#!/bin/bash
# Prepare FISBe zarrs with PPP training arrays. From repo root:
#   sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh
#   DATASETS=partly sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh
#   OVERWRITE=0 DATASETS=partly sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh
#   DATASETS=completely,partly,synth_vh sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh
# Large sqf-backed sets (1k_v1, 1k_v1_u8) must use pack_ppp_sqf_sbatch.sh.
# Loose prepare of those writes one file per chunk and hits the scratch quota.
#SBATCH --job-name=ppp_prepare_fisbe
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs

DATASETS="${DATASETS:-partly}"
OVERWRITE="${OVERWRITE:-1}"
CLIPMAX="${CLIPMAX:-1500}"
ALLOW_LOOSE="${ALLOW_LOOSE:-0}"

# 1k-scale sqf datasets write ~8k chunk files per brain. Pack those instead.
if [[ "${ALLOW_LOOSE}" != "1" ]]; then
  IFS=',' read -r -a _loose_datasets <<< "${DATASETS}"
  for _ds in "${_loose_datasets[@]}"; do
    _ds="${_ds// /}"
    [[ -n "${_ds}" ]] || continue
    [[ -f "/scratch/vh2308/data/synth_MCFO/${_ds}.sqf" ]] || continue
    _n=0
    for _split in train val test; do
      _dir="fisbe/${_ds}/${_split}"
      [[ -d "${_dir}" ]] || continue
      for _z in "${_dir}/"*.zarr; do
        [[ -d "${_z}" ]] && _n=$((_n + 1))
      done
    done
    if (( _n > 50 )); then
      echo "Refusing loose prepare of ${_ds} (${_n} zarrs; source /scratch/vh2308/data/synth_MCFO/${_ds}.sqf)." >&2
      echo "That writes one file per 64^3 chunk onto scratch and exceeds the file quota." >&2
      echo "Use: DATASETS=${_ds} sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh" >&2
      echo "Override with ALLOW_LOOSE=1." >&2
      exit 1
    fi
  done
fi

OVERWRITE_FLAG=""
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

# shellcheck source=sbatch/ppp/prep/sqf_overlays.sh
source sbatch/ppp/prep/sqf_overlays.sh
sqf_resolve_overlays

singularity exec \
  --overlay env/ppp.ext3:ro \
  "${SQF_OVERLAY_ARGS[@]}" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; ${SQF_MOUNT_CHECK}\
  python -u PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py \
    --fisbe-root fisbe --opening-radius 1 --datasets ${DATASETS} \
    --clipmax ${CLIPMAX} ${OVERWRITE_FLAG}"
