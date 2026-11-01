#!/bin/bash
# Inventory or MIP sqf-backed datasets whose raw/gt arrays are symlinks into
# /synth_MCFO/<name>/. Those links resolve only while the matching .sqf is
# mounted. From repo root:
#   DATASETS=1k_v1,1k_v1_u8 \
#     sbatch sbatch/ppp/prep/synth/inspect_sqf_dataset_sbatch.sh
#   MODE=mip LIMIT=3 DATASETS=1k_v1 \
#     sbatch sbatch/ppp/prep/synth/inspect_sqf_dataset_sbatch.sh
#   MODE=mip SAMPLE=brain_00004 SPLITS=train DATASETS=1k_v1_u8 \
#     sbatch sbatch/ppp/prep/synth/inspect_sqf_dataset_sbatch.sh
# Smoke-test one val brain before scanning both datasets. Prepared arrays
# (raw_normalized, gt_instances_rm_5, gt_numinst, gt_fg_rm_5) are read from
# fisbe/sqf/<dataset>/*.sqf when those symlinks point at /synth_MCFO/.
#SBATCH --job-name=ppp_inspect_sqf
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs imaging_helpers_hpc/output

DATASETS="${DATASETS:-1k_v1,1k_v1_u8}"
MODE="${MODE:-inventory}"
SPLITS="${SPLITS:-train val test}"
LIMIT="${LIMIT:-3}"
SAMPLE="${SAMPLE:-}"
DPI="${DPI:-100}"
SQF_ROOT="/scratch/vh2308/data/synth_MCFO"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

if [[ "${MODE}" != "inventory" && "${MODE}" != "mip" ]]; then
  echo "MODE must be inventory or mip: ${MODE}" >&2
  exit 1
fi
if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || (( LIMIT < 1 )); then
  echo "LIMIT must be an integer >= 1: ${LIMIT}" >&2
  exit 1
fi
if ! [[ "${DPI}" =~ ^[0-9]+$ ]] || (( DPI < 1 )); then
  echo "DPI must be an integer >= 1: ${DPI}" >&2
  exit 1
fi

SPLITS="${SPLITS//,/ }"
read -r -a split_arr <<< "${SPLITS}"
if [[ ${#split_arr[@]} -eq 0 ]]; then
  echo "SPLITS is empty" >&2
  exit 1
fi
for split in "${split_arr[@]}"; do
  case "${split}" in
    train|val|test) ;;
    *)
      echo "SPLITS entries must be train, val, or test: ${split}" >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a dataset_names <<< "${DATASETS}"
if [[ ${#dataset_names[@]} -eq 0 ]]; then
  echo "DATASETS is empty" >&2
  exit 1
fi

root_real=$(realpath -e "${SQF_ROOT}")
SQF_OVERLAY_ARGS=()
declare -A seen_sqf=()
clean_names=()
for ds in "${dataset_names[@]}"; do
  ds="${ds// /}"
  [[ -n "${ds}" ]] || continue
  if ! [[ "${ds}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "dataset name is not usable: ${ds}" >&2
    exit 1
  fi
  sqf="${SQF_ROOT}/${ds}.sqf"
  if [[ ! -f "${sqf}" ]]; then
    echo "missing source image: ${sqf}" >&2
    exit 1
  fi
  sqf_real=$(realpath -e "${sqf}")
  case "${sqf_real}" in
    "${root_real}"/*) ;;
    *)
      echo "SQF must be under ${SQF_ROOT}: ${sqf_real}" >&2
      exit 1
      ;;
  esac
  clean_names+=("${ds}")
  if [[ -z "${seen_sqf[${sqf_real}]:-}" ]]; then
    seen_sqf["${sqf_real}"]=1
    SQF_OVERLAY_ARGS+=(--overlay "${sqf_real}:ro")
    echo "SQF=${sqf_real} mount=/synth_MCFO/${ds}"
  fi

  # Packed PPP arrays are symlinks into /synth_MCFO/<dataset>_ppp/.
  PPP_LINK_ARRAYS=(raw_normalized gt_instances_rm_5 gt_numinst gt_fg_rm_5)
  need_ppp=0
  for split in train val test; do
    for arr in "${PPP_LINK_ARRAYS[@]}"; do
      for link in "fisbe/${ds}/${split}/"*.zarr/volumes/"${arr}"; do
        [[ -L "${link}" ]] || continue
        target="$(readlink "${link}")"
        if [[ "${target}" == /synth_MCFO/* ]]; then
          need_ppp=1
        fi
      done
    done
  done
  shard_dir="fisbe/sqf/${ds}"
  shard_found=0
  for shard in "${shard_dir}/"*.sqf; do
    [[ -f "${shard}" ]] || continue
    shard_found=1
    shard_real="$(realpath -e "${shard}")"
    if [[ -n "${seen_sqf[${shard_real}]:-}" ]]; then
      continue
    fi
    seen_sqf["${shard_real}"]=1
    SQF_OVERLAY_ARGS+=(--overlay "${shard_real}:ro")
    echo "PPP_SQF=${shard_real}"
  done
  if [[ "${need_ppp}" -eq 1 && "${shard_found}" -eq 0 ]]; then
    echo "error: ${ds} prepared-array symlink points at /synth_MCFO/ but no image matched ${shard_dir}/*.sqf" >&2
    echo "DATASETS=${ds} sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh" >&2
    exit 1
  fi
done
if [[ ${#clean_names[@]} -eq 0 ]]; then
  echo "DATASETS is empty" >&2
  exit 1
fi

DATASETS_CLEAN=$(IFS=','; echo "${clean_names[*]}")
echo "MODE=${MODE}"
echo "DATASETS=${DATASETS_CLEAN}"
echo "SPLITS=${SPLITS}"

singularity exec \
  --overlay env/ppp.ext3:ro \
  "${SQF_OVERLAY_ARGS[@]}" \
  "${SIF}" \
  /bin/bash -s -- \
    "${MODE}" "${DATASETS_CLEAN}" "${SPLITS}" "${LIMIT}" "${SAMPLE}" "${DPI}" <<'EOF'
set -e
source /ext3/env.sh
conda activate ppp
set -uo pipefail
cd /scratch/wmz2007/neuroinfo_fruitfly
export MPLBACKEND=Agg
MODE="$1"
DATASETS="$2"
SPLITS="$3"
LIMIT="$4"
SAMPLE="$5"
DPI="$6"

IFS=',' read -r -a names <<< "${DATASETS}"
PPP_LINK_ARRAYS=(raw_normalized gt_instances_rm_5 gt_numinst gt_fg_rm_5)
for name in "${names[@]}"; do
  if [[ ! -d "/synth_MCFO/${name}/samples" ]]; then
    echo "missing /synth_MCFO/${name}/samples (sqf did not mount at the expected path)" >&2
    echo "Mount /scratch/vh2308/data/synth_MCFO/${name}.sqf read-only via this sbatch script." >&2
    exit 1
  fi
  if [[ "${MODE}" != "mip" ]]; then
    continue
  fi
  need_ppp=0
  for split in train val test; do
    for arr in "${PPP_LINK_ARRAYS[@]}"; do
      for link in "fisbe/${name}/${split}/"*.zarr/volumes/"${arr}"; do
        [[ -L "${link}" ]] || continue
        target="$(readlink "${link}")"
        if [[ "${target}" == /synth_MCFO/* ]]; then
          need_ppp=1
        fi
      done
    done
  done
  if [[ "${need_ppp}" -eq 1 && ! -d "/synth_MCFO/${name}_ppp" ]]; then
    echo "missing /synth_MCFO/${name}_ppp (packed PPP sqf did not mount)" >&2
    echo "DATASETS=${name} sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh" >&2
    exit 1
  fi
done

read -r -a split_arr <<< "${SPLITS}"
args=(
  python -u -m imaging_helpers_hpc dataset "${MODE}"
  --datasets "${DATASETS}"
  --splits "${split_arr[@]}"
)
if [[ -n "${SAMPLE}" ]]; then
  args+=(--sample "${SAMPLE}")
fi
if [[ "${MODE}" == "mip" ]]; then
  args+=(--limit "${LIMIT}" --dpi "${DPI}")
fi
"${args[@]}"
EOF
