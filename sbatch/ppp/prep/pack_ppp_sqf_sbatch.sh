#!/bin/bash
# Prepare PPP arrays on node-local disk and pack each batch into one SquashFS
# image. Scratch keeps the build_synth skeleton plus symlinks into the image.
# From repo root (this invocation counts samples and submits the array):
#   DATASETS=1k_v1 sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh
#   CLIPMAX=255 DATASETS=1k_v1_u8 sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh
#   BATCH=40 OVERWRITE=1 DATASETS=1k_v1 sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh
# Rebuild skeletons first if fisbe/<dataset> is missing:
#   SQF=/scratch/vh2308/data/synth_MCFO/1k_v1.sqf \
#     sbatch sbatch/ppp/prep/synth/build_synth_from_sqf_sbatch.sh
#SBATCH --job-name=ppp_pack_sqf
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

DATASETS="${DATASETS:-}"
BATCH="${BATCH:-40}"
OVERWRITE="${OVERWRITE:-0}"
CLIPMAX="${CLIPMAX:-1500}"
SQF_ROOT="/scratch/vh2308/data/synth_MCFO"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
REPO_ROOT="/scratch/wmz2007/neuroinfo_fruitfly"
PPP_ARRAYS=(raw_normalized gt_instances_rm_5 gt_numinst gt_fg_rm_5)

if [[ -z "${DATASETS}" ]]; then
  echo "DATASETS is required (one folder name under fisbe/, e.g. 1k_v1)" >&2
  exit 1
fi
if [[ "${DATASETS}" == *","* || "${DATASETS}" == *"+"* || "${DATASETS}" == *" "* ]]; then
  echo "DATASETS must be a single name, not a list: ${DATASETS}" >&2
  exit 1
fi
if ! [[ "${DATASETS}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "DATASETS is not a usable folder name: ${DATASETS}" >&2
  exit 1
fi
if ! [[ "${BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH must be a positive integer: ${BATCH}" >&2
  exit 1
fi
if ! [[ "${CLIPMAX}" =~ ^[0-9]+$ ]]; then
  echo "CLIPMAX must be a non-negative integer: ${CLIPMAX}" >&2
  exit 1
fi
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1: ${OVERWRITE}" >&2
  exit 1
fi

# Flat sample list: "split name.zarr", train then val then test, each sorted.
list_dataset_samples() {
  local ds="$1" split d z
  for split in train val test; do
    d="${REPO_ROOT}/fisbe/${ds}/${split}"
    [[ -d "${d}" ]] || continue
    for z in "${d}/"*.zarr; do
      [[ -d "${z}" ]] || continue
      echo "${split} $(basename "${z}")"
    done
  done
}

# Point the four prepared arrays at the packed image. Removes a leftover
# real directory from a loose prepare so ln can replace it.
plant_symlinks() {
  local split="$1" name="$2" arr link target vol
  vol="${REPO_ROOT}/fisbe/${DATASETS}/${split}/${name}/volumes"
  if [[ ! -d "${vol}" ]]; then
    echo "missing skeleton ${vol}; run build_synth_from_sqf first" >&2
    exit 1
  fi
  for arr in "${PPP_ARRAYS[@]}"; do
    link="${vol}/${arr}"
    target="/synth_MCFO/${DATASETS}_ppp/${split}/${name}/volumes/${arr}"
    if [[ -L "${link}" ]]; then
      rm -f "${link}"
    elif [[ -e "${link}" ]]; then
      rm -rf "${link}"
    fi
    ln -s "${target}" "${link}"
  done
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ ! -f "${SQF_ROOT}/${DATASETS}.sqf" ]]; then
    echo "missing source image ${SQF_ROOT}/${DATASETS}.sqf" >&2
    exit 1
  fi
  mapfile -t ALL_SAMPLES < <(list_dataset_samples "${DATASETS}")
  n=${#ALL_SAMPLES[@]}
  if (( n == 0 )); then
    echo "No zarrs under fisbe/${DATASETS}/{train,val,test}." >&2
    echo "Rebuild with: SQF=${SQF_ROOT}/${DATASETS}.sqf sbatch sbatch/ppp/prep/synth/build_synth_from_sqf_sbatch.sh" >&2
    exit 1
  fi
  n_batches=$(( (n + BATCH - 1) / BATCH ))
  echo "Submitting array 0-$((n_batches - 1)) for ${n} samples (BATCH=${BATCH})"
  sbatch --array="0-$((n_batches - 1))" \
    --export=ALL,DATASETS="${DATASETS}",CLIPMAX="${CLIPMAX}",BATCH="${BATCH}",OVERWRITE="${OVERWRITE}" \
    sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh
  exit 0
fi

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "SLURM_TMPDIR is unset" >&2
  exit 1
fi

SRC_SQF="${SQF_ROOT}/${DATASETS}.sqf"
if [[ ! -f "${SRC_SQF}" ]]; then
  echo "missing source image ${SRC_SQF}" >&2
  exit 1
fi

mapfile -t ALL_SAMPLES < <(list_dataset_samples "${DATASETS}")
start=$(( SLURM_ARRAY_TASK_ID * BATCH ))
if (( start >= ${#ALL_SAMPLES[@]} )); then
  echo "task ${SLURM_ARRAY_TASK_ID} past ${#ALL_SAMPLES[@]} samples; nothing to do"
  exit 0
fi
end=$(( start + BATCH ))
if (( end > ${#ALL_SAMPLES[@]} )); then
  end=${#ALL_SAMPLES[@]}
fi

SQF_DIR="${REPO_ROOT}/fisbe/sqf/${DATASETS}"
mkdir -p "${SQF_DIR}"
OUT_SQF="${SQF_DIR}/batch_$(printf '%04d' "${SLURM_ARRAY_TASK_ID}").sqf"
echo "batch ${SLURM_ARRAY_TASK_ID}: samples ${start}-$((end - 1)) -> ${OUT_SQF}"

if [[ -f "${OUT_SQF}" && "${OVERWRITE}" != "1" ]]; then
  echo "skip existing ${OUT_SQF}"
  for (( i = start; i < end; i++ )); do
    read -r split name <<< "${ALL_SAMPLES[$i]}"
    plant_symlinks "${split}" "${name}"
  done
  exit 0
fi

WORK="${SLURM_TMPDIR}/work"
PACK_ROOT="${SLURM_TMPDIR}/synth_MCFO/${DATASETS}_ppp"
rm -rf "${SLURM_TMPDIR}/work" "${SLURM_TMPDIR}/synth_MCFO"
mkdir -p "${WORK}/${DATASETS}" "${PACK_ROOT}"

for (( i = start; i < end; i++ )); do
  read -r split name <<< "${ALL_SAMPLES[$i]}"
  src="${REPO_ROOT}/fisbe/${DATASETS}/${split}/${name}"
  dst="${WORK}/${DATASETS}/${split}/${name}"
  if [[ ! -L "${src}/volumes/raw" || ! -L "${src}/volumes/gt_instances" ]]; then
    echo "expected raw and gt_instances symlinks: ${src}/volumes" >&2
    exit 1
  fi
  mkdir -p "${dst}/volumes"
  for meta in .zgroup .zattrs; do
    if [[ -f "${src}/${meta}" ]]; then
      cp -a "${src}/${meta}" "${dst}/${meta}"
    fi
    if [[ -f "${src}/volumes/${meta}" ]]; then
      cp -a "${src}/volumes/${meta}" "${dst}/volumes/${meta}"
    fi
  done
  cp -a "${src}/volumes/raw" "${src}/volumes/gt_instances" "${dst}/volumes/"
done

singularity exec \
  --overlay env/ppp.ext3:ro \
  --overlay "${SRC_SQF}:ro" \
  --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" \
  "${SIF}" \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; \
    if [[ ! -d /synth_MCFO/${DATASETS}/samples ]]; then \
      echo \"missing /synth_MCFO/${DATASETS}/samples\" >&2; exit 1; \
    fi; \
    python -u PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py \
      --fisbe-root ${WORK} --opening-radius 1 --datasets ${DATASETS} \
      --clipmax ${CLIPMAX} --overwrite"

for (( i = start; i < end; i++ )); do
  read -r split name <<< "${ALL_SAMPLES[$i]}"
  src_vol="${WORK}/${DATASETS}/${split}/${name}/volumes"
  dst_vol="${PACK_ROOT}/${split}/${name}/volumes"
  mkdir -p "${dst_vol}"
  for arr in "${PPP_ARRAYS[@]}"; do
    if [[ ! -f "${src_vol}/${arr}/.zarray" ]]; then
      echo "prepare did not write ${src_vol}/${arr}" >&2
      exit 1
    fi
    cp -a "${src_vol}/${arr}" "${dst_vol}/${arr}"
  done
done

find "${SLURM_TMPDIR}/synth_MCFO" -type d -exec chmod 755 {} \;
find "${SLURM_TMPDIR}/synth_MCFO" -type f -exec chmod 644 {} \;

# Bundled binary is a symlink to .wrapper, so match the path rather than -type f.
MKSQUASHFS="$(ls -d /share/apps/apptainer/*/x86_64/utils/bin/mksquashfs 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "${MKSQUASHFS}" ]]; then
  MKSQUASHFS="$(command -v mksquashfs || true)"
fi
if [[ -z "${MKSQUASHFS}" ]]; then
  echo "mksquashfs not found under /share/apps/apptainer or on PATH" >&2
  exit 1
fi
echo "mksquashfs ${MKSQUASHFS}"

TMP_SQF="${OUT_SQF}.tmp"
rm -f "${TMP_SQF}"
"${MKSQUASHFS}" "${SLURM_TMPDIR}/synth_MCFO" "${TMP_SQF}" \
  -keep-as-directory -mem 16G -processors "${SLURM_CPUS_PER_TASK:-4}"
mv "${TMP_SQF}" "${OUT_SQF}"

for (( i = start; i < end; i++ )); do
  read -r split name <<< "${ALL_SAMPLES[$i]}"
  plant_symlinks "${split}" "${name}"
done

rm -rf "${SLURM_TMPDIR}/work" "${SLURM_TMPDIR}/synth_MCFO"
echo "packed ${OUT_SQF}"
