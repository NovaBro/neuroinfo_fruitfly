#!/bin/bash
# Generate the simple_v1 synthetic tube dataset as SquashFS parts (CPU only).
# Plan: md_guides/dataset/simple_dataset_plan.md §5. Submit from the repo root:
#   sbatch sbatch/ppp/prep/simple/generate_simple_sbatch.sh
#   DATASET=simple_v1_dry TOTAL=60 SPLITS=40,10,10 PARTS=1 sbatch sbatch/ppp/prep/simple/generate_simple_sbatch.sh
#
# Without SLURM_ARRAY_TASK_ID (the plain sbatch above) this job writes fisbe/<DATASET>
# (train/ and meta/ link into /<DATASET>, loose val/test zarr skeletons, sample list,
# source_sqf.txt) and links fisbe/fg_points/<DATASET> -> /<DATASET>/fg_points, then submits
# array 0..PARTS-1. Task i renders its share of the indices on $SLURM_TMPDIR (verify on),
# with every PatchPerPix array, meta sidecars and fg point CSVs inside the image, packs
# fisbe/sqf/<DATASET>_src/part_<i>.sqf (image root <DATASET>/, mounts at /<DATASET>) and
# verifies the part again through the mount. No prepare_fisbe_for_ppp / pack / fg-points
# job is needed afterwards: train.py and predict_no_gp.py read the uint8 raw directly.
#
# Env: DATASET (simple_v1) TOTAL (50200) SPLITS (50000,100,100; must sum to TOTAL)
#      PARTS (20) SEED (0) OVERWRITE (0: skip parts that exist) WORKERS (cpus-per-task)
#      CONFETTI_SPLITS (val,test: train gets confetti on the GPU in training)
#      MIP_SPLITS (val,test)
#SBATCH --job-name=simple_tubes_gen
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs

DATASET="${DATASET:-simple_v1}"
TOTAL="${TOTAL:-50200}"
SPLITS="${SPLITS:-50000,100,100}"
PARTS="${PARTS:-20}"
CONFETTI_SPLITS="${CONFETTI_SPLITS:-val,test}"
MIP_SPLITS="${MIP_SPLITS:-val,test}"
SEED="${SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
REPO_ROOT="/scratch/wmz2007/neuroinfo_fruitfly"
DS_DIR="${REPO_ROOT}/fisbe/${DATASET}"
SRC_DIR="${REPO_ROOT}/fisbe/sqf/${DATASET}_src"

if ! [[ "${DATASET}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "DATASET is not a usable folder name: ${DATASET}" >&2
  exit 1
fi
if [[ "${DATASET}" == synth_MCFO* ]]; then
  echo "DATASET must not mount under /synth_MCFO (ppp_job.sh maps that to vh2308's images)" >&2
  exit 1
fi
IFS=, read -r n_train n_val n_test <<< "${SPLITS}"
if (( n_train + n_val + n_test != TOTAL )); then
  echo "SPLITS=${SPLITS} does not sum to TOTAL=${TOTAL}" >&2
  exit 1
fi
if (( PARTS < 1 || PARTS > 20 )); then
  echo "PARTS=${PARTS}: keep it small (each part is one --overlay; ~50 overlays breaks apptainer)" >&2
  exit 1
fi

in_container() {
  # in_container <extra singularity args...> -- <bash command>
  local extra=()
  while [[ "$1" != "--" ]]; do extra+=("$1"); shift; done
  shift
  singularity exec --overlay env/ppp.ext3:ro "${extra[@]}" "${SIF}" /bin/bash -c \
    "source /ext3/env.sh; conda activate ppp; cd ${REPO_ROOT}; \
     export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; $1"
}

PART_SQFS=()
for (( i = 0; i < PARTS; i++ )); do
  PART_SQFS+=("${SRC_DIR}/part_${i}.sqf")
done

# ---- submitter: skeleton + sample list + source_sqf.txt, then the array ----
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  mkdir -p "${SRC_DIR}"
  skel_extra=""
  [[ "${OVERWRITE}" == "1" ]] && skel_extra="--overwrite"
  in_container -- "python -u -m synthetic.simple_tubes.generate skeleton \
      --dataset-dir ${DS_DIR} --dataset ${DATASET} --splits ${SPLITS} \
      --fg-dir ${REPO_ROOT}/fisbe/fg_points/${DATASET} \
      --sources ${PART_SQFS[*]} ${skel_extra}"
  echo "Submitting array 0-$((PARTS - 1)) for ${TOTAL} samples (${SPLITS})"
  # SPLITS holds commas, which --export=VAR=... would split, so export and pass ALL.
  export DATASET TOTAL SPLITS PARTS SEED OVERWRITE CONFETTI_SPLITS MIP_SPLITS
  sbatch --array="0-$((PARTS - 1))" --export=ALL \
    sbatch/ppp/prep/simple/generate_simple_sbatch.sh
  exit 0
fi

# ---- array task: render, pack, verify through the mount ----
if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "SLURM_TMPDIR is unset" >&2
  exit 1
fi
per=$(( (TOTAL + PARTS - 1) / PARTS ))
start=$(( SLURM_ARRAY_TASK_ID * per ))
count=$(( start + per > TOTAL ? TOTAL - start : per ))
OUT_SQF="${PART_SQFS[${SLURM_ARRAY_TASK_ID}]}"
echo "part ${SLURM_ARRAY_TASK_ID}: samples ${start}-$((start + count - 1)) -> ${OUT_SQF}"
if (( count <= 0 )); then
  echo "nothing to do"
  exit 0
fi
if [[ -f "${OUT_SQF}" && "${OVERWRITE}" != "1" ]]; then
  echo "skip existing ${OUT_SQF} (OVERWRITE=1 to rebuild)"
  exit 0
fi

PACK="${SLURM_TMPDIR}/pack"
rm -rf "${PACK}"
mkdir -p "${PACK}/${DATASET}"

in_container --bind "${SLURM_TMPDIR}:${SLURM_TMPDIR}" -- \
  "python -u -m synthetic.simple_tubes.generate render --dataset ${DATASET} \
     --out ${PACK}/${DATASET} --meta-out ${PACK}/${DATASET}/meta \
     --fg-out ${PACK}/${DATASET}/fg_points \
     --confetti-splits ${CONFETTI_SPLITS} --mip-splits ${MIP_SPLITS} \
     --start ${start} --count ${count} --seed ${SEED} --splits ${SPLITS} \
     --workers ${WORKERS} --verify"

find "${PACK}" -type d -exec chmod 755 {} \;
find "${PACK}" -type f -exec chmod 644 {} \;

# Bundled binary is a symlink to .wrapper, so match the path rather than -type f.
MKSQUASHFS="$(ls -d /share/apps/apptainer/*/x86_64/utils/bin/mksquashfs 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "${MKSQUASHFS}" ]]; then
  MKSQUASHFS="$(command -v mksquashfs || true)"
fi
if [[ -z "${MKSQUASHFS}" ]]; then
  echo "mksquashfs not found under /share/apps/apptainer or on PATH" >&2
  exit 1
fi
mkdir -p "${SRC_DIR}"
TMP_SQF="${OUT_SQF%.sqf}.tmp.sqf"   # keep .sqf so apptainer can mount it for the check
rm -f "${TMP_SQF}"
"${MKSQUASHFS}" "${PACK}/${DATASET}" "${TMP_SQF}" -keep-as-directory -noappend \
  -mem 8G -processors "${WORKERS}"

# Verify again through the mounted image before publishing it.
in_container --overlay "${TMP_SQF}:ro" -- \
  "python -u -m synthetic.simple_tubes.verify /${DATASET} --meta-dir /${DATASET}/meta \
     --fg-dir /${DATASET}/fg_points --workers ${WORKERS}"
mv "${TMP_SQF}" "${OUT_SQF}"
rm -rf "${PACK}"
echo "packed ${OUT_SQF} ($(du -h "${OUT_SQF}" | cut -f1))"
