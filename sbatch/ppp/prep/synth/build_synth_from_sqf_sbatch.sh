#!/bin/bash
# Build fisbe/<name>/{train,val,test} from a synth_MCFO SquashFS image (CPU).
# volumes/raw and volumes/gt_instances are symlinks into the image mount
# (/synth_MCFO/<name>). They resolve only in a job that mounts the same .sqf.
# From repo root:
#   SQF=/scratch/vh2308/data/synth_MCFO/1k_v1_u8.sqf \
#     sbatch sbatch/ppp/prep/synth/build_synth_from_sqf_sbatch.sh
#   SQF=... OUT=fisbe/custom RATIOS="0.8 0.1 0.1" SEED=0 \
#     sbatch sbatch/ppp/prep/synth/build_synth_from_sqf_sbatch.sh
#SBATCH --job-name=ppp_build_synth_sqf
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs

SQF_ROOT="/scratch/vh2308/data/synth_MCFO"
if [[ -z "${SQF:-}" ]]; then
  echo "SQF is required (an existing *.sqf under ${SQF_ROOT})" >&2
  exit 1
fi
if [[ "${SQF}" != *.sqf || ! -f "${SQF}" ]]; then
  echo "SQF must be an existing .sqf file: ${SQF}" >&2
  exit 1
fi

sqf_real=$(realpath -e "${SQF}")
root_real=$(realpath -e "${SQF_ROOT}")
case "${sqf_real}" in
  "${root_real}"/*) ;;
  *)
    echo "SQF must be under ${SQF_ROOT}: ${sqf_real}" >&2
    exit 1
    ;;
esac

name=$(basename "${sqf_real}" .sqf)
if ! [[ "${name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "SQF basename is not a usable dataset name: ${name}" >&2
  exit 1
fi

OUT="${OUT:-fisbe/${name}}"
if ! [[ "${OUT}" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "OUT has unsupported characters: ${OUT}" >&2
  exit 1
fi

RATIOS="${RATIOS:-0.8 0.1 0.1}"
SEED="${SEED:-0}"
read -r -a ratio_arr <<< "${RATIOS}"
if [[ ${#ratio_arr[@]} -ne 3 ]]; then
  echo "RATIOS must be three floats (train val test): ${RATIOS}" >&2
  exit 1
fi
for r in "${ratio_arr[@]}"; do
  if ! [[ "${r}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "RATIOS must be three floats (train val test): ${RATIOS}" >&2
    exit 1
  fi
done
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer: ${SEED}" >&2
  exit 1
fi

MOUNT="/synth_MCFO/${name}"
echo "SQF=${sqf_real}"
echo "OUT=${OUT}"
echo "mount=${MOUNT}"

singularity exec \
  --overlay env/ppp.ext3:ro \
  --overlay "${sqf_real}:ro" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -s -- "${MOUNT}" "${OUT}" "${SEED}" \
    "${ratio_arr[0]}" "${ratio_arr[1]}" "${ratio_arr[2]}" <<'EOF'
set -e
source /ext3/env.sh
conda activate ppp
set -uo pipefail
MOUNT="$1"
OUT="$2"
SEED="$3"
R0="$4"
R1="$5"
R2="$6"
if [[ ! -d "${MOUNT}/samples" ]]; then
  echo "missing ${MOUNT}/samples (sqf did not mount at the expected path)" >&2
  exit 1
fi
args=(
  python -u PatchPerPix/experiments/flylight/build_synth_vh.py
  --src "${MOUNT}/samples"
  --out "${OUT}"
  --ratios "${R0}" "${R1}" "${R2}"
  --seed "${SEED}"
)
if [[ -f "${MOUNT}/sample_list_per_split.txt" ]]; then
  echo "using ${MOUNT}/sample_list_per_split.txt"
  args+=(--sample-list "${MOUNT}/sample_list_per_split.txt")
else
  echo "no sample list under ${MOUNT}; ratios ${R0} ${R1} ${R2} seed ${SEED}"
fi
"${args[@]}"
EOF
