#!/bin/bash
# Per-sample vote_instances (label only) for ppp_basic_long8h @ ckpt 70000.
# Protocol: vi_th_0_3 + eval_rm0 + label_resume (skip existing HDF).
# Requires processed/70000/<sample>.zarr. Evaluate separately on CPU.
#
# Submit from repo root (array bounds set by chain or CLI):
#   sbatch --export=ALL,SPLIT=val  --array=0-4%2 sbatch/ppp/ppp_basic_long8h_label_array_sbatch.sh
#   sbatch --export=ALL,SPLIT=test --array=0-6%2 sbatch/ppp/ppp_basic_long8h_label_array_sbatch.sh
#SBATCH --job-name=ppp_basic_long8h_label
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/array/ppp_basic_long8h_label_%A_%a.out
#SBATCH --error=sbatch/ppp/array/ppp_basic_long8h_label_%A_%a.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

SPLIT="${SPLIT:-}"
if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
  echo "Error: set SPLIT=val or SPLIT=test (e.g. --export=ALL,SPLIT=val)" >&2
  exit 1
fi

DATA_DIR="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/${SPLIT}"
PROC_DIR="metrics/ppp/ppp_basic_long8h/test/processed/70000"
CKPT=70000

mapfile -t SAMPLES < <(find "${DATA_DIR}" -maxdepth 1 -name '*.zarr' -printf '%f\n' \
  | sed 's/\.zarr$//' | sort)
if (( SLURM_ARRAY_TASK_ID >= ${#SAMPLES[@]} )); then
  echo "task ${SLURM_ARRAY_TASK_ID} >= ${#SAMPLES[@]} samples; nothing to do"
  exit 0
fi
SAMPLE=${SAMPLES[${SLURM_ARRAY_TASK_ID}]}
PRED_ZARR="${PROC_DIR}/${SAMPLE}.zarr"
echo "[task ${SLURM_ARRAY_TASK_ID}] split=${SPLIT} sample=${SAMPLE} ckpt=${CKPT}"

if [[ ! -d "${PRED_ZARR}" ]]; then
  echo "Error: missing prediction zarr: ${PRED_ZARR}" >&2
  exit 1
fi

mkdir -p sbatch/ppp/array
GPU_LOGGER_PID=
cleanup() {
  if [[ -n "${GPU_LOGGER_PID}" ]]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nvidia-smi \
  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 3 \
  > "sbatch/ppp/array/gpu_usage_label_${SPLIT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
GPU_LOGGER_PID=$!

singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_long8h.toml \
    --config flylight/setups/setup01/vi_th_0_3.toml \
    --config flylight/setups/setup01/eval_rm0.toml \
    --config flylight/setups/setup01/label_resume.toml \
    --test-data ${DATA_DIR} \
    -d label \
    --app flylight --root ../../metrics/ppp -id ppp_basic_long8h \
    --checkpoint ${CKPT} \
    --sample ${SAMPLE}"

echo "[task ${SLURM_ARRAY_TASK_ID}] finished ${SAMPLE}"
