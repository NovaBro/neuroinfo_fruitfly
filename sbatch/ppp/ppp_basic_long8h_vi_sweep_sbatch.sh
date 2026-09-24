#!/bin/bash
# VI label and/or evaluate for one overlay (no predict). Reuses processed/<CKPT>.
# Submit via: bash sbatch/ppp/ppp_basic_long8h_vi_sweep_chain.sh
# Or alone: VI_OVERLAY=vi_th_0_3.toml CKPT=70000 sbatch --export=ALL,VI_OVERLAY,CKPT \
#            sbatch/ppp/ppp_basic_long8h_vi_sweep_sbatch.sh
# Evaluate-only (reuse existing HDF): DO_TASKS=evaluate ...
# Multi-task: DO_TASKS=label+evaluate  (do NOT use commas — unsafe in SLURM --export)
#SBATCH --job-name=ppp_basic_long8h_vi
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/ppp_basic_long8h_vi_%x_%j.out
#SBATCH --error=sbatch/ppp/ppp_basic_long8h_vi_%x_%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

VI_OVERLAY="${VI_OVERLAY:-}"
if [[ -z "${VI_OVERLAY}" ]]; then
  echo "Error: set VI_OVERLAY (e.g. vi_th_0_3.toml)" >&2
  exit 1
fi
VI_PATH="PatchPerPix/experiments/flylight/setups/setup01/${VI_OVERLAY}"
if [[ ! -f "${VI_PATH}" ]]; then
  echo "Error: overlay not found: ${VI_PATH}" >&2
  exit 1
fi
CKPT="${CKPT:-30000}"
# Task list for -d. Prefer '+' separator — commas break SLURM --export=A,B,C.
# Also accept commas/spaces for older callers.
DO_TASKS="${DO_TASKS:-label+evaluate}"
DO_TASKS_NORM="${DO_TASKS//+/ }"
DO_TASKS_NORM="${DO_TASKS_NORM//,/ }"
# shellcheck disable=SC2206
DO_ARR=(${DO_TASKS_NORM})
TAG="${VI_OVERLAY%.toml}"
echo "VI overlay=${VI_OVERLAY} tag=${TAG} ckpt=${CKPT} do=${DO_TASKS_NORM} job=${SLURM_JOB_ID:-local}"

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
  > "gpu_usage_log_ppp_basic_long8h_${TAG}.csv" &
GPU_LOGGER_PID=$!

# Build -d args as separate quoted tokens inside the singularity bash -c string.
DO_ARGS=""
for t in "${DO_ARR[@]}"; do
  DO_ARGS+=" ${t}"
done

singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_long8h.toml \
    --config flylight/setups/setup01/${VI_OVERLAY} \
    --config flylight/setups/setup01/eval_rm0.toml \
    --test-data /scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/val \
    -d${DO_ARGS} \
    --app flylight --root ../../metrics/ppp -id ppp_basic_long8h \
    --checkpoint ${CKPT} \
    --sample R22C03-20180918_66_J2"
