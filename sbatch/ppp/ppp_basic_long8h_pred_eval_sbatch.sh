#!/bin/bash
# Intermediate prediction metrics (fg / numinst / patch) for one checkpoint.
# Reuses test/processed/<CKPT>/ — no predict, no VI.
# Submit: CKPT=30000 sbatch --export=ALL,CKPT sbatch/ppp/ppp_basic_long8h_pred_eval_sbatch.sh
#SBATCH --job-name=ppp_basic_long8h_pred_eval
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192g
#SBATCH --time=04:00:00
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/ppp_basic_long8h_pred_eval_%j.out
#SBATCH --error=sbatch/ppp/ppp_basic_long8h_pred_eval_%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

CKPT="${CKPT:-30000}"
echo "pred_eval ckpt=${CKPT} job=${SLURM_JOB_ID:-local}"

# GPU requested so job can land on public partitions (CPU partitions often stalled).
# Work itself is CPU/RAM; keep a light nvidia-smi logger for audit.
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
  > "gpu_usage_log_ppp_basic_long8h_pred_eval_${CKPT}.csv" &
GPU_LOGGER_PID=$!

singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  env CUDA_VISIBLE_DEVICES=0 \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_long8h.toml \
    --config flylight/setups/setup01/eval_pred.toml \
    --test-data /scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/val \
    -d evaluate \
    --app flylight --root ../../metrics/ppp -id ppp_basic_long8h \
    --checkpoint ${CKPT} \
    --sample R22C03-20180918_66_J2"
