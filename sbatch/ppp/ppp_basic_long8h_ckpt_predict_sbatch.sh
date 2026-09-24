#!/bin/bash
# Predict one val sample for a given checkpoint into test/processed/<CKPT>/.
# Submit: CKPT=20000 sbatch --export=ALL,CKPT sbatch/ppp/ppp_basic_long8h_ckpt_predict_sbatch.sh
#SBATCH --job-name=ppp_basic_long8h_ckpt_predict
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/ppp_basic_long8h_ckpt_predict_%j.out
#SBATCH --error=sbatch/ppp/ppp_basic_long8h_ckpt_predict_%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  echo "Error: set CKPT (e.g. 20000)" >&2
  exit 1
fi
echo "ckpt_predict ckpt=${CKPT} job=${SLURM_JOB_ID:-local}"

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
  > "gpu_usage_log_ppp_basic_long8h_ckpt_predict_${CKPT}.csv" &
GPU_LOGGER_PID=$!

singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_long8h.toml \
    --test-data /scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/val \
    -d predict \
    --app flylight --root ../../metrics/ppp -id ppp_basic_long8h \
    --checkpoint ${CKPT} \
    --sample R22C03-20180918_66_J2"

# Fast FG/numinst metrics (mid-aff) while still on the compute node.
singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; \
  python -u sbatch/ppp/diagnose_ppp_pred_metrics.py \
    --pred-zarr metrics/ppp/ppp_basic_long8h/test/processed/${CKPT}/R22C03-20180918_66_J2.zarr"
