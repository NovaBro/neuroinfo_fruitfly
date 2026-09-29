#!/bin/bash
# One SLURM task per volume for ppp_basic_8h @ ckpt 8000 (th=0.3 + rm0).
# Prefer: bash sbatch/ppp/ppp_basic_8h_full_array_chain.sh
# Or: sbatch --export=ALL,SPLIT=val --array=0-4%2 sbatch/ppp/ppp_basic_8h_array_sbatch.sh
#
# Requires SPLIT=val|test and --array set at submit time (chain discovers N).
#SBATCH --job-name=ppp_8h_arr
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/array/8h_%A_%a.out
#SBATCH --error=sbatch/ppp/array/8h_%A_%a.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

SPLIT="${SPLIT:-}"
if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
  echo "Error: set SPLIT=val or SPLIT=test" >&2
  exit 1
fi

CKPT=8000
DATA_DIR="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/${SPLIT}"
PROCESSED="metrics/ppp/ppp_basic_8h/test/processed/${CKPT}"

mapfile -t SAMPLES < <(find "${DATA_DIR}" -maxdepth 1 -name '*.zarr' -printf '%f\n' \
                       | sed 's/\.zarr$//' | sort)
if (( SLURM_ARRAY_TASK_ID >= ${#SAMPLES[@]} )); then
  echo "task ${SLURM_ARRAY_TASK_ID} >= ${#SAMPLES[@]} samples; nothing to do"
  exit 0
fi
SAMPLE=${SAMPLES[${SLURM_ARRAY_TASK_ID}]}
SAMPLE_ZARR="${PROCESSED}/${SAMPLE}.zarr"
echo "[task ${SLURM_ARRAY_TASK_ID}] split=${SPLIT} ckpt=${CKPT} sample=${SAMPLE}"

GPU_LOGGER_PID=
cleanup() {
  if [[ -n "${GPU_LOGGER_PID}" ]]; then
    kill "${GPU_LOGGER_PID}" 2>/dev/null || true
    wait "${GPU_LOGGER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
mkdir -p gpu_log/ppp
nvidia-smi \
  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 3 \
  > "gpu_log/ppp/gpu_usage_8h_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
GPU_LOGGER_PID=$!

# Discard incomplete pred zarrs (complete ones are reused; R22C03 already done).
singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; \
    if [ -d '${SAMPLE_ZARR}' ] && \
       ! python3 sbatch/ppp/check_prediction_complete.py '${SAMPLE_ZARR}'; then \
      echo 'discarding partial prediction: ${SAMPLE_ZARR}'; \
      rm -rf '${SAMPLE_ZARR}'; \
    fi; \
    cd PatchPerPix/experiments/; \
    env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -u run_ppp.py --setup setup01 \
      --config flylight/setups/setup01/default_l40s.toml \
      --config flylight/setups/setup01/basic_8h.toml \
      --config flylight/setups/setup01/vi_th_0_3.toml \
      --config flylight/setups/setup01/eval_rm0.toml \
      --test-data ${DATA_DIR} \
      -d predict label evaluate \
      --app flylight --root ../../metrics/ppp -id ppp_basic_8h \
      --checkpoint ${CKPT} \
      --sample ${SAMPLE}"

echo "[task ${SLURM_ARRAY_TASK_ID}] finished ${SAMPLE}"
