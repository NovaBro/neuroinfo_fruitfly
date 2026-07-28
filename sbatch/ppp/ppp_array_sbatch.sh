#!/bin/bash
# One SLURM task per volume of data.test_data. Submit from the repo root:
#     sbatch sbatch/ppp/ppp_array_sbatch.sh
# then, once every task has finished, aggregate the per-sample metrics with:
#     sbatch sbatch/ppp/ppp_eval_aggregate_sbatch.sh
#
# Sized for data.test_data = fisbe/completely/train (18 volumes). If you point
# test_data somewhere else, update --array to <n_volumes - 1>.
#SBATCH --job-name=ppp_arr
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_61_general
#SBATCH --array=0-17%4
#SBATCH --output=sbatch/ppp/array/ppp_%A_%a.out
#SBATCH --error=sbatch/ppp/array/ppp_%A_%a.err

set -euo pipefail
module purge

EXP_ID=ppp_experiments/flylight_setup01_260630_164437_336076
CONFIG=flylight/setups/setup01/default_l40s.toml
CKPT=8000
DATA_DIR=/scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/train

PROCESSED=PatchPerPix/experiments/${EXP_ID}/test/processed/${CKPT}

# run_ppp.py filters samples with a substring match (--sample), and every FISBe
# train volume name is mutually non-substring, so a full name selects exactly one.
mapfile -t SAMPLES < <(find "${DATA_DIR}" -maxdepth 1 -name '*.zarr' -printf '%f\n' \
                       | sed 's/\.zarr$//' | sort)
if (( SLURM_ARRAY_TASK_ID >= ${#SAMPLES[@]} )); then
    echo "task ${SLURM_ARRAY_TASK_ID} >= ${#SAMPLES[@]} samples; nothing to do"
    exit 0
fi
SAMPLE=${SAMPLES[${SLURM_ARRAY_TASK_ID}]}
SAMPLE_ZARR=${PROCESSED}/${SAMPLE}.zarr
echo "[task ${SLURM_ARRAY_TASK_ID}] sample=${SAMPLE}"

# NYU public GPU partitions cancel persistently low-utilization jobs; keep the audit trail.
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
    > "sbatch/ppp/array/gpu_usage_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
GPU_LOGGER_PID=$!

# Two things happen inside the container:
#  1. Drop a prediction zarr that exists but is only partially written --
#     run_ppp would otherwise log "Already exists!", skip it, and silently
#     evaluate zeros. A *complete* zarr is kept even if an older run wrote it
#     with a different tile size: the UNet is valid-padded, so the result is
#     tiling-invariant and re-predicting it would only burn GPU hours.
#  2. Run predict+label+evaluate for this one sample.
# No validate_checkpoints: validation.checkpoints is a single entry and there is
# one param combination, so it selects nothing and costs a full pass over val.
singularity exec --nv \
    --overlay env/ppp.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "set -euo pipefail; source /ext3/env.sh; conda activate ppp; \
        if [ -d '${SAMPLE_ZARR}' ] && \
           ! python3 sbatch/ppp/check_prediction_complete.py '${SAMPLE_ZARR}'; then \
            echo 'discarding partial prediction: ${SAMPLE_ZARR}'; \
            rm -rf '${SAMPLE_ZARR}'; \
        fi; \
        cd PatchPerPix/experiments/; \
        python3 -u run_ppp.py --setup setup01 --config ${CONFIG} \
            -d predict label evaluate -id ${EXP_ID} \
            --app flylight --root ppp_experiments --test-checkpoint last \
            --sample ${SAMPLE}"

echo "[task ${SLURM_ARRAY_TASK_ID}] finished ${SAMPLE}"
