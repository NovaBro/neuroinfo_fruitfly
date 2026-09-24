#!/bin/bash
# PatchPerPix long8h: RTX 6000 label+evaluate ALL test volumes @ ckpt 70000.
# Requires processed/70000 from a prior GPU predict. Protocol: vi_th_0_3 + eval_rm0.
# Resume after TIMEOUT: re-submit (skips existing instanced).
# Submit: sbatch sbatch/ppp/ppp_basic_long8h_full_test_rtx6000_sbatch.sh
#SBATCH --job-name=ppp_basic_long8h_full_test_rtx6000
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --partition=rtx6000
#SBATCH --output=sbatch/ppp/ppp_basic_long8h_full_test_rtx6000.out
#SBATCH --error=sbatch/ppp/ppp_basic_long8h_full_test_rtx6000.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

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
  > gpu_usage_log_ppp_basic_long8h_full_test_rtx6000.csv &
GPU_LOGGER_PID=$!

# All 20 volumes under fisbe/completely/test; --checkpoint 70000 (not last/80000).
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
    --test-data /scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/test \
    -d label evaluate \
    --app flylight --root ../../metrics/ppp -id ppp_basic_long8h \
    --checkpoint 70000"
