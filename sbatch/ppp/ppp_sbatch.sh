#!/bin/bash
#SBATCH --job-name=ppp
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/ppp/ppp.out
#SBATCH --error=sbatch/ppp/ppp.err

module purge

GPU_LOGGER_PID=
cleanup() {
  if [[ -n "$GPU_LOGGER_PID" ]]; then
    kill "$GPU_LOGGER_PID" 2>/dev/null || true
    wait "$GPU_LOGGER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nohup nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 3 > gpu_usage_log_ppp.csv &
GPU_LOGGER_PID=$!

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Whole
# singularity exec --nv \
# --overlay env/ppp.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
# env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d train validate_checkpoints predict decode label evaluate --app flylight --root ppp_experiments --test-checkpoint last'

# DECODER:

# Train
# singularity exec --nv \
# --overlay env/ppp.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
# env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d train --app flylight --root ppp_experiments'

# Checkpoint
# singularity exec --nv \
# --overlay env/ppp.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
# env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d validate_checkpoints predict decode label evaluate -id ppp_experiments/flylight_setup01_260623_104736_902321 --app flylight --root ppp_experiments --test-checkpoint last'


# NO DECODER:
PPP_METRIC_ROOT="../../metrics/ppp"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0
# Train 3-4 Hours
# singularity exec --nv \
#   --overlay env/ppp.ext3:ro \
#   /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#   /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
#   env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   python3 -u run_ppp.py --setup setup01 \
#   --config flylight/setups/setup01/default_l40s.toml -d train \
#   --app flylight --root ${PPP_METRIC_ROOT}"

# Checkpoint
  # Validatioin
# singularity exec --nv \
#   --overlay env/ppp.ext3:ro \
#   /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#   /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
#   env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_l40s.toml -d validate_checkpoints -id flylight_setup01_260714_113405_392086/ --app flylight --root ${PPP_METRIC_ROOT} --test-checkpoint last"
  # Prediction
singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_l40s.toml -d predict label evaluate -id flylight_setup01_260714_113405_392086/ --app flylight --root ${PPP_METRIC_ROOT} --test-checkpoint last"

# Current
# PatchPerPix/experiments/ppp_experiments/flylight_setup01_260630_164437_336076



