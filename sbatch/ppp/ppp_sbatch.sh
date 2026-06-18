#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH --job-name=ppp
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/ppp/ppp.out
#SBATCH --error=sbatch/ppp/ppp.err

module purge

# Train
singularity exec --nv \
--overlay env/ppp.ext3:ro \
/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d train validate_checkpoints predict decode label evaluate --app flylight --root ppp_experiments --test-checkpoint last'

# Checkpoint
# singularity exec --nv \
# --overlay env/ppp.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/;
# env CUDA_VISIBLE_DEVICES=0 ; python3 -u run_ppp.py --setup setup01 --config flylight/setups/setup01/default_train_code_l40s.toml -d validate_checkpoints predict decode label evaluate -id ppp_experiments/flylight_setup01_260617_130205_919927 --app flylight --root ppp_experiments --test-checkpoint last'

# cd PatchPerPix/experiments/
# nohup env CUDA_VISIBLE_DEVICES=0 \
# python -u run_ppp.py \
# --setup setup01 \
# --config flylight/setups/setup01/default_train_code.toml \
# -id ppp_experiments/flylight_setup01_260610_113923_103163 \
# --run_from_exp \
# -d validate_checkpoints predict decode label evaluate \
# --app flylight \
# --checkpoint 4000 \
# > evaluate.txt 2>&1 &