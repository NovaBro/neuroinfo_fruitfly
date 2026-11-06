#!/bin/bash
# From repo root: sbatch sbatch/ppp/tools/test_decoder_upsample_equiv_sbatch.sh
#SBATCH --job-name=ppp_dec_equiv
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err
set -uo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
S=flylight/setups/setup01
singularity exec --nv --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; \
    python -u ../../sbatch/ppp/tools/test_decoder_upsample_equiv.py $S/configs/base/original_default_train_code.toml $S/configs/presets/our-decoder-c1k_v1_u8.toml"
