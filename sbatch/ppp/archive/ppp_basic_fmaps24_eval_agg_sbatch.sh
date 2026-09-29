#!/bin/bash
# Aggregate per-sample metrics after fmaps24 array finishes for one SPLIT.
# Prefer: bash sbatch/ppp/ppp_basic_fmaps24_full_array_chain.sh
# Or: sbatch --export=ALL,SPLIT=val --dependency=afterok:<ARRAY_ID> \
#        sbatch/ppp/ppp_basic_fmaps24_eval_agg_sbatch.sh
#SBATCH --job-name=ppp_fmaps24_agg
#SBATCH --account=torch_pr_61_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/array/fmaps24_eval_agg_%j.out
#SBATCH --error=sbatch/ppp/array/fmaps24_eval_agg_%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

SPLIT="${SPLIT:-}"
if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
  echo "Error: set SPLIT=val or SPLIT=test" >&2
  exit 1
fi

CKPT=80000
DATA_DIR="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/${SPLIT}"
echo "eval_agg split=${SPLIT} ckpt=${CKPT} job=${SLURM_JOB_ID:-local}"

# no --sample: evaluate every volume in --test-data and emit the summary table
singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_fmaps24.toml \
    --config flylight/setups/setup01/vi_th_0_3.toml \
    --config flylight/setups/setup01/eval_rm0.toml \
    --test-data ${DATA_DIR} \
    -d evaluate \
    --app flylight --root ../../metrics/ppp -id ppp_basic_fmaps24 \
    --checkpoint ${CKPT}"
