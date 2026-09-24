#!/bin/bash
# CPU evaluate/aggregate for ppp_basic_long8h @ ckpt 70000 after label array.
# Protocol: vi_th_0_3 + eval_rm0 (from_scratch, remove_small_components=0).
# No GPU. Run after the corresponding label array completes.
#
# Submit from repo root:
#   sbatch --export=ALL,SPLIT=val  sbatch/ppp/ppp_basic_long8h_eval_agg_sbatch.sh
#   sbatch --export=ALL,SPLIT=test sbatch/ppp/ppp_basic_long8h_eval_agg_sbatch.sh
#SBATCH --job-name=ppp_basic_long8h_eval_agg
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --output=sbatch/ppp/array/ppp_basic_long8h_eval_agg_%j.out
#SBATCH --error=sbatch/ppp/array/ppp_basic_long8h_eval_agg_%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly

SPLIT="${SPLIT:-}"
if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
  echo "Error: set SPLIT=val or SPLIT=test (e.g. --export=ALL,SPLIT=val)" >&2
  exit 1
fi

DATA_DIR="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/${SPLIT}"
CKPT=70000
echo "eval_agg split=${SPLIT} ckpt=${CKPT} job=${SLURM_JOB_ID:-local}"

mkdir -p sbatch/ppp/array

singularity exec \
  --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
  python3 -u run_ppp.py --setup setup01 \
    --config flylight/setups/setup01/default_l40s.toml \
    --config flylight/setups/setup01/basic_long8h.toml \
    --config flylight/setups/setup01/vi_th_0_3.toml \
    --config flylight/setups/setup01/eval_rm0.toml \
    --test-data ${DATA_DIR} \
    -d evaluate \
    --app flylight --root ../../metrics/ppp -id ppp_basic_long8h \
    --checkpoint ${CKPT}"

echo "eval_agg finished split=${SPLIT}"
