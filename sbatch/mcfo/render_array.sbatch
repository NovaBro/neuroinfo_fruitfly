#!/bin/bash
#SBATCH --job-name=mcfo_render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sbatch/mcfo/logs/%x-%A_%a.out
#SBATCH --error=sbatch/mcfo/logs/%x-%A_%a.err
#
# Array task k of N_SHARDS: --shard k/N_SHARDS
# Env: RUN_DIR, N_SHARDS, CACHE, POOL_OUT, MANIFEST_OUT, RENDER_OUT.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
# shellcheck source=/dev/null
source sbatch/mcfo/mcfo_env.sh
mcfo_activate

mkdir -p "${RENDER_OUT}"
K="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required}"
N="${N_SHARDS}"

echo "render shard ${K}/${N} -> ${RENDER_OUT}"
python -u -m mcfo_sim.brain.cli render \
  --manifest "${MANIFEST_OUT}" \
  --pool "${POOL_OUT}" \
  --cache "${CACHE}" \
  --out "${RENDER_OUT}" \
  --shard "${K}/${N}"

echo "render shard ${K}/${N} done"
