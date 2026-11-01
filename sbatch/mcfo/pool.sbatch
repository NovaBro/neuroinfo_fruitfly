#!/bin/bash
#SBATCH --job-name=mcfo_pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sbatch/mcfo/logs/%x-%j.out
#SBATCH --error=sbatch/mcfo/logs/%x-%j.err
#
# Env: RUN_DIR, MAX_NEURONS, WORKERS, CACHE, POOL_OUT (see mcfo_env.sh).
# Submit via mcfo_submit.sh or:
#   sbatch --export=ALL,RUN_DIR=... sbatch/mcfo/pool.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
# shellcheck source=/dev/null
source sbatch/mcfo/mcfo_env.sh
mcfo_activate

python -u -m mcfo_sim.brain.cli pool \
  --cache "${CACHE}" \
  --out "${POOL_OUT}" \
  --max-neurons "${MAX_NEURONS}" \
  --workers "${WORKERS}"

echo "pool done -> ${POOL_OUT}"
