#!/bin/bash
#SBATCH --job-name=mcfo_integrity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sbatch/mcfo/logs/%x-%j.out
#SBATCH --error=sbatch/mcfo/logs/%x-%j.err
#
# Env: RUN_DIR, RENDER_OUT, MANIFEST_OUT.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
# shellcheck source=/dev/null
source sbatch/mcfo/mcfo_env.sh
mcfo_activate

python -u -m mcfo_sim.brain.cli validate integrity \
  --rendered "${RENDER_OUT}" \
  --manifest "${MANIFEST_OUT}"

echo "integrity done -> ${RENDER_OUT}/integrity.csv"
