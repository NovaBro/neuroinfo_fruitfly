#!/bin/bash
#SBATCH --job-name=mcfo_manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sbatch/mcfo/logs/%x-%j.out
#SBATCH --error=sbatch/mcfo/logs/%x-%j.err
#
# Env: RUN_DIR, N_IMAGES, POOL_OUT, MANIFEST_OUT, optional CALIBRATION.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
# shellcheck source=/dev/null
source sbatch/mcfo/mcfo_env.sh
mcfo_activate

CAL_ARGS=()
if [[ -n "${CALIBRATION}" ]]; then
  CAL_ARGS+=(--calibration "${CALIBRATION}")
fi

python -u -m mcfo_sim.brain.cli manifest \
  --pool "${POOL_OUT}" \
  --out "${MANIFEST_OUT}" \
  --n-images "${N_IMAGES}" \
  "${CAL_ARGS[@]}"

echo "manifest done -> ${MANIFEST_OUT}"
