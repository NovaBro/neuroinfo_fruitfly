#!/bin/bash
#SBATCH --job-name=mcfo_install_fisbe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sbatch/mcfo/logs/%x-%j.out
#SBATCH --error=sbatch/mcfo/logs/%x-%j.err
#
# Symlink rendered zarrs into fisbe/<DATASET>/{train,val,test}/ by manifest split.
# Env: RUN_DIR, RENDER_OUT, MANIFEST_OUT, FISBE_ROOT, DATASET, SAMPLE_LIST, FORCE_INSTALL.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
# shellcheck source=/dev/null
source sbatch/mcfo/mcfo_env.sh
mcfo_activate

FORCE_ARGS=()
if [[ "${FORCE_INSTALL}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

python -u -m mcfo_sim.brain.cli install \
  --manifest "${MANIFEST_OUT}" \
  --rendered "${RENDER_OUT}" \
  --fisbe-root "${FISBE_ROOT}" \
  --dataset "${DATASET}" \
  --update-sample-list "${SAMPLE_LIST}" \
  "${FORCE_ARGS[@]}"

FISBE_LIST="${FISBE_ROOT}/sample_list_per_split.txt"
if [[ -f "${FISBE_LIST}" && "${SAMPLE_LIST}" != "${FISBE_LIST}" ]]; then
  cp -f "${SAMPLE_LIST}" "${FISBE_LIST}"
  echo "mirrored sample list -> ${FISBE_LIST}"
fi

echo "install done -> ${FISBE_ROOT}/${DATASET}/{train,val,test}/"
echo "Next: DATASETS=${DATASET} sbatch sbatch/ppp/prepare_fisbe_for_ppp_sbatch.sh"
echo "Then:  DATASETS=${DATASET} sbatch sbatch/ppp/prepare_ppp_stage_tars_sbatch.sh"
