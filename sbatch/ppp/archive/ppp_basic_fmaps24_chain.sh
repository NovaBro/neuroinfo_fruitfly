#!/bin/bash
# Submit PatchPerPix fmaps24 train then infer (afterok). From repo root:
#     bash sbatch/ppp/ppp_basic_fmaps24_chain.sh
# Prefer train-only first if wall time is uncertain:
#     sbatch sbatch/ppp/ppp_basic_fmaps24_train_sbatch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ID=$(sbatch --parsable sbatch/ppp/ppp_basic_fmaps24_train_sbatch.sh)
INFER_ID=$(sbatch --parsable --dependency="afterok:${TRAIN_ID}" \
  sbatch/ppp/ppp_basic_fmaps24_infer_sbatch.sh)
echo "train=${TRAIN_ID} infer=${INFER_ID} (afterok) exp=metrics/ppp/ppp_basic_fmaps24"
