#!/bin/bash
# Submit the nnU-Net fold-0 training as a chain of N 47 h jobs (each resumes with --c; a job that finds
# training + validation exports already finished exits immediately). Run from the repo root:
#   bash sbatch/nnunet/nnunet_submit_train.sh [N_JOBS=3] [AFTER_JOBID]
# AFTER_JOBID (optional): first job waits for it to succeed (afterok), e.g. the prep job.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
N="${1:-3}"
AFTER="${2:-}"
dep=()
[[ -n "${AFTER}" ]] && dep=(--dependency="afterok:${AFTER}")
prev=""
for i in $(seq 1 "${N}"); do
  jid=$(sbatch --parsable --export=ALL "${dep[@]}" sbatch/nnunet/nnunet_train_sbatch.sh)
  echo "train job ${i}/${N}: ${jid} ${dep[*]:-}"
  dep=(--dependency="afterany:${jid}")
  prev="${jid}"
done
echo "last job: ${prev}"
