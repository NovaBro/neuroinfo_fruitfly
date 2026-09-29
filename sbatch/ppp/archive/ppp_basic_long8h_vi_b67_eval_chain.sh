#!/bin/bash
# Evaluate-only for B6/B7 (HDFs already labeled; prior run skipped evaluate due to
# SLURM --export comma bug). Independent jobs, no afterok. Does NOT re-run B5.
# From repo root:
#     CKPT=70000 bash sbatch/ppp/ppp_basic_long8h_vi_b67_eval_chain.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT:-70000}"
JOBS=(
  "vi_merge_do_nms|vi_merge_do_nms.toml|evaluate"
  "vi_merge_aff_vote|vi_merge_aff_vote.toml|evaluate"
)
IDS=()

for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag ov do_tasks <<< "${spec}"
  JOB_ID=$(sbatch --parsable \
    --job-name="ppp_vi_${tag}_eval" \
    --output="sbatch/ppp/ppp_basic_long8h_${tag}_${CKPT}.out" \
    --error="sbatch/ppp/ppp_basic_long8h_${tag}_${CKPT}.err" \
    --export="ALL,VI_OVERLAY=${ov},CKPT=${CKPT},DO_TASKS=${do_tasks}" \
    sbatch/ppp/ppp_basic_long8h_vi_sweep_sbatch.sh)
  IDS+=("${JOB_ID}")
  echo "${tag}=${JOB_ID} do=${do_tasks}"
done

echo "vi_b67_eval independent jobs: ${IDS[*]} exp=metrics/ppp/ppp_basic_long8h ckpt=${CKPT}"
