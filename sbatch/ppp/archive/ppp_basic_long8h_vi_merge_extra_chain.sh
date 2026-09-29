#!/bin/bash
# Submit Layer B5–B7 unused VI knobs as independent jobs (no afterok).
# B5 evaluate-only (HDF already labeled); B6/B7 full label+evaluate.
# Does NOT re-run B1–B3. From repo root:
#     CKPT=70000 bash sbatch/ppp/ppp_basic_long8h_vi_merge_extra_chain.sh
# Default CKPT=70000 (A2 best). Override with CKPT=<n>.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT:-70000}"
# tag | overlay | DO_TASKS (+ separator; commas break SLURM --export)
JOBS=(
  "vi_merge_numinst_equal|vi_merge_numinst_equal.toml|evaluate"
  "vi_merge_do_nms|vi_merge_do_nms.toml|label+evaluate"
  "vi_merge_aff_vote|vi_merge_aff_vote.toml|label+evaluate"
)
IDS=()

for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag ov do_tasks <<< "${spec}"
  JOB_ID=$(sbatch --parsable \
    --job-name="ppp_vi_${tag}" \
    --output="sbatch/ppp/ppp_basic_long8h_${tag}_${CKPT}.out" \
    --error="sbatch/ppp/ppp_basic_long8h_${tag}_${CKPT}.err" \
    --export="ALL,VI_OVERLAY=${ov},CKPT=${CKPT},DO_TASKS=${do_tasks}" \
    sbatch/ppp/ppp_basic_long8h_vi_sweep_sbatch.sh)
  IDS+=("${JOB_ID}")
  echo "${tag}=${JOB_ID} do=${do_tasks}"
done

echo "vi_merge_extra independent jobs: ${IDS[*]} exp=metrics/ppp/ppp_basic_long8h ckpt=${CKPT}"
