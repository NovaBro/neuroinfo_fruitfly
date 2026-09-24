#!/bin/bash
# Submit Layer B VI merge experiments (label+evaluate) as afterok chain.
# From repo root:
#     CKPT=70000 bash sbatch/ppp/ppp_basic_long8h_vi_merge_chain.sh
# Default CKPT=70000 (A2 best). Override with CKPT=<n>.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CKPT="${CKPT:-70000}"
OVERLAYS=(vi_merge_mws_false.toml vi_merge_no_single_ccs.toml vi_merge_skel_false.toml)
PREV_ID=""
IDS=()

for ov in "${OVERLAYS[@]}"; do
  tag="${ov%.toml}"
  deps=()
  if [[ -n "${PREV_ID}" ]]; then
    deps=(--dependency="afterok:${PREV_ID}")
  fi
  JOB_ID=$(sbatch --parsable \
    --job-name="ppp_vi_${tag}" \
    --output="sbatch/ppp/ppp_basic_long8h_${tag}_${CKPT}.out" \
    --error="sbatch/ppp/ppp_basic_long8h_${tag}_${CKPT}.err" \
    --export="ALL,VI_OVERLAY=${ov},CKPT=${CKPT}" \
    "${deps[@]}" \
    sbatch/ppp/ppp_basic_long8h_vi_sweep_sbatch.sh)
  IDS+=("${JOB_ID}")
  echo "${tag}=${JOB_ID}"
  PREV_ID="${JOB_ID}"
done

echo "vi_merge afterok chain: ${IDS[*]} exp=metrics/ppp/ppp_basic_long8h ckpt=${CKPT}"
