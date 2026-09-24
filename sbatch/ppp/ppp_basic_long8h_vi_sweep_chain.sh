#!/bin/bash
# Submit VI threshold sweep (label+evaluate) as afterok chain. From repo root:
#     bash sbatch/ppp/ppp_basic_long8h_vi_sweep_chain.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OVERLAYS=(vi_th_0_3.toml vi_th_0_4.toml vi_th_0_5.toml vi_th_0_6.toml vi_th_0_7.toml)
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
    --output="sbatch/ppp/ppp_basic_long8h_${tag}.out" \
    --error="sbatch/ppp/ppp_basic_long8h_${tag}.err" \
    --export="ALL,VI_OVERLAY=${ov}" \
    "${deps[@]}" \
    sbatch/ppp/ppp_basic_long8h_vi_sweep_sbatch.sh)
  IDS+=("${JOB_ID}")
  echo "${tag}=${JOB_ID}"
  PREV_ID="${JOB_ID}"
done

echo "vi_sweep afterok chain: ${IDS[*]} exp=metrics/ppp/ppp_basic_long8h ckpt=30000"
