#!/bin/bash
# Checkpoint sweep for fmaps24: predict → label+evaluate (vi_th_0_3 + rm0).
# From repo root:
#     CKPTS="50000 60000 70000 80000" bash sbatch/ppp/ppp_basic_fmaps24_ckpt_sweep_chain.sh
#
# Skips predict when test/processed/<CKPT>/R22C03-*.zarr exists (80000 already from pilot).
# Override with FORCE_PREDICT=1. Subset with CKPTS="70000 80000".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CKPTS=(${CKPTS:-50000 60000 70000 80000})
SAMPLE="R22C03-20180918_66_J2"
FORCE_PREDICT="${FORCE_PREDICT:-0}"
PREV_ID=""
IDS=()

for ckpt in "${CKPTS[@]}"; do
  proc_zarr="metrics/ppp/ppp_basic_fmaps24/test/processed/${ckpt}/${SAMPLE}.zarr"
  need_predict=1
  if [[ "${FORCE_PREDICT}" != "1" && -d "${proc_zarr}" ]]; then
    need_predict=0
    echo "ckpt=${ckpt}: processed exists, skip predict"
  fi

  if [[ "${need_predict}" -eq 1 ]]; then
    deps=()
    if [[ -n "${PREV_ID}" ]]; then
      deps=(--dependency="afterok:${PREV_ID}")
    fi
    PRED_ID=$(sbatch --parsable \
      --job-name="ppp_fmaps24_pred_${ckpt}" \
      --output="sbatch/ppp/ppp_basic_fmaps24_ckpt_predict_${ckpt}.out" \
      --error="sbatch/ppp/ppp_basic_fmaps24_ckpt_predict_${ckpt}.err" \
      --export="ALL,CKPT=${ckpt}" \
      "${deps[@]}" \
      sbatch/ppp/ppp_basic_fmaps24_ckpt_predict_sbatch.sh)
    IDS+=("pred_${ckpt}=${PRED_ID}")
    PREV_ID="${PRED_ID}"
    echo "ckpt=${ckpt} predict=${PRED_ID}"
  fi

  deps=()
  if [[ -n "${PREV_ID}" ]]; then
    deps=(--dependency="afterok:${PREV_ID}")
  fi
  VI_ID=$(sbatch --parsable \
    --job-name="ppp_fmaps24_vi_${ckpt}" \
    --output="sbatch/ppp/ppp_basic_fmaps24_ckpt_vi_${ckpt}.out" \
    --error="sbatch/ppp/ppp_basic_fmaps24_ckpt_vi_${ckpt}.err" \
    --export="ALL,CKPT=${ckpt}" \
    "${deps[@]}" \
    sbatch/ppp/ppp_basic_fmaps24_ckpt_label_eval_sbatch.sh)
  IDS+=("vi_${ckpt}=${VI_ID}")
  PREV_ID="${VI_ID}"
  echo "ckpt=${ckpt} label_eval=${VI_ID}"
done

echo "ckpt_sweep afterok chain: ${IDS[*]} exp=metrics/ppp/ppp_basic_fmaps24 vi=vi_th_0_3 rm0"
