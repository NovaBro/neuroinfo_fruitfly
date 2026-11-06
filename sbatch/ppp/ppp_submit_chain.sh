#!/bin/bash
# Chain full inference pipelines (ppp_submit_test.sh array --stage all) over a
# list of checkpoints, one after another. Each pipeline's predict array waits for
# the previous pipeline's evaluate job (which strips pred_affs), so at most one
# pipeline's pred_affs per chain are on disk at a time. Run two chains in
# parallel to get "<= 2 val/test pipelines at once".
#
# From repo root:
#   ./sbatch/ppp/ppp_submit_chain.sh [--dry-run] [--after JOBID|SPEC] \
#     <preset> <exp_id> <split> <ckpt,ckpt,...> [extra ppp_submit_test.sh flags…]
#
#   ./sbatch/ppp/ppp_submit_chain.sh our-noenc-c1k_v1 our-noenc-c1k_v1 val 50000,40000 \
#     --base original_default.toml --pct 3 --account torch_pr_61_tandon_advanced
#
# --after: dependency for the first pipeline (bare job id → afterany:ID).
# Prints one "ckpt=<N> evaluate=<jobid>" line per pipeline and a final
# "chain_last=<jobid>" line (the last evaluate job), to chain further work on.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY=()
DEP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=(--dry-run); shift ;;
    --after)
      if [[ "$2" == *:* ]]; then DEP="$2"; else DEP="afterany:$2"; fi
      shift 2 ;;
    *) break ;;
  esac
done

if (( $# < 4 )); then
  sed -n '2,20p' "$0" >&2
  exit 1
fi
PRESET="$1"; EXP_ID="$2"; SPLIT="$3"; CKPTS="$4"; shift 4
EXTRA=("$@")

IFS=',' read -r -a ckpt_list <<< "${CKPTS}"
for ckpt in "${ckpt_list[@]}"; do
  dep_args=()
  [[ -n "${DEP}" ]] && dep_args=(--dependency "${DEP}")
  out="$("${SCRIPT_DIR}/ppp_submit_test.sh" "${DRY[@]+"${DRY[@]}"}" array "${PRESET}" \
    --id "${EXP_ID}" --split "${SPLIT}" --ckpt "${ckpt}" \
    "${dep_args[@]+"${dep_args[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}")"
  echo "${out}" | sed "s/^/  [${EXP_ID} ${SPLIT} ${ckpt}] /" >&2
  eval_id="$(echo "${out}" | sed -n 's/^evaluate=\([^ ]*\).*/\1/p' | tail -1)"
  if [[ -z "${eval_id}" ]]; then
    echo "Error: no evaluate job id for ckpt ${ckpt}" >&2
    exit 1
  fi
  echo "ckpt=${ckpt} evaluate=${eval_id}"
  DEP="afterany:${eval_id}"
  [[ ${#DRY[@]} -gt 0 ]] && DEP="afterany:DRYRUN_${ckpt}"
done
echo "chain_last=${eval_id}"
