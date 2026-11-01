#!/bin/bash
# Chain MCFO brain synth SLURM jobs: pool → manifest → render array → integrity
# [→ install]. From repo root:
#
#   ./sbatch/mcfo/mcfo_submit.sh
#   ./sbatch/mcfo/mcfo_submit.sh --dry-run
#   RUN_DIR=lm2lineage/runs/brain_5k N_IMAGES=5000 N_SHARDS=32 MAX_NEURONS=0 \
#     ./sbatch/mcfo/mcfo_submit.sh
#   ./sbatch/mcfo/mcfo_submit.sh install          # install only (after synth)
#   AFTER_JOB=12345 ./sbatch/mcfo/mcfo_submit.sh install
#
# Env vars: see sbatch/mcfo/mcfo_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
MODE="chain"   # chain | install

usage() {
  cat >&2 <<'EOF'
Usage:
  mcfo_submit.sh [--dry-run] [chain|install]

  chain    (default) pool → manifest → render array → integrity
  install  symlink rendered zarrs into fisbe/<DATASET>/{train,val,test}/
           Optional AFTER_JOB=<id> for --dependency=afterok

Env overrides: RUN_DIR, N_IMAGES, N_SHARDS, MAX_NEURONS, WORKERS, CALIBRATION,
  PARTITION, RENDER_PARTITION, RENDER_PCT, ACCOUNT, FISBE_ROOT, DATASET, …
  See sbatch/mcfo/mcfo_env.sh
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    chain|install) MODE="$1"; shift ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/mcfo_env.sh"
mkdir -p "${REPO_ROOT}/sbatch/mcfo/logs" "${RUN_DIR}"

# Export so sbatch --export=ALL passes them into jobs
export REPO_ROOT RUN_DIR CACHE POOL_OUT MANIFEST_OUT RENDER_OUT
export MAX_NEURONS N_IMAGES N_SHARDS WORKERS CALIBRATION
export FISBE_ROOT DATASET SAMPLE_LIST FORCE_INSTALL
export ACCOUNT PARTITION RENDER_PARTITION RENDER_PCT

submit() {
  # Usage: submit <extra sbatch flags...> <script>
  # Prints only the job id on stdout (for capture); logs go to stderr.
  local script="${*: -1}"
  local flags=("${@:1:$#-1}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] sbatch ${flags[*]} --export=ALL ${script}" >&2
    echo "DRYRUN"
    return 0
  fi
  sbatch --parsable "${flags[@]}" --export=ALL "${script}"
}

echo "mcfo_submit mode=${MODE} RUN_DIR=${RUN_DIR} N_IMAGES=${N_IMAGES} N_SHARDS=${N_SHARDS} MAX_NEURONS=${MAX_NEURONS}"

if [[ "${MODE}" == "install" ]]; then
  DEP_FLAGS=()
  if [[ -n "${AFTER_JOB:-}" ]]; then
    DEP_FLAGS+=(--dependency="afterok:${AFTER_JOB}")
  fi
  J=$(submit \
    --account="${ACCOUNT}" --partition="${PARTITION}" \
    --cpus-per-task="${INSTALL_CPUS}" --mem="${INSTALL_MEM}" --time="${INSTALL_TIME}" \
    "${DEP_FLAGS[@]}" \
    "${SCRIPT_DIR}/install_fisbe.sbatch")
  echo "submitted install job=${J}"
  exit 0
fi

# ---- chain ----
J_POOL=$(submit \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --cpus-per-task="${POOL_CPUS}" --mem="${POOL_MEM}" --time="${POOL_TIME}" \
  "${SCRIPT_DIR}/pool.sbatch")
echo "submitted pool job=${J_POOL}"

J_MAN=$(submit \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --cpus-per-task="${MANIFEST_CPUS}" --mem="${MANIFEST_MEM}" --time="${MANIFEST_TIME}" \
  --dependency="afterok:${J_POOL}" \
  "${SCRIPT_DIR}/manifest.sbatch")
echo "submitted manifest job=${J_MAN}"

LAST_SHARD=$((N_SHARDS - 1))
J_REN=$(submit \
  --account="${ACCOUNT}" --partition="${RENDER_PARTITION}" \
  --cpus-per-task="${RENDER_CPUS}" --mem="${RENDER_MEM}" --time="${RENDER_TIME}" \
  --array="0-${LAST_SHARD}%${RENDER_PCT}" \
  --dependency="afterok:${J_MAN}" \
  "${SCRIPT_DIR}/render_array.sbatch")
echo "submitted render array job=${J_REN} (0-${LAST_SHARD}%${RENDER_PCT})"

J_INT=$(submit \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --cpus-per-task="${INTEGRITY_CPUS}" --mem="${INTEGRITY_MEM}" --time="${INTEGRITY_TIME}" \
  --dependency="afterok:${J_REN}" \
  "${SCRIPT_DIR}/integrity.sbatch")
echo "submitted integrity job=${J_INT}"

if [[ "${WITH_INSTALL:-0}" == "1" ]]; then
  J_INS=$(submit \
    --account="${ACCOUNT}" --partition="${PARTITION}" \
    --cpus-per-task="${INSTALL_CPUS}" --mem="${INSTALL_MEM}" --time="${INSTALL_TIME}" \
    --dependency="afterok:${J_INT}" \
    "${SCRIPT_DIR}/install_fisbe.sbatch")
  echo "submitted install job=${J_INS}"
fi

echo "chain done. After integrity: ./sbatch/mcfo/mcfo_submit.sh install"
echo "  (or WITH_INSTALL=1 ./sbatch/mcfo/mcfo_submit.sh to include install in the chain)"
