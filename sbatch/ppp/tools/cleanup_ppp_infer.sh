#!/bin/bash
# Remove PatchPerPix inference outputs under metrics/ppp/<exp>/{val,test}/
# (processed, instanced, evaluated) while keeping train/ checkpoints and root
# metadata (config.toml, backups/, run.log, log/).
#
# From repo root:
#   ./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp ppp_basic_fmaps24_400k --dry-run
#   ./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp ppp_basic_fmaps24_400k --execute
#   ./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp ppp_basic_8h --purge-legacy-sample-logs --dry-run
#   ./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp ppp_basic_fmaps24_400k --purge-flat-outputs --dry-run
#
# --dry-run is the default when neither --dry-run nor --execute is passed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ID=""
MODE="dry-run"
PURGE_LEGACY_LOGS=0
PURGE_FLAT_OUTPUTS=0

usage() {
  cat >&2 <<'EOF'
Usage:
  cleanup_ppp_infer.sh --exp EXP_ID [--dry-run|--execute]
      [--purge-legacy-sample-logs | --purge-flat-outputs]

Default (no purge flags): delete
  metrics/ppp/<EXP_ID>/{val,test}/{processed,instanced,evaluated}

With --purge-legacy-sample-logs (standalone; does not delete processed/):
  pre-Phase-3 per-sample log dirs that contain only run.log
  (e.g. <exp>/R22C03-…/ or <exp>/completely/<stem>/),
  not the new <exp>/log/<dataset>/<stem>/ tree.

With --purge-flat-outputs (standalone; keeps nested {dataset}/ trees):
  flat *.zarr / *.hdf / *_vote_instances*.toml under processed|instanced|
  evaluated whose parent dir is not completely|partly|synth_vh|synth_vh_u8|himanshu|1k_v1|1k_v1_u8 (covers
  pure-flat leaves and flats beside a dataset dir).

Never touches:
  metrics/ppp/<EXP_ID>/train/
  metrics/ppp/<EXP_ID>/log/
  metrics/ppp/<EXP_ID>/{config.toml,backups,run.log}

Default mode is --dry-run (list paths only).

After purging flat leftovers, re-run label+evaluate with enough mem/time, e.g.:
  ./sbatch/ppp/ppp_submit_test.sh array <preset> --stage label --split test --ckpt N --mem 96g
  ./sbatch/ppp/ppp_submit_test.sh evaluate <preset> --split test --ckpt N --mem 128g --time 4:00:00
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)
      [[ $# -ge 2 ]] || usage
      EXP_ID="$2"
      shift 2
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --execute)
      MODE="execute"
      shift
      ;;
    --purge-legacy-sample-logs)
      PURGE_LEGACY_LOGS=1
      shift
      ;;
    --purge-flat-outputs)
      PURGE_FLAT_OUTPUTS=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage
      ;;
  esac
done

if [[ -z "${EXP_ID}" ]]; then
  echo "Error: --exp EXP_ID is required" >&2
  usage
fi

if [[ "${PURGE_LEGACY_LOGS}" -eq 1 && "${PURGE_FLAT_OUTPUTS}" -eq 1 ]]; then
  echo "Error: use only one of --purge-legacy-sample-logs / --purge-flat-outputs" >&2
  exit 1
fi

EXP_ROOT="metrics/ppp/${EXP_ID}"
if [[ ! -d "${EXP_ROOT}" ]]; then
  echo "Error: experiment dir not found: ${EXP_ROOT}" >&2
  exit 1
fi

TARGETS=()

collect_flat_outputs() {
  # Purge flat *.zarr / *.hdf / *_vote_instances*.toml under stage_root when
  # the immediate parent is not a known dataset dir (completely|partly|synth_vh|synth_vh_u8|himanshu|1k_v1|1k_v1_u8).
  # Covers pure-flat param leaves and flats beside a namespaced dataset dir.
  # Prune into *.zarr so we do not walk store internals.
  local stage_root="$1"
  [[ -d "${stage_root}" ]] || return 0
  local p parent
  while IFS= read -r -d '' p; do
    parent="$(basename "$(dirname "${p}")")"
    case "${parent}" in
      completely|partly|synth_vh|synth_vh_u8|himanshu|1k_v1|1k_v1_u8) continue ;;
    esac
    TARGETS+=("${p}")
  done < <(find "${stage_root}" \
    \( -name '*.zarr' -print0 -prune \) -o \
    \( -name '*.hdf' -print0 \) -o \
    \( -name '*_vote_instances*.toml' -print0 \) \
    2>/dev/null)
}

if [[ "${PURGE_LEGACY_LOGS}" -eq 1 || "${PURGE_FLAT_OUTPUTS}" -eq 1 ]]; then
  # Standalone purge modes: do not delete whole processed/instanced/evaluated.
  :
else
  for split in val test; do
    for sub in processed instanced evaluated; do
      path="${EXP_ROOT}/${split}/${sub}"
      if [[ -e "${path}" ]]; then
        TARGETS+=("${path}")
      fi
    done
  done
fi

if [[ "${PURGE_FLAT_OUTPUTS}" -eq 1 ]]; then
  for split in val test; do
    for sub in processed instanced evaluated; do
      collect_flat_outputs "${EXP_ROOT}/${split}/${sub}"
    done
  done
  # Deduplicate while preserving order.
  if (( ${#TARGETS[@]} > 0 )); then
    mapfile -t TARGETS < <(printf '%s\n' "${TARGETS[@]}" | awk '!seen[$0]++')
  fi
fi

# Legacy Phase-1/earlier sample log dirs: dirs whose only files are run.log.
is_runlog_only_tree() {
  local d="$1"
  [[ -d "${d}" ]] || return 1
  local files
  mapfile -t files < <(find "${d}" -type f 2>/dev/null)
  (( ${#files[@]} >= 1 )) || return 1
  local f
  for f in "${files[@]}"; do
    [[ "$(basename "${f}")" == "run.log" ]] || return 1
  done
  return 0
}

if [[ "${PURGE_LEGACY_LOGS}" -eq 1 ]]; then
  # Flat legacy: <exp>/<stem>/run.log (file directly in that dir)
  for d in "${EXP_ROOT}"/*; do
    [[ -d "${d}" ]] || continue
    base="$(basename "${d}")"
    case "${base}" in
      train|val|test|backups|log) continue ;;
    esac
    if [[ -f "${d}/run.log" ]] && is_runlog_only_tree "${d}"; then
      TARGETS+=("${d}")
    fi
  done
  # Nested legacy: <exp>/<dataset>/<stem>/run.log (not under log/)
  for ds in "${EXP_ROOT}"/*; do
    [[ -d "${ds}" ]] || continue
    ds_base="$(basename "${ds}")"
    case "${ds_base}" in
      train|val|test|backups|log) continue ;;
    esac
    for sample_dir in "${ds}"/*; do
      [[ -d "${sample_dir}" ]] || continue
      if [[ -f "${sample_dir}/run.log" ]] && is_runlog_only_tree "${sample_dir}"; then
        TARGETS+=("${sample_dir}")
      fi
    done
  done
fi

if (( ${#TARGETS[@]} == 0 )); then
  echo "Nothing to delete under ${EXP_ROOT}"
  exit 0
fi

echo "mode=${MODE} exp=${EXP_ID}"
echo "keeping: ${EXP_ROOT}/train/, ${EXP_ROOT}/log/, config.toml, backups/, run.log"
echo "targets:"
for path in "${TARGETS[@]}"; do
  if [[ -d "${path}" ]]; then
    size="$(du -sh "${path}" 2>/dev/null | awk '{print $1}')"
    echo "  ${path} (${size})"
  else
    echo "  ${path}"
  fi
done

if [[ "${MODE}" == "dry-run" ]]; then
  echo "dry-run only; re-run with --execute to delete"
  exit 0
fi

for path in "${TARGETS[@]}"; do
  echo "rm -rf ${path}"
  rm -rf "${path}"
done

# Drop empty legacy dataset parent dirs (e.g. <exp>/completely after sample purge).
if [[ "${PURGE_LEGACY_LOGS}" -eq 1 ]]; then
  for ds in "${EXP_ROOT}"/*; do
    [[ -d "${ds}" ]] || continue
    ds_base="$(basename "${ds}")"
    case "${ds_base}" in
      train|val|test|backups|log) continue ;;
    esac
    if [[ -z "$(find "${ds}" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]]; then
      rmdir "${ds}" 2>/dev/null && echo "removed empty ${ds}" || true
    fi
  done
fi

echo "done. train still present: $([[ -d "${EXP_ROOT}/train" ]] && echo yes || echo NO)"
