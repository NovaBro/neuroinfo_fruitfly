#!/bin/bash
# Merge the packed PPP shards fisbe/sqf/<DATASETS>/batch_*.sqf into fewer, larger images (GROUP shards each).
# Why: a job mounts every shard as one --overlay; at ~50 overlays apptainer fails with
#   "mount /dev->/dev error: ... destination /dev doesn't exist in container" (43 still worked; 2026-11-10,
#   swc2img_1k_v1: ext3 + 4 source parts + 45 shards). The image layout is unchanged (/synth_MCFO/<ds>_ppp/...),
#   so the fisbe/<ds> symlinks stay valid.
# Each array task unsquashfs's its GROUP shards into $SLURM_TMPDIR (merged tree) and mksquashfs's it into
#   fisbe/sqf/<DATASETS>_merged/merged_XXXX.sqf. Swap the directories by hand after every task succeeded:
#   mv fisbe/sqf/<ds> fisbe/sqf/<ds>_batches && mv fisbe/sqf/<ds>_merged fisbe/sqf/<ds>
# From repo root (submits the array):
#   GROUP=3 DATASETS=swc2img_1k_v1 sbatch sbatch/ppp/prep/merge_ppp_sqf_sbatch.sh
#SBATCH --job-name=ppp_merge_sqf
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --partition=cs
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
cd /scratch/wmz2007/neuroinfo_fruitfly
DATASETS="${DATASETS:?DATASETS is required (a folder under fisbe/sqf/)}"
GROUP="${GROUP:-3}"
SRC_DIR="fisbe/sqf/${DATASETS}"
OUT_DIR="fisbe/sqf/${DATASETS}_merged"
UTILS="$(ls -d /share/apps/apptainer/*/x86_64/utils/bin 2>/dev/null | sort -V | tail -1)"
mapfile -t SHARDS < <(ls "${SRC_DIR}"/batch_*.sqf)
n_groups=$(( (${#SHARDS[@]} + GROUP - 1) / GROUP ))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "${#SHARDS[@]} shards -> ${n_groups} merged images (GROUP=${GROUP})"
  mkdir -p "${OUT_DIR}"
  sbatch --array="0-$((n_groups - 1))" --export=ALL,DATASETS="${DATASETS}",GROUP="${GROUP}" \
    sbatch/ppp/prep/merge_ppp_sqf_sbatch.sh
  exit 0
fi

t=${SLURM_ARRAY_TASK_ID}
out="${OUT_DIR}/merged_$(printf '%04d' "${t}").sqf"
if [[ -f "${out}" ]]; then echo "skip existing ${out}"; exit 0; fi
mine=("${SHARDS[@]:$((t * GROUP)):${GROUP}}")
need_kb=0
for s in "${mine[@]}"; do need_kb=$(( need_kb + $(du -k "${s}" | cut -f1) * 13 / 10 )); done  # +30% for decompression
free_kb=$(df -Pk "${SLURM_TMPDIR}" | awk 'NR==2 {print $4}')
echo "task ${t}: ${mine[*]}"
echo "need ~$((need_kb / 1048576)) GB in ${SLURM_TMPDIR}, free $((free_kb / 1048576)) GB"
if (( free_kb < need_kb )); then echo "not enough local disk" >&2; exit 1; fi

root="${SLURM_TMPDIR}/merge_root"
rm -rf "${root}"  # first unsquashfs creates it; -f merges the later shards into it
for s in "${mine[@]}"; do
  "${UTILS}/unsquashfs" -f -n -p "${SLURM_CPUS_PER_TASK:-8}" -d "${root}" "${s}"
done
# same layout as pack_ppp_sqf_sbatch.sh: image root holds synth_MCFO/<ds>_ppp/...
"${UTILS}/mksquashfs" "${root}/synth_MCFO" "${out}.tmp" -keep-as-directory -noappend -no-progress \
  -mem 16G -processors "${SLURM_CPUS_PER_TASK:-8}"
n_in=$(for s in "${mine[@]}"; do "${UTILS}/unsquashfs" -l "${s}" | grep -c '/volumes$'; done | awk '{s+=$1} END {print s}')
n_out=$("${UTILS}/unsquashfs" -l "${out}.tmp" | grep -c '/volumes$')
echo "samples in: ${n_in}, out: ${n_out}"
if [[ "${n_in}" != "${n_out}" ]]; then echo "sample count mismatch" >&2; rm -f "${out}.tmp"; exit 1; fi
mv "${out}.tmp" "${out}"
rm -rf "${root}"
echo "merged ${out}"
