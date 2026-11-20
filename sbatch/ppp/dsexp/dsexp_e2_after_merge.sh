#!/bin/bash
# After sbatch/ppp/prep/merge_ppp_sqf_sbatch.sh (swc2img_1k_v1, 45 shards -> 15): swap the merged images in,
# check a container with the full EXP 2 overlay set starts and sees the 788 train volumes, then resubmit
# EXP 2.1 / 2.2 (gaurav-data_n1000 / _n500; first submissions 22453746/47 failed at ~50 overlays).
# Then delete the old 45 shards (fisbe/sqf/swc2img_1k_v1_batches, ~538 GB; /scratch was at 4.09 of 5 TB) -- only after
# the per-file count of the old shards equals the merged images' and the container check passed. They can be rebuilt
# from the SWC2IMG source parts (build_swc2img_skeleton.py + pack_ppp_sqf_sbatch.sh, ~1 h).
# Submit with: sbatch --dependency=afterok:<merge array id> sbatch/ppp/dsexp/dsexp_e2_after_merge.sh
#SBATCH --job-name=dsexp_e2_after_merge
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs,cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=sbatch/ppp/logs/dsexp/dsexp_e2_after_merge-%j.out
#SBATCH --error=sbatch/ppp/logs/dsexp/dsexp_e2_after_merge-%j.out
set -euo pipefail
cd /scratch/wmz2007/neuroinfo_fruitfly
DS=swc2img_1k_v1
SQF=fisbe/sqf
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

n=$(ls "${SQF}/${DS}_merged"/merged_*.sqf 2>/dev/null | wc -l)
if [[ "${n}" -ne 15 ]] || ls "${SQF}/${DS}_merged"/*.tmp >/dev/null 2>&1; then
  echo "expected 15 merged images and no .tmp in ${SQF}/${DS}_merged (found ${n})" >&2; exit 1
fi
U="$(ls -d /share/apps/apptainer/*/x86_64/utils/bin | sort -V | tail -1)"
count_files() { for s in "$@"; do "${U}/unsquashfs" -l "${s}" | grep -c '\.zarr/'; done | awk '{t+=$1} END {print t}'; }
OLD_DIR="${SQF}/${DS}"; [[ -d "${SQF}/${DS}_batches" ]] && OLD_DIR="${SQF}/${DS}_batches"
NEW_DIR="${SQF}/${DS}_merged"; [[ -d "${NEW_DIR}" ]] || NEW_DIR="${SQF}/${DS}"
n_old=$(count_files "${OLD_DIR}"/batch_*.sqf); n_new=$(count_files "${NEW_DIR}"/merged_*.sqf)
echo "zarr entries: old shards ${n_old}, merged ${n_new}"
if [[ "${n_old}" -ne "${n_new}" || "${n_old}" -eq 0 ]]; then echo "file count mismatch" >&2; exit 1; fi
if [[ ! -d "${SQF}/${DS}_batches" ]]; then
  mv "${SQF}/${DS}" "${SQF}/${DS}_batches"
  mv "${SQF}/${DS}_merged" "${SQF}/${DS}"
fi
echo "now in ${SQF}/${DS}: $(ls ${SQF}/${DS} | wc -l) images"

OV=(--overlay env/ppp.ext3:ro)
while read -r s; do [[ -z "${s}" || "${s}" == \#* ]] || OV+=(--overlay "${s}:ro"); done < "fisbe/${DS}/source_sqf.txt"
for s in "${SQF}/${DS}"/*.sqf; do OV+=(--overlay "${s}:ro"); done
echo "overlays: ${#OV[@]} args"
n_train=$(singularity exec "${OV[@]}" "${SIF}" bash -c "ls /synth_MCFO/${DS}_ppp/train | wc -l" 2>/dev/null)
n_val=$(singularity exec "${OV[@]}" "${SIF}" bash -c "ls /synth_MCFO/${DS}_ppp/val | wc -l" 2>/dev/null)
echo "container sees train=${n_train} val=${n_val}"
if [[ "${n_train}" -ne 788 || "${n_val}" -ne 105 ]]; then echo "unexpected sample counts" >&2; exit 1; fi

for arm in "gaurav-data_n1000|" "gaurav-data_n500|--config expv2_e1_n500.toml"; do
  id=${arm%%|*}; extra=${arm#*|}
  # shellcheck disable=SC2086
  ./sbatch/ppp/ppp_submit_train.sh our-noenc-c1k_v1.1 --base original_default.toml --id "${id}" \
    --config expv2_e1_common.toml --config dsexp_pre.toml --config dsexp_swc2img.toml ${extra} \
    --config dsexp_10h.toml --account torch_pr_61_general --time 10:00:00 | tail -1
done

echo "deleting verified duplicate ${SQF}/${DS}_batches ($(du -sh "${SQF}/${DS}_batches" | cut -f1))"
rm -rf "${SQF:?}/${DS:?}_batches"
echo "done"
