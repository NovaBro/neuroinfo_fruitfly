#!/bin/bash
# Submit 8h full val+test arrays @ ckpt 8000, then aggregate each split.
# From repo root:
#     bash sbatch/ppp/ppp_basic_8h_full_array_chain.sh
#
# Discovers N volumes per split; concurrency %2. Agent does not run this for you.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p sbatch/ppp/array

submit_split() {
  local split="$1"
  local data_dir="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/${split}"
  local n
  n=$(find "${data_dir}" -maxdepth 1 -name '*.zarr' | wc -l)
  if (( n < 1 )); then
    echo "Error: no .zarr under ${data_dir}" >&2
    exit 1
  fi
  local last=$((n - 1))
  local arr_id agg_id
  arr_id=$(sbatch --parsable \
    --job-name="ppp_8h_${split}" \
    --export="ALL,SPLIT=${split}" \
    --array="0-${last}%2" \
    sbatch/ppp/ppp_basic_8h_array_sbatch.sh)
  agg_id=$(sbatch --parsable \
    --job-name="ppp_8h_agg_${split}" \
    --output="sbatch/ppp/array/8h_eval_agg_${split}_%j.out" \
    --error="sbatch/ppp/array/8h_eval_agg_${split}_%j.err" \
    --export="ALL,SPLIT=${split}" \
    --dependency="afterok:${arr_id}" \
    sbatch/ppp/ppp_basic_8h_eval_agg_sbatch.sh)
  echo "${split}: n=${n} array=0-${last}%2 array_job=${arr_id} agg=${agg_id} (afterok)"
}

submit_split val
submit_split test
echo "exp=metrics/ppp/ppp_basic_8h ckpt=8000 vi=vi_th_0_3 rm0"
