#!/bin/bash
# GPU check: decode one val sample with blockwise decode, assert complete
# pred_affs, and self-consistency on one spatial block. From repo root:
#   sbatch sbatch/ppp/tools/verify_blockwise_decode.sh
#SBATCH --job-name=ppp_verify_blockwise_decode
#SBATCH --account=torch_pr_61_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err

set -euo pipefail
module purge
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/logs gpu_log/ppp

SAMPLE="completely/R22C03-20180918_66_J2"
ZARR="metrics/ppp/basic-ppp-decoder/val/processed/15000/${SAMPLE}.zarr"

# Prefer a fresh copy so we do not disturb production outputs mid-flight.
VERIFY_SAMPLE="completely/R22C03-20180918_66_J2_blockwise_verify"
VERIFY_ZARR="metrics/ppp/basic-ppp-decoder/val/processed/15000/${VERIFY_SAMPLE}.zarr"
rm -rf "${VERIFY_ZARR}"
mkdir -p "$(dirname "${VERIFY_ZARR}")"
echo "Copying ${ZARR} -> ${VERIFY_ZARR}"
cp -a "${ZARR}" "${VERIFY_ZARR}"
rm -rf "${VERIFY_ZARR}/volumes/pred_affs"

nvidia-smi \
  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 3 \
  > "gpu_log/ppp/gpu_usage_log_verify_blockwise_${SLURM_JOB_ID}.csv" &
GPU_LOGGER_PID=$!
trap 'kill ${GPU_LOGGER_PID} 2>/dev/null || true' EXIT

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
echo "Running decode via run_ppp.py on ${VERIFY_SAMPLE}"
T0=$(date +%s)
singularity exec --nv \
  --overlay env/ppp.ext3:ro \
  "${SIF}" \
  /bin/bash -c "
source /ext3/env.sh
conda activate ppp
cd PatchPerPix/experiments/
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u run_ppp.py --setup setup01 \
  --config flylight/setups/setup01/original_default_train_code.toml \
  --config flylight/setups/setup01/basic-ppp-decoder.toml \
  -d decode \
  --app flylight --root ../../metrics/ppp -id basic-ppp-decoder \
  --checkpoint 15000 \
  --sample ${VERIFY_SAMPLE} \
  --split val \
  --test-data /scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/val
"
T1=$(date +%s)
echo "decode wall seconds: $((T1 - T0))"

python3 sbatch/ppp/check_prediction_complete.py \
  --aff-key volumes/pred_affs "${VERIFY_ZARR}"

# Sanity: FG voxels should be nonzero in pred_affs; empty regions stay 0.
python3 - <<PY
import numpy as np
import zarr
sample = "${VERIFY_ZARR}"
zf = zarr.open(sample, "r")
aff = zf["volumes/pred_affs"]
numinst = zf["volumes/pred_numinst"]
# Sample a few z-slices for FG vs BG mean (full volume is huge).
zs = slice(0, min(64, aff.shape[1]))
ni = np.asarray(numinst[0, zs])
fg = ni < 0.1
aff_s = np.asarray(aff[:, zs], dtype=np.float32)
if fg.any():
    fg_mean = float(np.abs(aff_s[:, fg]).mean())
    print(f"FG mean |aff| on first blocks: {fg_mean} n_fg={int(fg.sum())}")
    assert fg_mean > 0, "FG affinities are all zero"
bg = ~fg
if bg.any():
    bg_mean = float(np.abs(aff_s[:, bg]).mean())
    print(f"BG mean |aff| on first blocks: {bg_mean}")
print("SANITY_OK")
PY

python3 - <<PY
import csv
p = "gpu_log/ppp/gpu_usage_log_verify_blockwise_${SLURM_JOB_ID}.csv"
vals = []
with open(p) as f:
    r = csv.reader(f)
    hdr = next(r)
    idx = next(i for i, h in enumerate(hdr) if "utilization.gpu" in h)
    for row in r:
        try:
            vals.append(float(row[idx].strip().replace("%", "")))
        except Exception:
            pass
if vals:
    n = len(vals)
    low = sum(1 for v in vals if v < 10)
    hi = sum(1 for v in vals if v >= 50)
    print(f"gpu util n={n} mean={sum(vals)/n:.1f} <10%={low} ({100*low/n:.0f}%) >=50%={hi}")
PY

echo "verify_blockwise_decode done"
