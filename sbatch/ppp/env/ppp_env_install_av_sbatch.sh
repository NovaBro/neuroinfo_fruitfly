#!/bin/bash
#SBATCH --job-name=ppp-env-install-av
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/ppp/logs/%x-%j.out
#SBATCH --error=sbatch/ppp/logs/%x-%j.err
# No --gres=gpu

# One-writer install of PyAV (av) into env/ppp.ext3 (conda env ppp).
# Required by data_download_methods/prepare_himanshu_for_ppp.py (.h5j decode).
#
# IMPORTANT: Do NOT mount this overlay from any other job while this runs
# (no PPP train/predict/label). Check: squeue -u "$USER"
#
# Submit from repo root:
#   sbatch sbatch/ppp/env/ppp_env_install_av_sbatch.sh
#
# Uses :rw --fakeroot WITHOUT --writable-tmpfs (apptainer 1.5.1).

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p sbatch/ppp/logs

OVERLAY="${REPO_ROOT}/env/ppp.ext3"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

if [[ ! -f "${OVERLAY}" ]]; then
  echo "Error: expected overlay at ${OVERLAY} (submit from repo root)" >&2
  exit 1
fi
if [[ ! -f "${SIF}" ]]; then
  echo "Error: Singularity image not found: ${SIF}" >&2
  exit 1
fi

echo "============================================================"
echo "WARNING: exclusive :rw mount of ${OVERLAY}"
echo "Do not run PPP train/predict/label jobs until this finishes."
echo "============================================================"
echo "Installing av (PyAV) into ppp"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "REPO_ROOT=${REPO_ROOT}"

singularity exec --fakeroot \
  --overlay "${OVERLAY}:rw" \
  "${SIF}" \
  /bin/bash -c '
    set -eo pipefail
    source /ext3/env.sh
    conda activate ppp
    echo "python=$(which python)"
    python -c "import sys; print(sys.version)"
    pip install "av>=10"
    python -c "import av; print(\"av\", av.__version__)"
    echo "install done"
  '

echo "Next: sbatch sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh"
