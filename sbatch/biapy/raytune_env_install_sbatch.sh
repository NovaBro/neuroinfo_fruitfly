#!/bin/bash
#SBATCH --job-name=biapy-raytune-env-install
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/%x-%j.out
#SBATCH --error=sbatch/biapy/%x-%j.err
# No --gres=gpu

# One-writer install of ray[tune] + optuna into env/BiaPy_env.ext3 (conda env BiaPy_env).
#
# IMPORTANT: Do NOT mount this overlay from any other job while this runs
# (no BiaPy train/test/controller). Check: squeue -u "$USER"
#
# Submit from repo root:
#   sbatch sbatch/biapy/raytune_env_install_sbatch.sh
#
# After success, smoke the controller (dry-run):
#   sbatch sbatch/biapy/raytune_controller_sbatch.sh \
#     --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run --lr 1e-3
#
# Uses :rw --fakeroot WITHOUT --writable-tmpfs (apptainer 1.5.1).

set -euo pipefail
module purge

# Prefer submit directory (repo root). BASH_SOURCE can point at a Slurm spool copy.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

OVERLAY="${REPO_ROOT}/env/BiaPy_env.ext3"
# Match raytune_controller_sbatch.sh (Ubuntu 24.04 for host-SLURM glibc).
SIF="/share/apps/images/ubuntu-24.04.4.sif"

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
echo "Do not run BiaPy train/test/controller jobs until this finishes."
echo "============================================================"
echo "Installing ray[tune] + optuna into BiaPy_env"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "REPO_ROOT=${REPO_ROOT}"

singularity exec --fakeroot \
  --overlay "${OVERLAY}:rw" \
  "${SIF}" \
  /bin/bash -c '
    set -euo pipefail
    source /ext3/env.sh
    conda activate BiaPy_env
    echo "python=$(which python)"
    python -c "import sys; print(sys.version)"
    pip install -U "ray[tune]" optuna
    python -c "import ray, optuna; from ray.tune.search.optuna import OptunaSearch; print(\"ray\", ray.__version__, \"optuna\", optuna.__version__, \"OptunaSearch\", OptunaSearch)"
    echo "install done"
  '
