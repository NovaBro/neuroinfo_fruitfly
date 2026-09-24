#!/bin/bash
#SBATCH --job-name=vesselfm-env-install
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/vesselfm/%x-%j.out
#SBATCH --error=sbatch/vesselfm/%x-%j.err

# One-writer install of conda env `vesselfm` into env/vesselfm.ext3.
# Do NOT mount this overlay from any other job while this runs.
# Submit from repo root:
#   sbatch sbatch/vesselfm/vesselfm_env_install_sbatch.sh

set -euo pipefail
module purge

# Prefer submit directory (repo root). BASH_SOURCE can point at a Slurm spool copy.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/env/vesselfm.ext3" ]]; then
  echo "Error: expected overlay at ${REPO_ROOT}/env/vesselfm.ext3 (submit from repo root)" >&2
  exit 1
fi

OVERLAY="${REPO_ROOT}/env/vesselfm.ext3"
VESSELFM_DIR="${REPO_ROOT}/vesselFM"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

if [[ ! -f "${OVERLAY}" ]]; then
  echo "Error: overlay not found: ${OVERLAY}" >&2
  exit 1
fi
if [[ ! -d "${VESSELFM_DIR}" ]]; then
  echo "Error: vesselFM dir not found: ${VESSELFM_DIR}" >&2
  exit 1
fi

echo "Installing into ${OVERLAY}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

singularity exec --fakeroot \
  --overlay "${OVERLAY}:rw" \
  "${SIF}" \
  /bin/bash -c "
    set -euo pipefail
    source /ext3/env.sh

    if conda env list | awk '{print \$1}' | grep -qx vesselfm; then
      echo \"conda env vesselfm already exists; reusing\"
    else
      echo \"creating conda env vesselfm (python=3.9)\"
      conda create -n vesselfm python=3.9 -y
    fi

    conda activate vesselfm
    echo \"python=\$(which python)\"
    pip install -r \"${VESSELFM_DIR}/requirements.txt\"
    pip install zarr
    cd \"${VESSELFM_DIR}\"
    # vesselFM is a PEP 420 namespace package (no __init__.py), so find_packages()
    # is empty and pip -e alone does not put it on sys.path. Drop an explicit .pth.
    pip install -e .
    SITE=\"\$(python -c 'import site; print(site.getsitepackages()[0])')\"
    echo \"${VESSELFM_DIR}\" > \"\${SITE}/vesselfm_src.pth\"
    # Prove import works from a non-source cwd
    cd /
    python -c \"import torch, monai, vesselfm; print('torch', torch.__version__); print('monai', monai.__version__); print('vesselfm', vesselfm)\"
    echo \"install done\"
  "
