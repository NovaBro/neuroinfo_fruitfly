#!/bin/bash
#SBATCH --job-name=nnunet-env-install
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=16g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/nnunet/logs/%x-%j.out
#SBATCH --error=sbatch/nnunet/logs/%x-%j.err

# One-writer install of conda env `nnunet` (Python 3.11, torch 2.8.0+cu128, nnU-Net v2.8.1 editable)
# into env/nnunet.ext3. Creates the overlay from the NYU 25 GB template and installs Miniforge into
# /ext3 if they are missing (same template as ppp/BiaPy_env/webdev). Re-running is safe (idempotent).
#
# Do NOT mount env/nnunet.ext3 from any other job while this runs.
# apptainer 1.5.x: overlay writes need `:rw --fakeroot` WITHOUT --writable-tmpfs.
#
# Prereq (once):  git clone --depth 1 --branch v2.8.1 https://github.com/MIC-DKFZ/nnUNet.git nnUNet
# Submit from repo root:
#   sbatch sbatch/nnunet/nnunet_env_install_sbatch.sh

set -euo pipefail
module purge

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"   # never BASH_SOURCE (Slurm spool copy)
cd "${REPO_ROOT}"

SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
OVERLAY="${REPO_ROOT}/env/nnunet.ext3"
TEMPLATE="/share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz"
export NNUNET_DIR="${REPO_ROOT}/nnUNet"
export PIP_CACHE_DIR="/scratch/wmz2007/.cache/pip"   # keep pip cache off /home (30k-file quota)

[[ -f "${NNUNET_DIR}/pyproject.toml" ]] || { echo "Error: clone nnUNet v2.8.1 into ${NNUNET_DIR} first" >&2; exit 1; }
mkdir -p "${REPO_ROOT}/env" "${PIP_CACHE_DIR}"

if [[ ! -f "${OVERLAY}" ]]; then
  echo "Creating ${OVERLAY} from ${TEMPLATE}"
  gunzip -c "${TEMPLATE}" > "${OVERLAY}.partial"
  mv "${OVERLAY}.partial" "${OVERLAY}"
fi

echo "Installing into ${OVERLAY}  (SLURM_JOB_ID=${SLURM_JOB_ID:-unset}, host=$(hostname))"

singularity exec --fakeroot --overlay "${OVERLAY}:rw" "${SIF}" /bin/bash -s <<'INNER'
set -eo pipefail   # no -u: conda activate.d scripts reference unset vars

if [[ ! -x /ext3/miniforge3/bin/conda ]]; then
  echo "Installing Miniforge into /ext3/miniforge3"
  wget -q -O /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/Miniforge3.sh -b -p /ext3/miniforge3
  rm -f /tmp/Miniforge3.sh
  # Same env.sh as the other overlays in env/ (NYU apptainer-with-conda recipe).
  printf '%s\n' '#!/bin/bash' 'unset -f which' 'source /ext3/miniforge3/etc/profile.d/conda.sh' \
    'export PATH=/ext3/miniforge3/bin:$PATH' 'export PYTHONPATH=/ext3/miniforge3/bin:$PATH' > /ext3/env.sh
fi

source /ext3/env.sh
if [[ ! -d /ext3/miniforge3/envs/nnunet ]]; then
  conda create -y -p /ext3/miniforge3/envs/nnunet python=3.11
fi
conda activate /ext3/miniforge3/envs/nnunet
echo "python=$(which python) $(python --version)"

# 1) torch + torchvision as a matched pair from the cu128 index. torch 2.8.0 is the newest release
#    nnU-Net accepts (pyproject excludes 2.9.*). torchvision must be pinned with it: timm (pulled by
#    dynamic-network-architectures) depends on torchvision and would otherwise upgrade torch.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
CONS=/tmp/nnunet_constraints.txt
printf '%s\n' 'torch==2.8.0' 'torchvision==0.23.0' 'zarr<3' > "${CONS}"

# 2) nnU-Net v2.8.1, editable (custom trainers go through nnUNet_extTrainer, no fork needed).
pip install -e "${NNUNET_DIR}" -c "${CONS}"

# 3) Extras for our converter / decoder (zarr v2 to match env/evaluate.ext3).
pip install 'zarr>=2.18.3,<3' tifffile imagecodecs scikit-image scipy -c "${CONS}"

pip check || true
cd /
python - <<'PY'
import torch, torchvision, nnunetv2, zarr, tifffile, skimage, scipy
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as f
import os
print("torch", torch.__version__, "cuda", torch.version.cuda, "torchvision", torchvision.__version__)
print("nnunetv2", os.path.dirname(nnunetv2.__file__))
print("zarr", zarr.__version__, "tifffile", tifffile.__version__, "skimage", skimage.__version__)
# Resolving a variant trainer imports primus -> timm -> torchvision (crashed with a mismatched pair).
print("trainer lookup:", f("nnUNetTrainer_250epochs"), f("nnUNetTrainerBenchmark_5epochs"))
PY
which nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict
df -h /ext3 | tail -1
echo "install done"
INNER
