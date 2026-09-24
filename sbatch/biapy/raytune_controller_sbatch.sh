#!/bin/bash
# CPU-only Ray Tune controller: runs run_tune.py, which submits nested GPU trains.
#
# Budget: controller_time ≳ (num_samples / max_concurrent) * train_time_limit + slack
#
# Requires Ray in env/BiaPy_env.ext3 (Stage 5). Nested sbatch needs host SLURM
# binaries at /opt/slurm, SLURM_CONF, /run/munge, host passwd/group, and libmunge
# (Greene / Singularity).
#
# Usage (from repo root):
#   sbatch sbatch/biapy/raytune_controller_sbatch.sh --exp NAME --num-samples N [run_tune flags...]
#
# Dry-run example (no nested GPU):
#   sbatch sbatch/biapy/raytune_controller_sbatch.sh \
#     --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run \
#     --lr 1e-3 --batch-size 30 --skel-weight 0.1
#
# Short nested smoke:
#   sbatch sbatch/biapy/raytune_controller_sbatch.sh \
#     --exp biapy_fdb_skel_smoke --num-samples 1 \
#     --search-epochs 2 --search-patience 2 \
#     --lr 1e-3 --batch-size 30
#
# Override resources: sbatch --time=1:00:00 --mem=8g sbatch/biapy/raytune_controller_sbatch.sh ...

#SBATCH --account=torch_pr_61_general
#SBATCH --job-name=biapy-raytune-controller
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=7-00:00:00
#SBATCH --output=sbatch/biapy/%x-%j.out
#SBATCH --error=sbatch/biapy/%x-%j.err
# No --gres=gpu — controller must not sit idle on a GPU partition.

set -euo pipefail
module purge

# Prefer submit directory (repo root). BASH_SOURCE can point at a Slurm spool copy.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 --exp NAME --num-samples N [run_tune.py flags...]" >&2
  echo "Example dry-run:" >&2
  echo "  sbatch $0 --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run --lr 1e-3" >&2
  exit 1
fi

OVERLAY="${REPO_ROOT}/env/BiaPy_env.ext3"
# Ubuntu 24.04 (glibc ≥ 2.38): host /opt/slurm/bin/sbatch needs this after Greene
# RHEL 10. GPU trains keep cuda12.1.1-…-ubuntu22.04 via biapy-py_sbatch.sh.
SIF="/share/apps/images/ubuntu-24.04.4.sif"
SLURM_ROOT="/opt/slurm"

if [[ ! -f "$OVERLAY" ]]; then
  echo "Error: overlay not found: ${OVERLAY}" >&2
  exit 1
fi
if [[ ! -f "$SIF" ]]; then
  echo "Error: Singularity image not found: ${SIF}" >&2
  exit 1
fi
if [[ ! -d "$SLURM_ROOT" ]]; then
  echo "Error: host SLURM root not found: ${SLURM_ROOT} (needed for nested sbatch)" >&2
  exit 1
fi

SLURM_CONF_HOST="${SLURM_CONF:-/opt/slurm/data/slurmd/conf-cache/slurm.conf}"
MUNGE_RUN="/run/munge"
# Host munge shared lib (auth_munge.so depends on it; not in the Ubuntu SIF).
MUNGE_LIB="$(readlink -f /usr/lib64/libmunge.so.2 2>/dev/null || true)"
if [[ -z "$MUNGE_LIB" || ! -f "$MUNGE_LIB" ]]; then
  MUNGE_LIB="$(readlink -f /lib64/libmunge.so.2 2>/dev/null || true)"
fi
if [[ ! -f "$SLURM_CONF_HOST" ]]; then
  echo "Error: SLURM conf not found: ${SLURM_CONF_HOST}" >&2
  exit 1
fi
if [[ ! -d "$MUNGE_RUN" ]]; then
  echo "Error: munge run dir not found: ${MUNGE_RUN}" >&2
  exit 1
fi
if [[ ! -f /etc/passwd || ! -f /etc/group ]]; then
  echo "Error: host /etc/passwd or /etc/group missing (needed for SlurmUser)" >&2
  exit 1
fi
if [[ -z "$MUNGE_LIB" || ! -f "$MUNGE_LIB" ]]; then
  echo "Error: host libmunge.so.2 not found under /usr/lib64 or /lib64" >&2
  exit 1
fi

export RAY_TMPDIR="/tmp/${USER}_ray_${SLURM_JOB_ID:-manual}"
mkdir -p "$RAY_TMPDIR"
# Optional: export RAY_DEDUP_LOGS=0

echo "Controller REPO_ROOT=${REPO_ROOT}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"
echo "SLURM_CONF=${SLURM_CONF_HOST}"
echo "MUNGE_LIB=${MUNGE_LIB}"
echo "run_tune.py args: $*"

# Queue hygiene: refuse if too many nested train jobs already queued (skip dry-run).
MAX_USER_TRAINS="${MAX_USER_TRAINS:-4}"
is_dry_run=0
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then
    is_dry_run=1
    break
  fi
done
if [[ "$is_dry_run" -eq 0 && "$MAX_USER_TRAINS" -ge 0 ]]; then
  train_count=0
  while IFS= read -r jname; do
    if [[ "$jname" == *-train ]]; then
      train_count=$((train_count + 1))
    fi
  done < <(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
  if [[ "$train_count" -ge "$MAX_USER_TRAINS" ]]; then
    echo "Error: ${train_count} user jobs matching '*-train' already in squeue" >&2
    echo "  (MAX_USER_TRAINS=${MAX_USER_TRAINS}). Wait or export MAX_USER_TRAINS=-1 to disable." >&2
    exit 1
  fi
  echo "Queue check OK: ${train_count} *-train jobs (limit ${MAX_USER_TRAINS})"
fi

# Forward "$@" into the container bash via the fake $0 trick so flags survive.
singularity exec \
  --bind "${SLURM_ROOT}:${SLURM_ROOT}" \
  --bind "${MUNGE_RUN}:${MUNGE_RUN}" \
  --bind /etc/passwd:/etc/passwd:ro \
  --bind /etc/group:/etc/group:ro \
  --bind "${MUNGE_LIB}:/usr/lib64/libmunge.so.2:ro" \
  --bind "${MUNGE_LIB}:/usr/lib64/libmunge.so.2.0.0:ro" \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -c '
    set -euo pipefail
    export PATH="'"${SLURM_ROOT}"'/bin:${PATH}"
    export SLURM_CONF="'"${SLURM_CONF_HOST}"'"
    export LD_LIBRARY_PATH="/usr/lib64:'"${SLURM_ROOT}"'/lib64:'"${SLURM_ROOT}"'/lib64/slurm:${LD_LIBRARY_PATH:-}"
    export RAY_TMPDIR="'"${RAY_TMPDIR}"'"
    cd "'"${REPO_ROOT}"'"
    source /ext3/env.sh
    conda activate BiaPy_env
    if ! sbatch --version >/dev/null 2>&1; then
      echo "Error: sbatch cannot read SLURM config inside container" >&2
      echo "  SLURM_CONF=${SLURM_CONF}" >&2
      sbatch --version >&2 || true
      exit 1
    fi
    if ! python -c "import ray" 2>/dev/null; then
      echo "Error: Ray is not installed in BiaPy_env. Complete Stage 5 first:" >&2
      echo "  pip install \"ray[tune]\"  (into env/BiaPy_env.ext3 via :rw --fakeroot)" >&2
      exit 1
    fi
    echo "Ray OK; sbatch=$(command -v sbatch); SLURM_CONF=${SLURM_CONF}"
    exec python -u biapy_work_folder/raytune/run_tune.py "$@"
  ' bash "$@"
