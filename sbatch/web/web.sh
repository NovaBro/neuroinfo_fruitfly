#!/bin/bash
#SBATCH --job-name=web
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --mem=96g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/web/web.out
#SBATCH --error=sbatch/web/web.err

# Run BOTH the FISBe web viewer server and client on a single compute node.
#
#   - API server (FastAPI/uvicorn) runs INSIDE the Singularity webdev container
#     (its base conda env carries fastapi/uvicorn/zarr) bound to 127.0.0.1.
#   - Frontend (Vite) runs on the host using node from the user's nvm, exposed
#     on the node's network interface so a laptop SSH tunnel can reach it.
#   - Downsample thread pool size follows SLURM_CPUS_PER_TASK (this --cpus-per-task).
#
# Submit from the repo root:  sbatch sbatch/web/web.sh
set -uo pipefail

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=env/webdev.ext3
API_PORT=8000
CLIENT_PORT=5173
NODE=$(hostname -s)
STAGE_DIR="${SLURM_TMPDIR}/fisbe_web_stage"

cat <<EOF
============================================================
FISBe web viewer starting on compute node: ${NODE}
From your LAPTOP, open an SSH tunnel:

  ssh -N -L ${CLIENT_PORT}:${NODE}:${CLIENT_PORT} ${USER}@greene.hpc.nyu.edu

then browse:  http://localhost:${CLIENT_PORT}
(Only the client port is tunneled; Vite proxies /api to the server.)
============================================================
EOF

# --- API server: uvicorn inside the Singularity webdev container -------------
# 1k_v1 / 1k_v1_u8 volumes/raw and volumes/gt_instances are symlinks into
# /synth_MCFO/<name>/. Those paths exist only when the source SquashFS is an
# overlay. A missing image is a warning so completely/partly still start.
SQF_ROOT="/scratch/vh2308/data/synth_MCFO"
DATA_OVERLAY_ARGS=()
for _name in 1k_v1 1k_v1_u8; do
  _sqf="${SQF_ROOT}/${_name}.sqf"
  if [[ -f "${_sqf}" ]]; then
    DATA_OVERLAY_ARGS+=(--overlay "${_sqf}:ro")
    echo "SQF=${_sqf} mount=/synth_MCFO/${_name}"
  else
    echo "warning: not mounting missing ${_sqf} (${_name} volumes will not open)" >&2
  fi
done

singularity exec "${DATA_OVERLAY_ARGS[@]}" --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -c "
  source /ext3/env.sh
  cd web/server
  export FISBE_ROOT=../../fisbe/completely
  export PPP_EXPERIMENTS_BASE=../../metrics/ppp
  export WEB_STAGE_DIR=${STAGE_DIR}
  exec uvicorn main:app --host 127.0.0.1 --port ${API_PORT}
" &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null' EXIT

# --- Frontend: Vite dev server on the host (node comes from nvm) -------------
export NVM_DIR="/scratch/wmz2007/nvm"
# shellcheck disable=SC1090
[ -s "${NVM_DIR}/nvm.sh" ] && . "${NVM_DIR}/nvm.sh"

cd web/client || exit 1
# Serve a production build: the dev server ships hundreds of unbundled vtk.js
# modules (one round trip each over the tunnel) and double-runs effects under
# StrictMode. `vite preview` reuses server.proxy, so /api still proxies.
# Set WEB_DEV=1 for the hot-reloading dev server instead.
# --host exposes Vite on the node's interface so the login-node tunnel reaches it.
if [ "${WEB_DEV:-0}" = "1" ]; then
  npm run dev -- --host 0.0.0.0 --port "${CLIENT_PORT}"
else
  npm run build || exit 1
  npx vite preview --host 0.0.0.0 --port "${CLIENT_PORT}" --strictPort
fi
