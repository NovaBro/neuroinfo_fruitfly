#!/bin/bash
#SBATCH --job-name=web
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=64g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/web/web.out
#SBATCH --error=sbatch/web/web.err

# Run BOTH the FISBe web viewer server and client on a single compute node.
#
#   - API server (FastAPI/uvicorn) runs INSIDE the Singularity webdev container
#     (its base conda env carries fastapi/uvicorn/zarr) bound to 127.0.0.1.
#   - Frontend (Vite) runs on the host using node from the user's nvm, exposed
#     on the node's network interface so a laptop SSH tunnel can reach it.
#
# Submit from the repo root:  sbatch sbatch/web/web.sh
set -uo pipefail

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=env/webdev.ext3
API_PORT=8000
CLIENT_PORT=5173
NODE=$(hostname -s)

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
singularity exec --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -c "
  source /ext3/env.sh
  cd web/server
  export FISBE_ROOT=../../fisbe/completely
  exec uvicorn main:app --host 127.0.0.1 --port ${API_PORT}
" &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null' EXIT

# --- Frontend: Vite dev server on the host (node comes from nvm) -------------
export NVM_DIR="${HOME}/.nvm"
# shellcheck disable=SC1090
[ -s "${NVM_DIR}/nvm.sh" ] && . "${NVM_DIR}/nvm.sh"

cd web/client || exit 1
# --host exposes Vite on the node's interface so the login-node tunnel reaches it.
npm run dev -- --host 0.0.0.0 --port "${CLIENT_PORT}"
