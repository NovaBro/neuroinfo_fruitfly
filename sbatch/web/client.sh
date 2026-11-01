#!/bin/bash
#SBATCH --job-name=client
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=16g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/web/client.out
#SBATCH --error=sbatch/web/client.err

# Standalone Vite client (production build) for the FISBe viewer. Prefer web.sh, which runs the
# client and API server on ONE node. If you run this separately, set
# VITE_API_TARGET to the server node so the /api proxy reaches it, e.g.:
#   VITE_API_TARGET=http://cs123:8000 sbatch sbatch/web/client.sh
#
# Submit from the repo root:  sbatch sbatch/web/client.sh
set -uo pipefail

CLIENT_PORT=5173
NODE=$(hostname -s)

echo "Vite client node: ${NODE}  (port ${CLIENT_PORT})"
echo "Tunnel from your laptop:"
echo "  ssh -N -L ${CLIENT_PORT}:${NODE}:${CLIENT_PORT} ${USER}@greene.hpc.nyu.edu"

# node comes from the user's nvm install (not a module).
export NVM_DIR="/scratch/wmz2007/nvm"
# shellcheck disable=SC1090
[ -s "${NVM_DIR}/nvm.sh" ] && . "${NVM_DIR}/nvm.sh"

cd web/client || exit 1
# Serve a production build: the dev server ships hundreds of unbundled vtk.js
# modules (one round trip each over the tunnel) and double-runs effects under
# StrictMode. `vite preview` reuses server.proxy, so /api still proxies.
# Set WEB_DEV=1 for the hot-reloading dev server instead.
# --host exposes Vite on the node's interface so the login-node tunnel reaches it.
# VITE_API_TARGET (if set) redirects the /api proxy to a separate server node.
if [ "${WEB_DEV:-0}" = "1" ]; then
  npm run dev -- --host 0.0.0.0 --port "${CLIENT_PORT}"
else
  npm run build || exit 1
  npx vite preview --host 0.0.0.0 --port "${CLIENT_PORT}" --strictPort
fi
