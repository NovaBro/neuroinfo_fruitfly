#!/bin/bash
#SBATCH --job-name=server
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/web/server.out
#SBATCH --error=sbatch/web/server.err

# Standalone API server for the FISBe viewer, run inside the Singularity webdev
# container (its base conda env carries fastapi/uvicorn/zarr). Prefer web.sh,
# which co-locates the server and client on ONE node so Vite's /api proxy works.
# If you run this separately, point the client at this node with VITE_API_TARGET
# (see sbatch/web/client.sh) since it will land on a different host.
#
# Submit from the repo root:  sbatch sbatch/web/server.sh
set -uo pipefail

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=env/webdev.ext3
API_PORT=8000

echo "API server node: $(hostname -s)  (port ${API_PORT})"

# --host 0.0.0.0 so a Vite dev server on another node (or a tunnel) can reach it.
singularity exec --overlay "${OVERLAY}:ro" "${SIF}" /bin/bash -c "
  source /ext3/env.sh
  cd web/server
  export FISBE_ROOT=../../fisbe/completely
  exec uvicorn main:app --host 0.0.0.0 --port ${API_PORT}
"
