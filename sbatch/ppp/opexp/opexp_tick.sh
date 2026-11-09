#!/bin/bash
# Runs one OP-EXP driver tick, then resubmits itself ~10 min later until the driver reports done.
# Start (from repo root):  sbatch sbatch/ppp/opexp/opexp_tick.sh
# Stop:                    scancel -n opexp_tick
#SBATCH --job-name=opexp_tick
#SBATCH --account=torch_pr_61_tandon_advanced
#SBATCH --partition=cs,cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=sbatch/ppp/logs/opexp/opexp_tick-%j.out
#SBATCH --error=sbatch/ppp/logs/opexp/opexp_tick-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
python3 sbatch/ppp/opexp/opexp.py tick
rc=$?
if python3 -c 'import json,sys; sys.exit(0 if json.load(open("metrics/ppp/opexp/state.json")).get("done") else 1)'; then
  echo "OP-EXP driver done; not resubmitting"; exit 0
fi
# keep exactly one pending tick
if [[ "$(squeue -h -u "$USER" -n opexp_tick -t PENDING | wc -l)" -eq 0 ]]; then
  sbatch --begin=now+600 sbatch/ppp/opexp/opexp_tick.sh
fi
exit $rc
