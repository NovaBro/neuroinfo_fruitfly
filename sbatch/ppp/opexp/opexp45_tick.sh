#!/bin/bash
# Runs one experiment_v2 OP-EXP 4/5 driver tick, then resubmits itself ~10 min later until the driver is done.
# Start (from repo root):  sbatch sbatch/ppp/opexp/opexp45_tick.sh
# Stop:                    scancel -n opexp45_tick
#SBATCH --job-name=opexp45_tick
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs,cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=sbatch/ppp/logs/opexp/opexp45_tick-%j.out
#SBATCH --error=sbatch/ppp/logs/opexp/opexp45_tick-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
python3 sbatch/ppp/opexp/opexp45.py tick
rc=$?
if python3 -c 'import json,sys; sys.exit(0 if json.load(open("metrics/ppp/opexp45/state.json")).get("done") else 1)'; then
  echo "OP-EXP 4/5 driver done; not resubmitting"; exit 0
fi
if [[ "$(squeue -h -u "$USER" -n opexp45_tick -t PENDING | wc -l)" -eq 0 ]]; then
  sbatch --begin=now+600 sbatch/ppp/opexp/opexp45_tick.sh
fi
exit $rc
