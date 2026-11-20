#!/bin/bash
# Runs one experiment_v4 driver tick, then resubmits itself ~10 min later until the driver is done.
# Start (from repo root):  sbatch sbatch/ppp/expv4/expv4_tick.sh
# Stop:                    scancel -n expv4_tick
#SBATCH --job-name=expv4_tick
#SBATCH --account=torch_pr_61_tandon_priority
#SBATCH --partition=cs,cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=sbatch/ppp/logs/expv4/expv4_tick-%j.out
#SBATCH --error=sbatch/ppp/logs/expv4/expv4_tick-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
python3 sbatch/ppp/expv4/expv4.py tick
rc=$?
if python3 -c 'import json,sys; sys.exit(0 if json.load(open("metrics/ppp/expv4/state.json")).get("done") else 1)'; then
  echo "experiment_v4 driver done; not resubmitting"; exit 0
fi
if [[ "$(squeue -h -u "$USER" -n expv4_tick -t PENDING | wc -l)" -eq 0 ]]; then
  sbatch --begin=now+600 sbatch/ppp/expv4/expv4_tick.sh
fi
exit $rc
