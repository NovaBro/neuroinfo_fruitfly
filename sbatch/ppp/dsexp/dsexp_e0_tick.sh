#!/bin/bash
# Runs one Dataset Experiments EXP 0 driver tick, then resubmits itself ~10 min later until the driver is done.
# Start (from repo root):  sbatch sbatch/ppp/dsexp/dsexp_e0_tick.sh
# Stop:                    scancel -n dsexp_e0_tick
#SBATCH --job-name=dsexp_e0_tick
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs,cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=sbatch/ppp/logs/dsexp/dsexp_e0_tick-%j.out
#SBATCH --error=sbatch/ppp/logs/dsexp/dsexp_e0_tick-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/wmz2007/neuroinfo_fruitfly}"
python3 sbatch/ppp/dsexp/dsexp_e0.py tick
rc=$?
if python3 -c 'import json,sys; sys.exit(0 if json.load(open("metrics/ppp/dsexp_e0/state.json")).get("done") else 1)'; then
  echo "EXP 0 driver done; not resubmitting"; exit 0
fi
if [[ "$(squeue -h -u "$USER" -n dsexp_e0_tick -t PENDING | wc -l)" -eq 0 ]]; then
  sbatch --begin=now+600 sbatch/ppp/dsexp/dsexp_e0_tick.sh
fi
exit $rc
