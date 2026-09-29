#!/bin/bash
# Aggregate the per-sample metrics produced by ppp_array_sbatch.sh into the
# summary table. Run only after every array task has finished. Submit from the
# repo root:
#     sbatch sbatch/ppp/ppp_eval_aggregate_sbatch.sh
#
# evaluation.from_scratch = false and run_ppp only recomputes a sample whose
# result file is missing, so this re-reads the per-sample results rather than
# redoing the expensive matching.
#SBATCH --job-name=ppp_eval
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/ppp/array/ppp_eval_agg.out
#SBATCH --error=sbatch/ppp/array/ppp_eval_agg.err

set -euo pipefail
module purge

EXP_ID=ppp_experiments/flylight_setup01_260630_164437_336076
CONFIG=flylight/setups/setup01/default_l40s.toml

# no --sample: evaluate every volume in data.test_data and emit the summary
singularity exec --nv \
    --overlay env/ppp.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
        python3 -u run_ppp.py --setup setup01 --config ${CONFIG} \
            -d evaluate -id ${EXP_ID} \
            --app flylight --root ppp_experiments --test-checkpoint last"
