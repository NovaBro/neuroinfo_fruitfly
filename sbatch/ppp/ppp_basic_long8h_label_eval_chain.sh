#!/bin/bash
# Submit per-sample label arrays (val + test) on l40s_public, then CPU eval
# with --dependency=afterok. From repo root:
#   bash sbatch/ppp/ppp_basic_long8h_label_eval_chain.sh
set -euo pipefail
cd /scratch/wmz2007/neuroinfo_fruitfly
mkdir -p sbatch/ppp/array

ARRAY_SCRIPT=sbatch/ppp/ppp_basic_long8h_label_array_sbatch.sh
EVAL_SCRIPT=sbatch/ppp/ppp_basic_long8h_eval_agg_sbatch.sh

VAL=$(sbatch --parsable --export=ALL,SPLIT=val --array=0-4%2 "${ARRAY_SCRIPT}")
echo "submitted val label array: ${VAL}"
VAL_EVAL=$(sbatch --parsable --export=ALL,SPLIT=val --dependency=afterok:"${VAL}" "${EVAL_SCRIPT}")
echo "submitted val eval_agg (afterok:${VAL}): ${VAL_EVAL}"

TEST=$(sbatch --parsable --export=ALL,SPLIT=test --array=0-6%2 "${ARRAY_SCRIPT}")
echo "submitted test label array: ${TEST}"
TEST_EVAL=$(sbatch --parsable --export=ALL,SPLIT=test --dependency=afterok:"${TEST}" "${EVAL_SCRIPT}")
echo "submitted test eval_agg (afterok:${TEST}): ${TEST_EVAL}"

echo "---"
echo "Monitor: squeue -u \$USER"
echo "HDFs: metrics/ppp/ppp_basic_long8h/test/instanced/70000/patch_threshold_0_3/fc_threshold_0_3/mws_True/skeletonize_foreground_True/numinst_threshs__0_9_0_1_/"
