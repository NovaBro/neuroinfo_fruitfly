#!/bin/bash
#SBATCH --job-name=nnunet-decode-eval
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=24:00:00
#SBATCH --output=sbatch/nnunet/logs/%x-%A_%a.out
#SBATCH --error=sbatch/nnunet/logs/%x-%A_%a.err

# CPU job (env/evaluate.ext3): decode nnU-Net region probabilities into instances, then score with
# FISBe metrics (same evalinstseg settings as PatchPerPix). Two ways to use it:
#
# 1) Val tuning grid (array): one task per non-comment line of GRID
#      line format:  <name> <ckpt: final|best> <mode: cc|contact> <t_fg> <t_c> <min_core>
#    NPZ = $RUN_DIR/validation_<ckpt>/*.npz, GT = fisbe/completely/val, RM_LIST default "1500".
#      GRID=nnunet_work_folder/val_tune_grid.txt sbatch --export=ALL --array=1-<n> \
#        sbatch/nnunet/nnunet_decode_eval_sbatch.sh
#    -> metrics/nnunet/$EXP/val_tune/<name>/{pred,eval_rm1500/fisbe_summary_rm1500.csv}
#    Rank with: python nnunet_work_folder/collect_scores.py metrics/nnunet/$EXP/val_tune
#
# 2) Single setting (no array), e.g. the frozen val-tuned setting on test:
#      SPLIT=test NAME=final_contact CKPT=final MODE=contact T_FG=0.5 T_C=0.5 MIN_CORE=20 \
#        RM_LIST="1500 500 800" sbatch --export=ALL sbatch/nnunet/nnunet_decode_eval_sbatch.sh
#    NPZ = metrics/nnunet/$EXP/test/nnunet_out_<ckpt>/*.npz -> metrics/nnunet/$EXP/test/<name>/...

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "${REPO_ROOT}"
source sbatch/nnunet/nnunet_common.sh

EXP="${EXP:-fisbe501_resencL_f0}"
SPLIT="${SPLIT:-val}"
RM_LIST="${RM_LIST:-1500}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ -f "${GRID:-}" ]] || { echo "Error: array task needs GRID=<file>" >&2; exit 1; }
  line=$(grep -v '^\s*#' "${GRID}" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
  [[ -n "${line}" ]] || { echo "Error: no line ${SLURM_ARRAY_TASK_ID} in ${GRID}" >&2; exit 1; }
  read -r NAME CKPT MODE T_FG T_C MIN_CORE <<< "${line}"
fi
: "${NAME:?}" "${CKPT:?}" "${MODE:?}" "${T_FG:?}" "${T_C:=0.5}" "${MIN_CORE:=20}"

if [[ "${SPLIT}" == val ]]; then
  NPZ_DIR="${NPZ_DIR:-${RUN_DIR}/validation_${CKPT}}"
  GT_DIR="${GT_DIR:-fisbe/completely/val}"
  BASE="metrics/nnunet/${EXP}/val_tune/${NAME}"
else
  NPZ_DIR="${NPZ_DIR:-metrics/nnunet/${EXP}/test/nnunet_out_${CKPT}}"
  GT_DIR="${GT_DIR:-fisbe/completely/test}"
  BASE="metrics/nnunet/${EXP}/test/${NAME}"
fi
echo "decode-eval name=${NAME} split=${SPLIT} ckpt=${CKPT} mode=${MODE} t_fg=${T_FG} t_c=${T_C} min_core=${MIN_CORE} rm=[${RM_LIST}] npz=${NPZ_DIR}"
ls "${NPZ_DIR}"/*.npz >/dev/null

eval_exec "python -u nnunet_work_folder/decode_instances.py --npz-dir '${NPZ_DIR}' --out '${BASE}/pred' \
  --mode ${MODE} --t-fg ${T_FG} --t-c ${T_C} --min-core ${MIN_CORE} --gt-dir '${GT_DIR}' --workers ${WORKERS} --overwrite"

for RM in ${RM_LIST}; do
  echo "=== score rm${RM} $(date) ==="
  eval_exec "python -u nnunet_work_folder/score_fisbe.py --pred-dir '${BASE}/pred' --gt-dir '${GT_DIR}' \
    --rm ${RM} --out-dir '${BASE}/eval_rm${RM}' --workers ${WORKERS}"
done
echo "=== done $(date) ==="
