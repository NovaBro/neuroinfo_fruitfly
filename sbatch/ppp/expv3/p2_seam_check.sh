#!/bin/bash
# experiment_v3 P2 seam check (md_guides/experiments/experiment_v3.md). From the repo root:
#   sbatch/ppp/expv3/p2_seam_check.sh SRC_EXP CKPT [SAMPLE]
# Predicts one completely val volume with checkpoint SRC_EXP @ CKPT two ways, both with 160^3 output tiles on the
# same grid: abutting tiles (margin 0, expv3_predict_m0.toml) and production (224^3 margin 32, expv3_predict.toml).
# Each run gets its own exp dir expv3-p2-m{0,32} whose train/ holds a symlink to the checkpoint. A CPU job (afterok)
# then runs compare_same_margin.py (m0 vs m32) and writes metrics/ppp/expv3/p2_seam_check.md.
# Prints "p2 jobs=<pred ids> compare=<id>". (A 256^3 / margin-48 arm was dropped: 256^3 predict tiles OOM on L40S,
# a 216 GiB allocation; the 2026-11-17 m0-vs-m32 run already showed the border effect at noise level beyond 16-32 vox.)
set -euo pipefail
SRC=$1; CKPT=$2; SAMPLE=${3:-completely/R22C03-20180918_66_J2}
REPO=/scratch/wmz2007/neuroinfo_fruitfly; cd "$REPO"
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
STACK=(--config expv2_e1_common.toml --config expv3_s6d2w64.toml --config expv3_crop128_bs2.toml
       --config expv3_pre.toml --config pred_sparse_affs.toml)
ids=()
for m in 0 32; do
  exp=expv3-p2-m$m
  case $m in 0) ov=expv3_predict_m0.toml;; 32) ov=expv3_predict.toml;; esac
  mkdir -p metrics/ppp/$exp/train
  ln -sfn "$REPO/metrics/ppp/$SRC/train/train_net_checkpoint_$CKPT" metrics/ppp/$exp/train/train_net_checkpoint_$CKPT
  cp -f metrics/ppp/$SRC/train/train_net_config.json metrics/ppp/$exp/train/ 2>/dev/null || true
  out=$(./sbatch/ppp/ppp_submit_test.sh run our-noenc-c1k_v1.1 --id $exp --ckpt "$CKPT" --split val --sample "$SAMPLE" \
        --base original_default.toml --eval eval_rm1500.toml "${STACK[@]}" --config $ov --do predict \
        --keep-pred-numinst --time 00:45:00 --account torch_pr_61_general 2>&1)
  id=$(echo "$out" | grep -o 'run=[0-9]*' | cut -d= -f2)
  [[ -n "$id" ]] || { echo "submit failed for m$m:"; echo "$out"; exit 1; }
  ids+=("$id")
done
P=val/processed/$CKPT/$SAMPLE.zarr
mkdir -p metrics/ppp/expv3 sbatch/ppp/logs/expv3
REPORT=metrics/ppp/expv3/p2_seam_check.md
cmp=$(sbatch --parsable --account=torch_pr_61_general --partition=cs,cpu_short --cpus-per-task=4 --mem=32G \
  --time=00:40:00 --job-name=expv3_p2_compare --dependency=afterok:$(IFS=:; echo "${ids[*]}") \
  --output=sbatch/ppp/logs/expv3/p2_compare-%j.out \
  --wrap="cd $REPO && singularity exec --overlay env/ppp.ext3:ro $SIF /bin/bash -c 'source /ext3/env.sh; conda activate ppp; \
    { echo \"# experiment_v3 P2 seam check: $SRC @ $CKPT, $SAMPLE\"; echo; \
      echo \"## margin 0 (abutting 160^3 tiles) vs margin 32 (224^3 tiles, production)\"; echo; \
      python sbatch/ppp/tools/compare_same_margin.py metrics/ppp/expv3-p2-m0/$P metrics/ppp/expv3-p2-m32/$P --tile 160; \
    } > $REPORT'")
echo "p2 jobs=$(IFS=,; echo "${ids[*]}") compare=$cmp"
