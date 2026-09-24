# Layer A — Prediction quality

Diagnose whether weak `pred_affs` / `pred_numinst` explain over-fragmentation on
`ppp_basic_long8h`, then improve via checkpoint pick and/or resume train.

Companion: [`ppp_performance_plan.md`](ppp_performance_plan.md) §3,
[`ppp_recommendations.md`](ppp_recommendations.md).

Primary sample: `R22C03-20180918_66_J2`. Exp: `metrics/ppp/ppp_basic_long8h`.

---

## Branch criteria

| Signal | Next |
|--------|------|
| FG/numinst/patch F1 decent; MIP has structure but shredded | **Layer B** (VI merge: `mws`, `includeSinglePatchCCS`, …) |
| Weak FG/patch metrics; MIP dots / missing ends; sizes ≪200 | **A1** ckpt sweep → maybe **A2** resume train |
| Mixed | A1 on 20k/40k, then B on best ckpt |

---

## A0 — Diagnose ckpt 30000 (no retrain)

### Pred metrics (fg / numinst / patch)

Overlay: [`PatchPerPix/experiments/flylight/setups/setup01/eval_pred.toml`](../PatchPerPix/experiments/flylight/setups/setup01/eval_pred.toml)
(`prediction_only_test=true`).

```bash
cd /scratch/wmz2007/neuroinfo_fruitfly
CKPT=30000 sbatch --export=ALL,CKPT \
  --job-name=ppp_ckpt_preeval_30000 \
  --output=sbatch/ppp/ppp_basic_long8h_pred_eval_30000.out \
  --error=sbatch/ppp/ppp_basic_long8h_pred_eval_30000.err \
  sbatch/ppp/ppp_basic_long8h_pred_eval_sbatch.sh
```

Output: `metrics/ppp/ppp_basic_long8h/test/evaluated/30000/summary_prediction.csv`
(and stdout logs). Partition: `cs` (192g). Full `eval_patch` may OOM — if so, set
`eval_patch_prediction=false` in the overlay and re-run; use FG mid-channel + diagnose.

### Instance / size / aff activity

```bash
sbatch sbatch/ppp/diagnose_ppp_basic_long8h_sbatch.sh
```

Defaults (th=0.3 VI): HDF under `instanced/30000/patch_threshold_0_3/...`,
zarr `processed/30000/`, report
`metrics/ppp/ppp_basic_long8h/diagnose_instances_report.txt`.

MIP: `test/instanced/30000/patch_threshold_0_3/.../R22C03-20180918_66_J2.png`

### A0 results

| Metric | Value |
|--------|-------|
| FG mid-aff F1 @0.3 / 0.5 | **0.518 / 0.377** (ckpt 30000) |
| Numinst F1 class1 | **0.124** (high prec 0.93, low recall 0.07) |
| Patch F1 | skipped (`eval_patch_prediction=false`; full affs OOM) |
| Diagnose verdict | **MIXED** (198 labels; 12 ≥500; median size 183) |
| n_labels / median / n_ge_500 | 198 / 183 / 12 |
| Branch decision | **Layer B on ckpt 30000** — FG exists but recall/assembly weak; not WEAK_PREDICTIONS |

Pred metrics CSV: `test/processed/30000/R22C03-20180918_66_J2_pred_metrics.csv`  
Diagnose report: `metrics/ppp/ppp_basic_long8h/diagnose_instances_report.txt` (HDF-only; zarr skipped on login)

---

## A1 — Checkpoint sweep (fixed VI th=0.3 + rm0)

For each of **20000, 30000, 40000**: predict (if needed) → fast pred metrics → label+evaluate.

```bash
cd /scratch/wmz2007/neuroinfo_fruitfly
bash sbatch/ppp/ppp_basic_long8h_ckpt_sweep_chain.sh
# FORCE_PREDICT=1 to re-predict even when processed/ exists
# CKPTS="20000 40000" to subset
```

Scripts:

| Role | Path |
|------|------|
| Chain | `sbatch/ppp/ppp_basic_long8h_ckpt_sweep_chain.sh` |
| Predict (+ optional mid-aff metrics) | `sbatch/ppp/ppp_basic_long8h_ckpt_predict_sbatch.sh` |
| Pred eval (optional; `eval_pred.toml`) | `sbatch/ppp/ppp_basic_long8h_pred_eval_sbatch.sh` |
| Label+eval (runs `diagnose_ppp_pred_metrics.py` then VI) | `sbatch/ppp/ppp_basic_long8h_ckpt_label_eval_sbatch.sh` |

Fast metrics helper: `sbatch/ppp/diagnose_ppp_pred_metrics.py` (mid-aff FG + numinst; no full patch load).

### Completed chain (2026-09-18)

| Job | ID | Role |
|-----|-----|------|
| COMPLETED | 17950828 | predict 20000 |
| COMPLETED | 17951057 | VI 20000 |
| COMPLETED | 17951058 | VI 30000 |
| COMPLETED | 17951059 | predict 40000 |
| COMPLETED | 17951060 | VI 40000 |
| COMPLETED | 17951063 | pred metrics 30000 |

Sources:

- `test/processed/<ckpt>/R22C03-*_pred_metrics.csv`
- `test/evaluated/<ckpt>/patch_threshold_0_3/.../summary.csv`
- `sbatch/ppp/ppp_basic_long8h_ckpt_vi_<ckpt>.out` (`avg_f1_cov_score`)

### Comparison table

| Ckpt | FG F1@0.3 | Numinst1 F1 | soft avg_f1_cov | avg_gt_skel_cov | Num Pred | false_split |
|------|-----------|-------------|-----------------|-----------------|----------|-------------|
| 20000 | 0.263 | 0.077 | 0.018 | 0.036 | 81 | 66 |
| **30000** | **0.518** | **0.124** | **0.057** | **0.113** | 198 | 163 |
| 40000 | 0.466 | 0.107 | 0.048 | 0.097 | 133 | 123 |

All three: TP@0.5 = 0, false_merge = 0.

**Best ckpt (A1): 30000.** 40k is slightly worse on FG F1, numinst F1, soft F1, and coverage.
Used `test/processed/30000` for Layer B (B1–B3 failed → return here for A2).

---

## A2 — Resume train to 80k — **COMPLETED**

Layer B B1–B3 did not beat soft F1 ≈ 0.057 (see [`ppp_layer_b_merge.md`](ppp_layer_b_merge.md)).
Resumed from ckpt **40000** → **80000** (train job `18195847` COMPLETED), then
re-swept 50/60/70/80k.

```bash
# Train only (do NOT use ppp_basic_long8h_chain.sh for scoring)
sbatch sbatch/ppp/ppp_basic_long8h_train_sbatch.sh

# After train_net_checkpoint_80000 exists:
CKPTS="50000 60000 70000 80000" bash sbatch/ppp/ppp_basic_long8h_ckpt_sweep_chain.sh
```

Do not change `num_fmaps` / `train_code` / `batch_size` in this pass. Keep VI at
`vi_th_0_3` + `eval_rm0`. Sample: `R22C03-20180918_66_J2`.

### Completed A2 sweep (2026-09-21 → 2026-09-22)

| Job | ID | Role |
|-----|-----|------|
| COMPLETED | 18195847 | train → 80000 |
| COMPLETED | 18229168 | predict 50000 |
| COMPLETED | 18229170 | VI 50000 |
| COMPLETED | 18229172 | predict 60000 |
| COMPLETED | 18229174 | VI 60000 |
| COMPLETED | 18229176 | predict 70000 |
| COMPLETED | 18229178 | VI 70000 |
| COMPLETED | 18229179 | predict 80000 |
| COMPLETED | 18229181 | VI 80000 |

### A2 comparison table

| Ckpt | FG F1@0.3 | Numinst1 F1 | soft avg_f1_cov | avg_gt_skel_cov | Num Pred | false_split |
|------|-----------|-------------|-----------------|-----------------|----------|-------------|
| 30000 (A1 baseline) | 0.518 | 0.124 | 0.057 | 0.113 | 198 | 163 |
| 50000 | 0.576 | 0.133 | 0.077 | 0.155 | 155 | 141 |
| 60000 | 0.600 | 0.161 | 0.079 | 0.159 | 186 | 163 |
| **70000** | **0.620** | **0.298** | **0.123** | **0.246** | **365** | **300** |
| 80000 | 0.605 | 0.141 | 0.099 | 0.197 | 256 | 218 |

All five: TP@0.5 = 0, false_merge = 0. Tiny th_0.1 fscore appears at 70k/80k (~0.005–0.008).

**Best ckpt (A2): 70000.** Soft F1 roughly doubled vs 30k; 80k regressed (same pattern as
40k vs 30k). Still over-fragmented (MIP denser dots, not continuous arbors). Layer B @ 70000
closed with no winners → next capacity run: **`ppp_basic_fmaps24`** (see performance plan).

---

## Status log

| Date | Step | Notes |
|------|------|-------|
| 2026-09-18 | Artifacts | Overlays + sbatches + this guide created |
| 2026-09-18 | A0 | Diagnose HDF MIXED; pred metrics 30000 done |
| 2026-09-18 | A1 COMPLETED | chain `17950828` / `17951057`–`17951060` / `17951063`; pick **30000** |
| 2026-09-18 | A2 skipped | FG usable; 40k lost to 30k; hand off to Layer B |
| 2026-09-20 | A2 started | Layer B failed; resume 40k→80k then CKPTS 50/60/70/80k |
| 2026-09-21 | A2 train COMPLETED | job `18195847` → `train_net_checkpoint_80000` |
| 2026-09-22 | A2 sweep COMPLETED | `18229168`–`18229181`; pick **70000**; Layer B retry |
