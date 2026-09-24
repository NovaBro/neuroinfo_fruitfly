# Layer B — VI merge

Reduce over-fragmentation on `ppp_basic_long8h` by changing vote-instances merge
knobs. No retrain / no re-predict.

Companion: [`ppp_performance_plan.md`](ppp_performance_plan.md) §4,
[`ppp_recommendations.md`](ppp_recommendations.md),
[`ppp_layer_a_prediction.md`](ppp_layer_a_prediction.md).

Primary sample: `R22C03-20180918_66_J2`.

**Active pass:** ckpt **70000** (A2 best). Reuse `test/processed/70000`.  
**Layer B CLOSED** — B1–B7 no winners → next = `ppp_basic_fmaps24` (capacity).  
**Historical:** ckpt 30000 B1–B3 (failed; kept below).

---

## Branch criteria (ckpt 70000)

| Signal | Next |
|--------|------|
| Soft F1 / coverage up vs 0.123, false_split down, MIP more connected | Keep winner; combine if multiple; then Layer C if sizes look healthy |
| B1–B3 near baseline F1≈0.123 | **B5–B7** unused knobs; skip B4 / Layer C; no same-recipe train yet |
| B5–B7 also flat/worse | **Done** — stop VI; next = `ppp_basic_fmaps24` |
| false_merge explodes while splits drop | do not keep that knob |

---

## Baseline (th=0.3, rm0, ckpt 70000)

`mws=true`, `includeSinglePatchCCS=true`, `skeletonize_foreground=true`.

| Metric | Value |
|--------|-------|
| soft avg_f1_cov | **0.123** |
| avg_gt_skel_cov | 0.246 |
| Num Pred / Num GT | 365 / 2 |
| false_split / false_merge / TP@0.5 | 300 / 0 / 0 |

Folder: `test/evaluated/70000/patch_threshold_0_3/fc_threshold_0_3/mws_True/skeletonize_foreground_True/numinst_threshs__0_9_0_1_/`

---

## Output-path rule

`run_ppp.py` names instanced/evaluated folders from `[validation] params_product`
+ `params_zip`. Knobs not in the default product list must be added when changed,
or they overwrite the baseline HDF.

Results for ckpt 70000 land under `evaluated/70000/...` (does not clobber 30000).

---

## B1–B3 @ 70000 — COMPLETED (failed)

Fixed: `patch/fc=0.3`, `ignore_small_comps=100`, `eval_rm0`, `--checkpoint 70000`.

| Job | ID | Role | Result |
|-----|-----|------|--------|
| COMPLETED | 18250469 | B1 mws=false | soft F1 0.123 (flat) |
| COMPLETED | 18250473 | B2 includeSinglePatchCCS=false | soft F1 0.099 (worse) |
| COMPLETED | 18250474 | B3 skeletonize_foreground=false | soft F1 0.122 (flat) |

| Setting | soft avg_f1_cov | avg_gt_skel_cov | Num Pred | false_split | false_merge | TP@0.5 |
|---------|-----------------|-----------------|----------|-------------|-------------|--------|
| baseline th=0.3 ckpt 70000 | 0.123 | 0.246 | 365 | 300 | 0 | 0 |
| B1 mws=false | 0.123 | 0.245 | 311 | 247 | 0 | 0 |
| B2 no single CCS | 0.099 | 0.196 | 140 | 127 | 0 | 0 |
| B3 skel=false | 0.122 | 0.243 | 350 | 289 | 0 | 0 |

**Winners:** none. Same pattern as @ 30000. **B4 skipped.**

---

## B5–B7 — unused knobs

One knob at a time on `processed/70000`. Do not re-run B1–B3.

| Step | Overlay | Change | Path note |
|------|---------|--------|-----------|
| B5 | `vi_merge_numinst_equal.toml` | `numinst_threshs = [0.334, 0.334]` | already in product |
| B6 | `vi_merge_do_nms.toml` | `do_nms = true` | `do_nms` added to product |
| B7 | `vi_merge_aff_vote.toml` | `affinity_graph_voting = true` | `affinity_graph_voting` added to product |

### Timeline

| Job | ID | Result |
|-----|-----|--------|
| TIMEOUT | 18256483 | B5 first try: VI 1:52, wall at 2h mid-eval; HDF written |
| COMPLETED | 18280472 | B5 evaluate-only recovery — soft F1 0.144 (see table) |
| COMPLETED | 18280512 | B6 **label only** (bug: `DO_TASKS=label,evaluate` split by SLURM `--export`) |
| COMPLETED | 18280513 | B7 **label only** (same bug) |

B5 is **not a winner**: soft F1 up slightly but Num Pred/false_split roughly doubled.

### B6/B7 evaluate recovery (HDFs already labeled)

`DO_TASKS` must use **`+`**, not commas (`label+evaluate`), or SLURM truncates at the comma.

```bash
cd /scratch/wmz2007/neuroinfo_fruitfly
CKPT=70000 bash sbatch/ppp/ppp_basic_long8h_vi_b67_eval_chain.sh
```

### Comparison table

| Setting | soft avg_f1_cov | avg_gt_skel_cov | Num Pred | false_split | false_merge | TP@0.5 |
|---------|-----------------|-----------------|----------|-------------|-------------|--------|
| baseline th=0.3 ckpt 70000 | 0.123 | 0.246 | 365 | 300 | 0 | 0 |
| B5 numinst equal | 0.144 | 0.287 | 902 | 613 | 0 | 0 |
| B6 do_nms | 0.123 | — | — | — | — | 0 |
| B7 aff_vote | 0.123 | — | — | — | — | 0 |

**Winners:** none. B5 more fragments; B6/B7 identical to baseline (`18343360` /
`18343361`). **Layer B closed** → `ppp_basic_fmaps24`.

---

## Historical: B1–B3 @ 30000 (2026-09-18) — failed

Baseline then: soft F1 ≈ 0.057, Num Pred 198, false_split 163.

| Job | ID | Role | Result |
|-----|-----|------|--------|
| COMPLETED | 17958803 | B1 mws=false | soft F1 0.056 (flat) |
| COMPLETED | 17958804 | B2 includeSinglePatchCCS=false | soft F1 0.036 (worse) |
| COMPLETED | 17958805 | B3 skeletonize_foreground=false | soft F1 0.057 (flat) |

| Setting | soft avg_f1_cov | avg_gt_skel_cov | Num Pred | false_split | false_merge | TP@0.5 |
|---------|-----------------|-----------------|----------|-------------|-------------|--------|
| baseline th=0.3 | 0.057 | 0.113 | 198 | 163 | 0 | 0 |
| B1 mws=false | 0.056 | 0.113 | 183 | 149 | 0 | 0 |
| B2 no single CCS | 0.036 | 0.072 | 61 | 53 | 0 | 0 |
| B3 skel=false | 0.057 | 0.113 | 194 | 155 | 0 | 0 |

**Winners @ 30000:** none → B4 skipped → A2 resume train.

---

## B4 — combine winners — **skipped** (@ 70000 B1–B3)

None of B1–B3 @ 70000 beat baseline → skip B4; run B5–B7. Do not start Layer C
while MIP is still dots.

---

## Status log

| Date | Step | Notes |
|------|------|-------|
| 2026-09-18 | Artifacts | Overlays + merge chain + this guide |
| 2026-09-18 | B1–B3 @ 30000 COMPLETED | `17958803` → `17958804` → `17958805`; no winner |
| 2026-09-20 | B4 skipped | hand off to Layer A resume 40k→80k |
| 2026-09-22 | Retarget | A2 pick **70000**; B1–B3 submitted |
| 2026-09-22 | B1–B3 @ 70000 COMPLETED | `18250469` → `18250473` → `18250474`; no winner; B4 skipped |
| 2026-09-22 | B5–B7 ready | extra chain: numinst / do_nms / aff_vote |
| 2026-09-22 | B5 TIMEOUT | `18256483` VI 1:52 / 903 labels; B6/B7 blocked by afterok |
| 2026-09-22 | Recovery | 4h wall; independent jobs; B5 evaluate-only |
| 2026-09-22 | B5 eval COMPLETED | `18280472` soft F1 0.144 / Pred 902 — not a winner |
| 2026-09-22 | B6/B7 label-only | `18280512`/`18280513`; commas broke `DO_TASKS` export |
| 2026-09-23 | B6/B7 eval fix | `+` separator; `vi_b67_eval_chain.sh` ready |
| 2026-09-23 | B6/B7 COMPLETED | `18343360`/`18343361` soft F1 0.123 = baseline; Layer B **closed** |
| 2026-09-23 | Next | `ppp_basic_fmaps24` (`num_fmaps=24`, new `-id`) |
