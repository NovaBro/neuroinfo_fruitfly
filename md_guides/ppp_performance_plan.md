# PatchPerPix performance improvement plan

Actionable plan to raise **instance quality** (F1 / skeleton coverage / fewer false
splits) and **train GPU utilization** for the FISBe PPP runs under
`metrics/ppp/ppp_basic_long8h`.

Companion post-mortem / job plumbing: [`ppp_recommendations.md`](ppp_recommendations.md).
Architecture: [`PatchPerPix/CODEBASE.md`](../PatchPerPix/CODEBASE.md).

Primary debug sample: `R22C03-20180918_66_J2` (GT = 2 neurons under
`gt_instances_rm_5`).

---

## 0. Current baseline (as of 2026-09-23)

| Item | Status |
|------|--------|
| Exp id (closed) | `metrics/ppp/ppp_basic_long8h` |
| Eval ckpt | **70000** (best of A2 50–80k sweep); 80000 exists but lost vs 70k |
| Soft eval (`rm0`) best | **avg_f1_cov ≈ 0.123** at `patch/fc=0.3` (ckpt 70000) |
| Hard truth | **TP = 0** at th_0.5; Num Pred 365 vs GT 2 (ckpt 70000, th=0.3) |
| Failure mode | **Over-fragmentation** (high `false_split`, near-zero merges) |
| Layer A | **Done** — A1 pick 30000; A2 train+sweep done; pick **70000** (FG F1@0.3 = 0.62) |
| Layer B @ 30000 | Failed (B1–B3 flat/worse); B4 skipped |
| Layer B @ 70000 | **Closed** — B1–B7 no winners (B6/B7 soft F1 0.123 = baseline) |
| Next | **fmaps24 ckpt sweep** 50/60/70/80k (pilot @ 80k soft F1 **0.170** > long8h 0.123) |
| rm500 trap | Eval `remove_small_components=500` → F1 ≈ 0.001 (fragments all tiny) |
| Train GPU util | Mean **~44%**, VRAM **~91%** (batch_size=1, L40S) |

Threshold-only VI sweep already done (`vi_th_0_{3..7}`). Affinities improved with
A2; VI knobs did not fix assembly → capacity experiment under a new `-id`.

---

## 1. Mental model — three layers

```text
┌─────────────────────────────────────────────────────────────┐
│  A. Prediction quality   (pred_affs / pred_numinst)         │
│     cost: retrain and/or re-predict                         │
├─────────────────────────────────────────────────────────────┤
│  B. VI merge behavior    (vote_instances assembly)          │
│     cost: label + evaluate only (reuse processed/)          │
├─────────────────────────────────────────────────────────────┤
│  C. Size filters         (mask / post-VI / eval cutoffs)    │
│     cost: evaluate only, or relabel if VI filter changes    │
└─────────────────────────────────────────────────────────────┘
```

Do **not** tune C under `rm500` while debugging — it hides fragment counts.
Use `eval_rm0.toml` until instance sizes are large enough to keep.

---

## 2. Priority order (recommended)

```text
1. Diagnose prediction  ……… is VI the bottleneck or are affs weak?
2. VI merge knobs       ……… mws, includeSinglePatchCCS, numinst, skeletonize
3. Size filters         ……… only after fragments look like arbors
4. Train longer / ckpt  ……… better affs; pick best checkpoint
5. Widen samples        ……… drop --sample once one volume improves
6. GPU util             ……… parallel: safer long jobs on l40s_public
```

Success bar per stage is below. Stop and branch when a stage fails its bar.

---

## 3. Layer A — Prediction quality

**Detailed runbook:** [`ppp_layer_a_prediction.md`](ppp_layer_a_prediction.md)
(A0/A1/A2 **done**). Best ckpt **70000** (was 30000 after A1).

### 3.1 Diagnose (before more VI)

Turn on intermediate metrics (currently off in config):

```toml
[evaluation.prediction]
eval_numinst_prediction = true
eval_fg_prediction = true
eval_patch_prediction = true
```

Also:

- Instance / FG MIP under `test/instanced/.../*.png`
- `sbatch/ppp/diagnose_ppp_basic_instances.py` on the HDF (size histogram)
- Spot-check `volumes/pred_affs` / `pred_numinst` on a **compute node** (login OOM risk)

| Signal | Branch |
|--------|--------|
| Strong FG/affs, shredded instances | Stay on Layer B (VI) |
| Weak / noisy affs, missing ends in MIP | Layer A train / ckpt |
| All labels size ≪ 200 | Affs + cover weak; VI alone won’t save it |

### 3.2 Train / checkpoint levers

| Lever | Config / CLI | Notes |
|-------|--------------|-------|
| More iterations | `[training] max_iterations` | long8h reached 80k; plateaued for VI |
| Checkpoint pick | `--checkpoint 50000\|60000\|70000\|80000` | Last ≠ best; re-predict per ckpt |
| Resume same `-id` | `ppp_basic_long8h_train_sbatch.sh` | Closed recipe; do not extend |
| Sampling balance | `[training.sampling]` | Leave for a later `-id` if fmaps24 fails |
| Loss / bg | `mask_bg_weight`, `add_affinities="loss"` | Keep aff targets on GPU |
| Capacity | `num_fmaps=24` under **`ppp_basic_fmaps24`** | Pilot soft F1 0.170 @ 80k; **ckpt sweep next** |

**Layer B closed → next is `ppp_basic_fmaps24` (do not resume long8h same-recipe).**

**Success (A):** MIP shows continuous processes (not only dots); soft F1 and
`avg_gt_skel_coverage` rise before VI retuning; false_split drops at fixed VI
settings.

---

## 4. Layer B — VI merge behavior

**Detailed runbook:** [`ppp_layer_b_merge.md`](ppp_layer_b_merge.md)
(**CLOSED** on ckpt 70000 — B1–B7 no winners).

### 4.1 Already swept (no further VI on long8h)

| Knob | Tried | Best so far |
|------|-------|-------------|
| `patch_threshold` / `fc_threshold` | 0.3–0.7 | **0.3** (soft F1 ≈ 0.123 at ckpt 70000) |
| `ignore_small_comps` | 100 in sweep overlays | Soften while debugging |
| B1–B3 @ 30000 / @ 70000 | mws / single CCS / skel | no winner |
| B5 numinst equal | `[0.334, 0.334]` | soft F1 0.144 but Pred 902 — not a winner |
| B6 `do_nms` / B7 `aff_vote` | evaluate `18343360`/`18343361` | soft F1 **0.123** = baseline |

Overlays: `PatchPerPix/experiments/flylight/setups/setup01/vi_th_0_{3..7}.toml`  
**Stop VI tweaking on this recipe.** Next: capacity under new `-id` (§8 / `ppp_basic_fmaps24`).

---

## 5. Layer C — Size filters

Three different cutoffs — do not confuse them:

| Knob | Section | When applied | Debug default | Report later |
|------|---------|--------------|---------------|--------------|
| `ignore_small_comps` | `[vote_instances]` | Before FG cover | 50–100 | ~200 |
| `remove_small_comps` | `[vote_instances]` | After labeling (HDF) | **0** | match desired min neuron size |
| `remove_small_components` | `[evaluation]` | Metric only | **0** (`eval_rm0`) | 200–500 once sizes OK |
| `prediction.clean_mask` | `[prediction]` | Predict FG cleanup | 600 | retune if FG sparse |

**Rule:** Keep eval at **rm0** until `diagnose_ppp_basic_instances.py` shows a
healthy tail of labels ≥200 (and preferably ≥500). Then set VI
`remove_small_comps` and eval filter consistently.

---

## 6. GPU utilization (train throughput / idle-kill)

Latest train log: `gpu_usage_log_ppp_basic_long8h_train.csv` — mean util **~44%**,
VRAM **~42 / 46 GB**, `training.batch_size = 1`.

Already in place: `$SLURM_TMPDIR` staging, `num_workers=16`, `cache_size=48`, AMP,
nvidia-smi logger.

### 6.1 Highest leverage

| Change | Where | Risk |
|--------|-------|------|
| `training.batch_size` 1 → 2 | overlay under **`[training]`** (not `[prediction]`) | OOM — VRAM already ~91% |
| If OOM: shrink `train_input_shape_valid` slightly (140→128) then retry batch 2 | `[model]` | Slightly less context/crop |
| More CPUs + workers (24–32) | sbatch `--cpus-per-task` + `[training] num_workers` / `cache_size` | Queue time |

### 6.2 Secondary

- Raise `val_log_step` (100 → 500+) to cut val-pipeline dips
- Less frequent snapshots if they stall the loop
- Diagnose CPU bound: briefly soften elastic/overlay; if util jumps, PreCache is starving

### 6.3 How to judge

After a short train window, compare:

- Mean `utilization.gpu` in the CSV (target **~60%+** if batch 2 fits)
- Mean `Batch: iteration=…, time=` in the `.out` log

Util up + batch time up ≈ more GPU work per step. Util flat ≈ still CPU/I/O bound.

**Note:** Util tweaks do **not** fix F1 by themselves; they make longer Part-3 trains
safer on `l40s_public`.

---

## 7. Evaluation protocol (keep consistent)

While debugging:

```text
-d label evaluate   # or evaluate-only
+ eval_rm0.toml
+ best / experimental VI overlay
--checkpoint <N>
--sample R22C03-20180918_66_J2
```

Report at least:

- `general.Num Pred` vs `Num GT`
- `general.avg_f1_cov_score`, `avg_gt_skel_coverage`
- `confusion_matrix.th_0_5` TP / FP / FN / false_split / false_merge
- MIP under `instanced/.../*.png`
- Optional: label-size histogram from diagnose script

When one setting wins on the pilot sample:

1. Drop `--sample` (14 val volumes; costlier)
2. Re-enable a moderate size filter for “reporting” numbers
3. Optionally re-predict with `--test-checkpoint last` or best numeric ckpt

---

## 8. Concrete next checklist

- [x] **A0** Pred metrics + diagnose on ckpt 30000 — MIXED; FG F1@0.3 = 0.52
- [x] **A1** Ckpt sweep 20k/30k/40k — pick **30000**
- [x] **B1–B3 @ 30000** — no winner (`17958803`–`05`); B4 skipped
- [x] **A2** Resume to 80k + sweep 50/60/70/80k — pick **70000** (soft F1 0.123)
- [x] **B1–B3 @ 70000** — no winner (`18250469` / `18250473` / `18250474`); B4 skipped
- [x] **B5 @ 70000** `numinst_threshs=[0.334, 0.334]` — scored; soft F1 0.144 / Pred 902 — not a winner
- [x] **B6 @ 70000** `do_nms=true` — soft F1 0.123 (flat); not a winner
- [x] **B7 @ 70000** `affinity_graph_voting=true` — soft F1 0.123 (flat); not a winner
- [x] **Capacity** `ppp_basic_fmaps24` — train+pilot infer @ 80000 soft F1 **0.170** (Pred 257) > long8h 0.123
- [ ] **fmaps24 ckpt sweep** 50/60/70/80k — pick best before more recipe changes
- [ ] **C** Only if sizes look healthy: set `remove_small_comps` / eval filter deliberately
- [ ] **U** Try `training.batch_size=2` (fix OOM); log util before/after
- [ ] **W** Full val (14) + test (20) @ long8h ckpt 70000 — sequential scripts ready (parallel OK with fmaps24)

---

## 9. Pointers

| Item | Path |
|------|------|
| This plan | `md_guides/ppp_performance_plan.md` |
| Job post-mortem | `md_guides/ppp_recommendations.md` |
| Layer A runbook | `md_guides/ppp_layer_a_prediction.md` |
| Layer B runbook (closed) | `md_guides/ppp_layer_b_merge.md` |
| **Next** | `CKPTS="50000 60000 70000 80000" bash sbatch/ppp/ppp_basic_fmaps24_ckpt_sweep_chain.sh` |
| Full val @ 70000 | `sbatch sbatch/ppp/ppp_basic_long8h_full_val_sbatch.sh` |
| Full test @ 70000 | `sbatch sbatch/ppp/ppp_basic_long8h_full_test_sbatch.sh` |
| fmaps24 overlay | `…/setup01/basic_fmaps24.toml` (`num_fmaps=24`) |
| fmaps24 chain | `bash sbatch/ppp/ppp_basic_fmaps24_chain.sh` |
| Soft eval overlay | `PatchPerPix/experiments/flylight/setups/setup01/eval_rm0.toml` |
| Long overlay (closed) | `…/setup01/basic_long8h.toml` |
| Diagnose script | `sbatch/ppp/diagnose_ppp_basic_instances.py` |
| Exp outputs (closed) | `metrics/ppp/ppp_basic_long8h/` |
| Exp outputs (next) | `metrics/ppp/ppp_basic_fmaps24/` |
| GPU util log | `gpu_usage_log_ppp_basic_fmaps24_train.csv` |

---

## 10. Glossary

- **VI (`vote_instances`):** Affinities + numinst → instance IDs (CUDA graph assembly).
- **`patch_threshold` / `fc_threshold`:** Patch admission / foreground-cover thresholds.
- **`mws`:** Mutex Watershed partition of the patch affinity graph.
- **`ignore_small_comps`:** Drop tiny FG components **inside** VI (pre-cover).
- **`remove_small_comps`:** Drop tiny instances **after** VI (writes into result).
- **`remove_small_components`:** Drop tiny instances **only at eval** (scoring).
- **`avg_f1_cov_score`:** Primary scalar in PPP flylight eval (`general.avg_f1_cov_score`).
