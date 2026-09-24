# PatchPerPix (`ppp_basic`) — Recommendations

Post-mortem and next steps after the **`ppp_basic_8h`** smoke run and the
**`ppp_basic_long8h`** (~30k iter) follow-up on FISBe. Primary debug sample:
`R22C03-20180918_66_J2`.

**Forward performance plan** (fmaps24 pilot soft F1 **0.170** @ 80k; **ckpt sweep next**):
[`ppp_performance_plan.md`](ppp_performance_plan.md).

**Layer A (prediction quality) runbook — A0/A1/A2 filled, pick ckpt 70000:**
[`ppp_layer_a_prediction.md`](ppp_layer_a_prediction.md).

**Layer B (VI merge) runbook — CLOSED (B1–B7 no winners):**
[`ppp_layer_b_merge.md`](ppp_layer_b_merge.md).

Use this as a checklist before the next PatchPerPix train/infer cycle.

---

## 1. What already worked

| Area | Outcome |
|------|---------|
| Train job | Smoke `ppp_basic_train` (17917065) COMPLETED; ~1h15m, 8700 iters |
| Infer job | Smoke `ppp_basic_infer` (17917066) COMPLETED; predict → VI → evaluate |
| Job split | Train and infer as **separate sbatches** with `afterok` freed GPU VRAM |
| `@fork` on train | Must stay **disabled** (spawn + gunpowder `PreCache` after CUDA hung the GPU) |
| One-shot sbatch | `ppp_basic_8h_sbatch.sh` deprecated; use chain scripts |
| Eval filter (smoke) | `remove_small_components=500` wiped all preds; re-eval with `0` → Num Pred = 170 |
| Longer train | `ppp_basic_long8h` resumed to **30000** (jobs 17928977 train + 17928978 infer COMPLETED) |
| GPU feed | Stage to `$SLURM_TMPDIR`; `num_workers=16`, `cache_size=48`; nvidia-smi logger |

**Submit patterns:**

```bash
cd /scratch/wmz2007/neuroinfo_fruitfly
bash sbatch/ppp/ppp_basic_8h_chain.sh        # smoke (~8.7k)
bash sbatch/ppp/ppp_basic_long8h_chain.sh    # longer (resume-friendly; currently 30k / 5h wall)
```

Stable roots: `--root ../../metrics/ppp` with `-id ppp_basic_8h` or `-id ppp_basic_long8h`.

---

## 2. Training quality snapshot

### 2.1 Smoke (`ppp_basic_8h`, ckpt 8000)

- Config: `default_l40s.toml` + `basic_8h.toml`
- `max_iterations = 8700`. Full-ish schedule in `default.toml` uses **`400002`**
- Loss: ~0.72 @ iter 100 → ~0.002 @ iter 8700
- Augmentations **on** during train (not at predict/VI/eval)

| Setting | Num Pred | avg_f1_cov_score | Notes |
|---------|----------|------------------|-------|
| `remove_small_components=500` | 0 | 0.0 | All VI labels size ≤432 |
| `remove_small_components=0` | 170 | ~0.020 | 145 false_splits; no TP @ 0.5 |

Verdict then: **`FILTERED_BY_RM500`** for zero Pred; underlying issue **over-fragmentation**.

### 2.2 Longer run (`ppp_basic_long8h`, ckpt 30000)

- Config: `default_l40s.toml` + `basic_long8h.toml` (`max_iterations=30000`, workers=16)
- Eval with default `remove_small_components=500`: **Num GT=2, Num Pred=2**, but
  `avg_f1_cov_score ≈ 0.001`, `avg_gt_skel_coverage ≈ 0.002`, **0 TP** at all thresholds
- **MIP (human review):** many dots in roughly the right region; missing neuron ends /
  thin processes; looks like an **unmerged / under-assembled** VI output, not a
  totally misplaced prediction

So count can match GT after rm500 while quality is still bad: surviving blobs do not
cover GT skeletons.

### 2.3 Why only one “validation” sample at infer?

Infer was intentionally limited for cheap pilots. Both
`ppp_basic_8h_infer_sbatch.sh` and `ppp_basic_long8h_infer_sbatch.sh` pass:

```text
--sample R22C03-20180918_66_J2
```

`fisbe/completely/val` has **14** volumes. Drop `--sample` to score more (costlier).
`data.max_val_samples=10` only caps train-time val logging, not this CLI filter.

---

## 3. Score-fix plan (current priority)

Given the MIP (dots, right area, missing ends) and metrics (count OK, coverage ~0):
treat this as **assembly + affinity quality**, not job plumbing.

### Part 2 — Soft eval + VI threshold sweep (no retrain; do first)

**Goal:** Merge speckles into continuous arbors; measure with a soft size filter so
dust does not hide true fragment counts.

#### 2A. Soft re-eval of current instances (CPU, minutes)

Reuse existing HDF under `metrics/ppp/ppp_basic_long8h/test/instanced/30000/...`.
Only change the eval filter:

- Overlay: `eval_rm0.toml` (`remove_small_components = 0`, `from_scratch = true`)
- `-id ppp_basic_long8h`, `--checkpoint 30000` (numeric; `--test-checkpoint` only accepts `last`/`best`)
- `-d evaluate` only (pattern: `ppp_basic_long8h_eval_rm0_sbatch.sh`)
- Keep `--sample R22C03-20180918_66_J2` for now

**Does not** re-merge dots — only reports true Num Pred / false_splits before rm500.

#### 2B. VI threshold sweep (GPU; reuse `processed/30000` preds)

Predictions stay; only re-run assembly:

1. Add a small TOML overlay (or one setting at a time under `[vote_instances]`):
   - `patch_threshold` / `fc_threshold` pairs: `(0.3,0.3)`, `(0.4,0.4)`, `(0.5,0.5)`,
     `(0.6,0.6)`, `(0.7,0.7)`
   - Optionally lower `ignore_small_comps` (e.g. 50–100) while debugging
   - Stack `eval_rm0.toml` so scores are not wiped by rm500
2. For each setting run:
   - `-d label evaluate` (**no** `predict` — preds already in `test/processed/30000/`)
   - same `-id`, `--checkpoint 30000`, same `--sample`
3. Compare: MIP continuity, Num Pred vs GT=2, `avg_f1_cov_score`, false_split
4. Later: drop `--sample` and run a few more val volumes with the **best** VI settings only

One-at-a-time overlays + `label evaluate` is simpler than a full
`validate_checkpoints` product sweep. Outputs land under distinct
`instanced/30000/patch_threshold_…/fc_threshold_…/` folders.

| Knob | Current | Direction to try |
|------|---------|------------------|
| `patch_threshold` | 0.5 | Sweep 0.3–0.7 |
| `fc_threshold` | 0.5 | Sweep with patch_threshold |
| `ignore_small_comps` | 200 | Try 50–100 while debugging |
| `numinst_threshs` | [0.9, 0.1] | Only if numinst looks mis-calibrated |
| `prediction.fg_thresh` | 0.5 | Lower (e.g. 0.3) if FG too sparse (needs re-predict) |

**Success signal (Part 2):** MIP looks like connected arbors (not dots); Num Pred near 2;
coverage/F1 up. If dots remain across thresholds → affs still weak → Part 3.

---

### Part 3 — Train longer (A2 COMPLETED)

**Goal:** Better affinities so thin ends and merges exist *before* VI.

A2 reached 80k; best ckpt **70000** (soft F1 0.123, FG F1@0.3 0.620). Layer B @ 70000
**closed** (B1–B7 no winners). Next is a **new** `-id` capacity run (`ppp_basic_fmaps24`),
not more same-recipe train.

---

### Suggested order (given current MIP)

```text
1. Soft re-eval (Part 2A) — done via VI sweep with eval_rm0
2. VI threshold sweep (Part 2B) — done; best th=0.3
3. Layer A ckpt sweep A1 — done; pick 30000
4. Layer B VI merge @ 30000 — done; no winner; B4 skipped
5. A2 resume 40k→80k + sweep — done; pick 70000 (soft F1 0.123)
6. Layer B VI merge @ 70000 B1–B3 — done; no winner; B4 skipped
7. Layer B B5–B7 — done; no winners (B6/B7 soft F1 0.123 = baseline)
8. **Capacity** `ppp_basic_fmaps24` — done; pilot @ 80k soft F1 **0.170**
9. **Next:** fmaps24 ckpt sweep 50/60/70/80k
10. Widen beyond one sample last
```

What will **not** fix the score by itself: more GPU-util/staging tweaks, re-enabling
`@fork` on train, or staring at F1 under rm500 without MIP/coverage.

### Implemented artifacts (fmaps24 pilot done → ckpt sweep)

**Layer B B5 (COMPLETED, not a winner):** evaluate `18280472` — soft F1 0.144, Num Pred
902, false_split 613 vs baseline 0.123 / 365 / 300.

**Layer B B6/B7 (COMPLETED, not winners):** evaluate `18343360` / `18343361` — soft F1
**0.123** (identical to baseline). Layer B closed on this recipe.

**fmaps24 pilot (COMPLETED):** infer `18429226` @ ckpt 80000 — soft F1 **0.170**, skel cov
0.337, Num Pred 257, false_split 224 (vs long8h @ 70k: 0.123 / 0.246 / 365 / 300). Still TP=0.

**Next — fmaps24 ckpt sweep** (pick best before more recipe changes):

```bash
cd /scratch/wmz2007/neuroinfo_fruitfly
CKPTS="50000 60000 70000 80000" bash sbatch/ppp/ppp_basic_fmaps24_ckpt_sweep_chain.sh
```

Do **not** submit `train_code` or sampling changes until the sweep picks a ckpt. Compare soft F1 /
Num Pred / MIP to long8h @ 70000 (baseline soft F1 **0.123** on R22C03).

**Full-split sequential (long8h @ 70000, th=0.3 + rm0):** can run alongside fmaps24 train.
Pilot sample remains the primary debug number; val/test summaries land under
`evaluated/70000/...`. If wall TIMEOUT, re-submit (skips finished volumes); else switch to array.

```bash
sbatch sbatch/ppp/ppp_basic_long8h_full_val_sbatch.sh
sbatch sbatch/ppp/ppp_basic_long8h_full_test_sbatch.sh
```

---

## 4. Broader practices (still in force)

**Keep:**

- Train/infer **job split** + stable `-id`
- GPU util logger on public partitions
- Staging train/val to `$SLURM_TMPDIR` for train only; infer can use scratch val
- Soft eval (`remove_small_components` 0–50) while debugging; restore ~200–500 for reporting
  once instances are large (`…_rm500.toml` vs no `_rm0` suffix when rm=0)

**Avoid:**

- `# @fork` on `train()` in `run_ppp.py` (spawn → CUDA → PreCache stall)
- One-shot `-d train predict label evaluate` on L40S
- Treating F1≈0 as “model broken” before checking eval `remove_small_components` **and** MIP

**Sanity checks before more cycles:**

1. `diagnose_ppp_basic_instances.py` (HDF label sizes vs 200/500)
2. Instance MIP under `test/instanced/.../*.png`
3. Spot-check `volumes/pred_affs` / `pred_numinst` on a compute node if needed

**Branching:** strong affs + shredded instances → VI **merge** knobs (Layer B);
weak/noisy affs → longer train / capacity. **Current branch:** fmaps24 pilot won @ 80k
(soft F1 0.170) → **ckpt sweep** to pick best.

---

## 5. Pointers

| Item | Path |
|------|------|
| Smoke chain | `sbatch/ppp/ppp_basic_8h_chain.sh` |
| Layer A runbook (A2 done; pick 70000) | `md_guides/ppp_layer_a_prediction.md` |
| Layer B runbook (closed) | `md_guides/ppp_layer_b_merge.md` |
| **Next** | `CKPTS="50000 60000 70000 80000" bash sbatch/ppp/ppp_basic_fmaps24_ckpt_sweep_chain.sh` |
| Full val @ 70000 | `sbatch sbatch/ppp/ppp_basic_long8h_full_val_sbatch.sh` |
| Full test @ 70000 | `sbatch sbatch/ppp/ppp_basic_long8h_full_test_sbatch.sh` |
| fmaps24 overlay | `PatchPerPix/.../setup01/basic_fmaps24.toml` |
| fmaps24 infer / chain | `sbatch/ppp/ppp_basic_fmaps24_{infer_sbatch,chain}.sh` |
| Longer overlay (closed) | `PatchPerPix/experiments/flylight/setups/setup01/basic_long8h.toml` |
| Soft re-eval (long8h) | `sbatch/ppp/ppp_basic_long8h_eval_rm0_sbatch.sh` |
| VI sweep chain | `sbatch/ppp/ppp_basic_long8h_vi_sweep_chain.sh` |
| VI threshold overlays | `PatchPerPix/.../setup01/vi_th_0_{3..7}.toml` |
| Infer uses best VI | `vi_th_0_3.toml` + `eval_rm0.toml` on fmaps24 / long8h infer |
| Eval rm0 overlay | `PatchPerPix/experiments/flylight/setups/setup01/eval_rm0.toml` |
| Instance diagnose | `sbatch/ppp/diagnose_ppp_basic_instances.py` |
| Smoke exp | `metrics/ppp/ppp_basic_8h/` |
| Longer exp (closed) | `metrics/ppp/ppp_basic_long8h/` |
| **fmaps24 exp** | `metrics/ppp/ppp_basic_fmaps24/` |
| Architecture | `PatchPerPix/CODEBASE.md` |

---

## 6. Glossary (quick)

- **VI (`vote_instances`):** Assembles instance IDs from predicted affinities/numinst.
- **`patch_threshold` / `fc_threshold`:** Scores used when selecting patches / covering foreground during VI.
- **`ignore_small_comps`:** Size cutoff **inside** VI.
- **`remove_small_components`:** Size cutoff **at evaluation** when scoring Pred vs GT.

---

## 7. Web viewer notes

### Two products on disk

Under `metrics/ppp/<exp>/test/` (PPP inference tree; often fed FISBe **val**):

| Product | Stage | Path | Web overlay |
|---------|--------|------|-------------|
| 1 — network outputs | `-d predict` | `processed/<ckpt>/<sample>.zarr` → `volumes/pred_affs`, `volumes/pred_numinst` | **numinst** |
| 2 — assembled IDs | `-d label` | `instanced/<ckpt>/<VI params…>/<sample>.hdf` → `vote_instances` (+ `.png` MIP) | **instances** |

Numinst is a count/FG map (not per-neuron). Instances are hard neuron IDs after vote-instances. Good numinst silhouette + shredded instances = Layer B problem.

### Numinst color choice (2026-09-22)

Keep **argmax → discrete labels `{0,1,2}`**. Use a **fixed categorical palette** (not seeded random, not a continuous colormap on soft probs):

| ID | Meaning | RGB |
|----|---------|-----|
| 0 | background | `(0, 0, 0)` |
| 1 | single instance | cyan `(0, 220, 255)` |
| 2 | overlap (2+) | magenta `(255, 40, 200)` |

Implemented in `web/server/services/ppp_loader.py` (`_encode_numinst_rgb`). True instance overlays still use seeded random colors per ID.

**Out of scope for now:** continuous heatmap on `P(2+)`; displaying `pred_affs` in the web viewer.
