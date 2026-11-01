# BiaPy watershed parameter tuning

Plan for a **post-training** system that optimizes watershed thresholds for any
trained BiaPy instance-seg setup, starting with **FDb**.

Related: [`raytune_dice_vs_skel_fdb.md`](raytune_dice_vs_skel_fdb.md)
(Stage 1 = train hparams; this doc is Stage 2 = watershed hparams).
Reuse patterns from `biapy_work_folder/raytune/` (search space, trial config,
winner export) — CPU-only trials, no GPU train.

Motivation: FDb-dice / FDb-skel RayTune winners reach decent **channel** metrics
on val, but test instance PQ collapses because watershed emits ~1 instance per
volume (`SEED_CHANNELS_THRESH: [0.05]` is far too loose for Db seeds).

---

## Locked decisions

| Choice | Value |
|--------|--------|
| Split for optimization | **val** (lock params; report final scores on **test**) |
| Objectives | **Multi-metric** — improve several instance scores together (see below); not a single scalar in isolation |
| Channel scope (v1) | **FDb only**; code structured so FDcDn / other setups plug in later |
| Search style (v1) | **Ray Tune** (Optuna searcher), CPU trials |
| Predictions | **Require** existing `per_image/` channel preds — no auto-predict |
| Inference during search | **None** (CPU watershed + matching only) |

---

## Multi-metric objectives

We care about several matching stats at once (same family as BiaPy test logs).
Every trial logs **all** of these (mean over val volumes); the tuner does not
optimize one metric while ignoring the rest.

| Metric | Why it matters |
|--------|----------------|
| **PQ @ 0.3** | Lenient instance quality (thin neurons often fail stricter IoU) |
| **PQ @ 0.5** | Stricter overlap — rewards real shape agreement |
| **F1 @ 0.3** | Detection quality (TP/FP/FN) without the matched-IoU term |
| **F1 @ 0.5** | Stricter detection |
| **precision / recall @ 0.3** | Diagnose over-seg vs under-seg |
| **`n_pred` / `n_true`** | Sanity — today `n_pred ≈ 1`; want counts near GT scale |
| **mean_matched_score @ 0.3** | Average IoU of matched pairs |

### What is PQ?

**Panoptic Quality (PQ)** comes from BiaPy’s `matching(...)`. At IoU threshold *t*:

1. Match each pred instance to at most one GT if IoU ≥ *t*.
2. Count TP / FP / FN.
3. PQ ≈ (average IoU of matches) × (recognition quality).

Higher is better. Current short_winner **test** PQ@0.3 ≈ 0.05 (dice) / 0 (skel).

### How Ray ranks trials (scalarization)

Ray needs one number to maximize per trial, but we still **report every metric**.

**v1 composite (maximize):**

```text
score = w_pq03 * PQ@0.3
      + w_pq05 * PQ@0.5
      + w_f103 * F1@0.3
      + w_f105 * F1@0.5
      − w_count * count_penalty
```

Suggested default weights (tunable in config):

| Weight | Default | Role |
|--------|---------|------|
| `w_pq03` | 1.0 | Main lenient quality |
| `w_pq05` | 0.5 | Pull toward better overlap |
| `w_f103` | 0.5 | Detection |
| `w_f105` | 0.25 | Stricter detection |
| `w_count` | 0.25 | Soft count regularizer |

`count_penalty` — e.g. `|log((n_pred+ε)/(n_true+ε))|` or
`|n_pred/n_true − 1|`, clipped so early trials with `n_pred=1` are penalized
but do not dominate once counts are reasonable.

Every trial still writes the full metric dict to Ray + CSV so we can:

- Re-rank winners by PQ@0.5 only, F1 only, etc. offline
- Inspect **Pareto-ish** tradeoffs (e.g. high PQ@0.3 but poor precision)

Optional later: Optuna multi-objective / Pareto front instead of a weighted sum.
v1 keeps a weighted composite so it fits the existing single-metric Ray trainables.

---

## Why Ray (not a CLI grid)

| | CLI grid | **Ray Tune (chosen)** |
|--|----------|------------------------|
| Search | Fixed cartesian list | Adaptive (Optuna) over continuous + discrete (`auto`) |
| Multi-metric | Manual post-hoc ranking | Built-in trial table + custom `score`; easy to add metrics |
| Growth | Painful as dims grow (FDcDn, morph) | Same Stage-2 entrypoint scales |
| Fit to repo | New one-off script | Mirrors `biapy_work_folder/raytune/` Stage 1 |
| Cost | Fine for ~60 points | Still cheap — trials are CPU watershed, not GPU train |

CLI grid remains useful as a **debug smoke** (evaluate one fixed combo), not the
main search driver.

Illustrative launch (API TBD):

```bash
python -m biapy_work_folder.watershed_tune.run_tune \
  --setup biapy-aug-zarr-seunet-FDb-dice \
  --run biapy_fdb_dice_l40s_wc_nopat_short_winner \
  --split val \
  --num-samples 40 \
  --metric-weights pq03=1,pq05=0.5,f103=0.5,f105=0.25,count=0.25
```

---

## Pipeline (Stage 2)

```
trained BiaPy setup
        │
        ▼
[0] Preconditions
    - metrics/biapy/<setup>/results/<setup>_<run>/per_image/  exists (val)
    - val GT instance labels available
        │
        ▼
[1] Ray Tune over watershed params (CPU)
    - each trial: load preds → watershed_by_channels → matching
    - report full metric dict + composite score
        │
        ▼
[2] Write artifacts
    - Ray results / CSV with all metrics
    - *_ws_winner.yaml (WATERSHED.* overrides) from best composite
    - optional: top-K by alternate metrics (PQ@0.5, F1@0.3, …)
        │
        ▼
[3] Certify
    - BiaPy test with REUSE_PREDICTIONS=true + winner thresholds
    - report full metric suite on test (do not retune)
```

---

## Search space (FDb v1)

Current configs:

```yaml
WATERSHED:
  SEED_CHANNELS: [Db]
  SEED_CHANNELS_THRESH: [0.05]      # too low → connected seeds → 1 instance
  TOPOGRAPHIC_SURFACE_CHANNEL: F
  GROWTH_MASK_CHANNELS: [F]
  GROWTH_MASK_CHANNELS_THRESH: [auto]
```

| Param | Ray space (v1) | Notes |
|-------|----------------|--------|
| `seed_db_mode` | `choice(["auto", "float"])` | Discrete branch |
| `seed_db` | `uniform(0.05, 0.9)` if mode=float | Db **≥** polarity; avoid staying stuck at 0.05 |
| `growth_f_mode` | `choice(["auto", "float"])` | |
| `growth_f` | `uniform(0.05, 0.7)` if mode=float | F **≥**; `auto` ≈ Otsu/2 in BiaPy |

Use Optuna / Ray conditional spaces (or sample float always and ignore when
mode=`auto`) so `auto` remains a first-class option.

Out of scope for v1 (hooks OK for later):

- `DATA_REMOVE_SMALL_OBJ_BEFORE`
- erode/dilate growth mask
- FDcDn multi-seed lists
- Changing which channels are seed vs growth

---

## Expandability (non-FDb)

Do **not** hardcode F/Db in the core loop. Adapter interface:

1. Read setup YAML → `DATA_CHANNELS`, `WATERSHED.SEED_CHANNELS`, `GROWTH_MASK_CHANNELS`.
2. Build search dims from channel polarity:
   - **≥** channels: `F`, `P`, `Db`, `D`
   - **≤** channels: `C`, `B`, `T`, `Dn`, `Dc`  
   (same convention as `ipynb/view_augments/metric_review.py`)
3. Map channel name → index in `per_image` volume.
4. Call `watershed_by_channels` with lists parallel to the YAML.

FDcDn v2 = new Ray search-space preset + same trainable.

---

## Implementation sketch

Detailed sequencing: [Implementation stages](#implementation-stages).

Suggested location: `biapy_work_folder/watershed_tune/`, parallel to
`biapy_work_folder/raytune/`.

| Module | Responsibility |
|--------|----------------|
| `paths.py` | Resolve `per_image/` + val GT; fail if missing |
| `load.py` | Load channel preds + GT instances |
| `run_ws.py` | Wrapper around BiaPy `watershed_by_channels` |
| `score.py` | Matching at 0.3/0.5/0.75; full metric dict + composite `score` |
| `search_space.py` | FDb (then FDcDn) Ray spaces; mode=`auto`\|`float` |
| `trainable.py` | Ray trainable: apply hparams → watershed → `tune.report(...)` |
| `run_tune.py` | Entrypoint (mirrors train `run_tune.py`, CPU resources) |
| `write_winner.py` | YAML fragment for `PROBLEM.INSTANCE_SEG.WATERSHED.*` |
| `winner.py` | Pick best trial by composite; optional re-rank by any logged metric |

Reuse:

- BiaPy: `watershed_by_channels`, `matching`
- Repo: polarity / path helpers from `ipynb/view_augments/metric_review.py`
- Repo: Ray patterns from `biapy_work_folder/raytune/` (Optuna, trial IDs, winner export)

HPC: CPU partition `sbatch` (no `--gres=gpu`). Singularity + `BiaPy_env` if
imports need the overlay; trials are lightweight relative to Stage 1 train.

---

## Preconditions checklist

For a given `--setup` / `--run`:

1. `per_image/` channel volumes exist for **val** samples.
2. Val GT instance labels under the setup’s `DATA.VAL.GT_PATH` (or mirrored path).
3. If val `per_image/` was never written: one-shot BiaPy predict on val
   **outside** this tuner (see [Stage 0](#stage-0--ensure-val-channel-predictions-exist)).
4. Tuner exits with a clear error if any required volume is missing.

FDb short_winners: use `--run …_short_winner_valpred` (not the test short_winner
run). Val preds live under
`metrics/biapy/<job>/results/<job>_<run>_valpred/per_image/`.

---

## Success criteria

1. Val `n_pred` moves away from ~1/volume toward `n_true` scale.
2. Composite **and** the individual logged metrics (PQ@0.3, PQ@0.5, F1@0.3, …)
   beat the untuned baseline (seed `0.05` + growth `auto`) on val.
3. Certified test run with winner thresholds improves the same suite vs current
   short_winner numbers (dice PQ@0.3 ≈ 0.05, skel ≈ 0).
4. Leaderboard CSV allows picking an alternate “winner” by any single metric
   without re-running the search.
5. Dice and skel setups tuned independently (separate `--run` / Ray exps).

---

## Out of scope

- Retraining the network
- Changing loss / architecture
- Tuning on the test set
- Auto-launching BiaPy predict when `per_image/` is missing
- Agglomeration / StarDist / other `INSTANCE_CREATION_PROCESS` modes (watershed only)
- Full Pareto multi-objective Optuna (v1 = weighted composite + full logs)

---

## Implementation stages

Work sequentially. Each stage has a clear exit gate before the next starts.
Package root: `biapy_work_folder/watershed_tune/`.

| Stage | Name | Depends on | Exit gate |
|-------|------|------------|-----------|
| **0** | Val `per_image/` preds | Trained checkpoint | Val channel volumes on disk |
| **1** | Core load + single watershed | Stage 0 | One volume → labels + metric dict |
| **2** | Multi-metric scorer | Stage 1 | Composite + full suite on all val vols |
| **3** | Ray trainable smoke | Stage 2 | 2–3 CPU trials report to Ray |
| **4** | FDb Ray sweep + winner export | Stage 3 | CSV + `*_ws_winner.yaml` for dice & skel |
| **5** | Certify on test | Stage 4 | Official BiaPy test metrics improve |
| **6** | Docs + expandability hooks | Stage 5 | Usage in this guide; FDcDn-ready adapter |

---

### Stage 0 — Ensure val channel predictions exist

**Goal.** Tuner requires `per_image/` for **val**. Winner jobs only wrote **test**.

**Locked layout.** Dedicated run-id `*_valpred` (do **not** overwrite test `per_image/`):

| Setup | `-j` | `-r` | Channel preds |
|-------|------|------|---------------|
| FDb-dice | `biapy-aug-zarr-seunet-FDb-dice` | `biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred` | `metrics/biapy/<j>/results/<j>_<r>/per_image/` |
| FDb-skel | `biapy-aug-zarr-seunet-FDb-skel` | `biapy_fdb_skel_l40s_wc_nopat_short_winner_valpred` | same pattern |

Configs (only change vs winner: `DATA.TEST` → val):

- `biapy_work_folder/configs/biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred.yaml`
- `biapy_work_folder/configs/biapy_fdb_skel_l40s_wc_nopat_short_winner_valpred.yaml`

```yaml
DATA.TEST.PATH: fisbe/biapy-no-aug-zarr/val/raw
DATA.TEST.GT_PATH: fisbe/biapy-no-aug-zarr/val/label
```

**One-time checkpoint hardlinks** (BiaPy loads `{job}_{run}-checkpoint-best.pth`):

```bash
# from repo root
ln \
  metrics/biapy/biapy-aug-zarr-seunet-FDb-dice/checkpoints/biapy-aug-zarr-seunet-FDb-dice_biapy_fdb_dice_l40s_wc_nopat_short_winner-checkpoint-best.pth \
  metrics/biapy/biapy-aug-zarr-seunet-FDb-dice/checkpoints/biapy-aug-zarr-seunet-FDb-dice_biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred-checkpoint-best.pth

ln \
  metrics/biapy/biapy-aug-zarr-seunet-FDb-skel/checkpoints/biapy-aug-zarr-seunet-FDb-skel_biapy_fdb_skel_l40s_wc_nopat_short_winner-checkpoint-best.pth \
  metrics/biapy/biapy-aug-zarr-seunet-FDb-skel/checkpoints/biapy-aug-zarr-seunet-FDb-skel_biapy_fdb_skel_l40s_wc_nopat_short_winner_valpred-checkpoint-best.pth
```

**Submit** (user runs; parallel OK). Reason: produce val channel `per_image/` for watershed tuning without clobbering test preds.

```bash
./sbatch/biapy/biapy-py_sbatch_chain.sh \
  -j biapy-aug-zarr-seunet-FDb-dice \
  -r biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred \
  biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred test

./sbatch/biapy/biapy-py_sbatch_chain.sh \
  -j biapy-aug-zarr-seunet-FDb-skel \
  -r biapy_fdb_skel_l40s_wc_nopat_short_winner_valpred \
  biapy_fdb_skel_l40s_wc_nopat_short_winner_valpred test
```

Expect 5 val stems under each `…_valpred/per_image/` (`R22C03-…`, `VT012403-…`, `VT033614-…_H1`, `VT033614-…_H5`, `VT041298-…`), matching `fisbe/biapy-no-aug-zarr/val/label/`. Stage 1+ must use `--run …_valpred`, not the test short_winner run.

Do **not** auto-wire this into the tuner.

**Exit gate.** For both dice and skel: every val volume has a channel pred file; GT paths resolve; test short_winner `per_image/` still has 7 test stems.

**Blocked if.** Checkpoint or val GT missing — fix data/setup first.

**Completed (2026-09-18).** Exit gate **passed**.

| Setup | Job | Result |
|-------|-----|--------|
| FDb-dice valpred | `17951311` | `COMPLETED` `0:0` (~16m); 5 val stems in `…_short_winner_valpred/per_image/` |
| FDb-skel valpred | `17951313` | `COMPLETED` `0:0` (~17m); 5 val stems in `…_short_winner_valpred/per_image/` |

Val stems match `fisbe/biapy-no-aug-zarr/val/label/`; test short_winner `per_image/` still has 7 test stems (untouched). Stage 1+ must use `--run …_valpred`, not the test short_winner run.

---

### Stage 1 — Paths, load, single watershed dry run

**Goal.** Prove we can go from disk → `watershed_by_channels` → instance labels without Ray.

**Deliverables.**

| File | Role |
|------|------|
| `paths.py` | Resolve setup/run → `per_image/` + GT dirs; error if missing |
| `load.py` | Load channel volume (ZYXC) + GT instances (ZYX); channel-name → index |
| `run_ws.py` | Call BiaPy `watershed_by_channels` with seed/growth thresh lists |
| CLI smoke | e.g. `python -m ...dry_run --setup ... --run ... --sample <id> --seed-th 0.4 --growth-th auto` |

**Work.**

1. Reuse polarity / stem helpers from `ipynb/view_augments/metric_review.py` (import or factor shared util).
2. Hard-fail with a clear message if `per_image/` or GT is absent (Stage 0 contract).
3. Dry-run baseline (`seed=0.05`, `growth=auto`) and one stricter seed (e.g. `0.4`) on a single val volume; print `n_pred`, `n_true`.

**Exit gate.** Dry run prints labels shape + `n_pred`/`n_true` for baseline and alternate thresh; baseline still shows under-seg (`n_pred≈1`) as a known reference.

**Completed (2026-09-18).** Package at `biapy_work_folder/watershed_tune/` (`paths`, `load`, `run_ws`, `dry_run`). Smoke job `17957471` (`cpu_short`, 64g):

```bash
python biapy_work_folder/watershed_tune/dry_run.py \
  --setup biapy-aug-zarr-seunet-FDb-dice \
  --run biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred \
  --compare-baseline
```

On `R22C03-20180918_66_J2`: `pred.shape=(399,683,686,2)`, `n_true=2`; baseline seed `0.05` and stricter `0.4` both gave `n_pred=1` (under-seg persists at 0.4 on this volume — Stage 2+ should search higher seeds / growth).

---

### Stage 2 — Multi-metric scoring

**Goal.** Every evaluation returns the full metric suite + composite `score`.

**Deliverables.**

| File | Role |
|------|------|
| `score.py` | `matching` at 0.3/0.5/0.75; aggregate means over a sample list |
| `metrics` schema | Fixed key names for Ray/CSV (`pq_0.3`, `f1_0.3`, `n_pred`, `score`, …) |
| Weight config | Defaults from [Multi-metric objectives](#multi-metric-objectives); CLI overrides |

**Work.**

1. Per volume: matching stats at 0.3/0.5/0.75.
2. Aggregate: mean PQ/F1/precision/recall/mean_matched; sum or mean `n_pred`/`n_true`.
3. `count_penalty` + weighted composite `score` (maximize).
4. Evaluate **all val volumes** for baseline vs one hand-picked thresh; print side-by-side suite.

**Exit gate.** Dict with all keys present; composite ranks stricter seed above `0.05` baseline on val (or document if not — then debug watershed before Ray).

**Completed (2026-09-18).** `score.py` + `eval_val.py` in `biapy_work_folder/watershed_tune/`. Job `17958383` (`cpu_short`, ~12m) on FDb-dice valpred (5 volumes):

| | baseline (`seed=0.05`, `growth=auto`) | strict (`seed=0.7`, `growth=auto`) |
|--|--|--|
| `score` | **−0.302** | −0.761 |
| `pq_0.3` | **0.023** | 0.0 |
| `f1_0.3` | **0.067** | 0.0 |
| `n_pred` / `n_true` | 5 / 21 | 1 / 21 |

`EXIT_GATE: DOCUMENT` — stricter seed **does not** beat baseline; `0.7` collapses seeds (`n_pred` 5→1) and zeros PQ/F1. Do **not** start Ray (Stage 3) until seed/growth search brackets are revisited (e.g. mid-range seeds, growth floats, or per-volume Db histograms).

---

### Stage 3 — Ray trainable smoke (CPU)

**Goal.** Wire Stage 1–2 into Ray like `biapy_work_folder/raytune/`, without a full sweep.

**Deliverables.**

| File | Role |
|------|------|
| `search_space.py` | FDb: `seed_db_mode` / `seed_db` / `growth_f_mode` / `growth_f` |
| `trainable.py` | Load once (or per trial); apply hparams → watershed all val → `tune.report` |
| `run_tune.py` | Entrypoint: Optuna searcher, `num_samples` small, CPU resources |
| Optional | `sbatch/.../watershed_tune_sbatch.sh` (CPU partition, no GPU) |

**Work.**

1. Mirror Stage-1 Ray patterns (`run_tune`, Optuna, trial naming) but **no** GPU / no BiaPy train.
2. Each `tune.report(...)` includes **full metric dict + `score`** (composite is what Optuna maximizes).
3. Smoke: `--num-samples 3` (or 2), local or one CPU sbatch; confirm results dir + trial metrics.

**Exit gate.** Three finished trials with logged `pq_0.3`, `pq_0.5`, `f1_*`, `n_pred`, `score`; no GPU allocation.

---

### Stage 4 — FDb production sweep + winner export

**Goal.** Real search on dice and skel short_winners; export YAML + leaderboard.

**Deliverables.**

| Artifact | Role |
|----------|------|
| Ray experiment dirs | One exp per setup/run |
| `leaderboard.csv` | All trials, all metrics (re-rank offline) |
| `*_ws_winner.yaml` | Only `PROBLEM.INSTANCE_SEG.WATERSHED.*` overrides |
| `write_winner.py` / `winner.py` | Best by `score`; CLI to re-pick by `pq_0.5` etc. |

**Work.**

1. Sweep dice winner and skel winner separately (`--num-samples` ~30–50 to start).
2. Include baseline as a fixed reference trial or post-hoc eval row.
3. Export winner YAML mergeable onto the train/test config.
4. Sanity-check: winner `n_pred` not stuck at ~1/volume; several metrics beat baseline.

**Exit gate.** Two winner YAMLs + CSVs; composite and at least PQ@0.3 / F1@0.3 improved on **val** vs baseline.

---

### Stage 5 — Certify on test (BiaPy official path)

**Goal.** Same thresholds through BiaPy’s real test + matching (no silent drift).

**Work.**

1. Merge `*_ws_winner.yaml` into test config.
2. `TEST.REUSE_PREDICTIONS: true` if test `per_image/` already exists; else one predict+watershed pass.
3. Run official test for dice and skel winners.
4. Compare full suite to pre-tune short_winner test numbers (PQ@0.3 ≈ 0.05 / 0).

**Exit gate.** Test logs / `test_results_metrics.csv` show multi-metric gains; `n_pred` no longer ≈1 per volume. If val improved but test did not, stop and analyze (domain shift / overfit thresh) before expanding scope.

---

### Stage 6 — Usage docs + expandability

**Goal.** Make Stage 2–5 reproducible; leave FDcDn as a preset, not a rewrite.

**Work.**

1. Add **Usage** section to this guide (exact commands, paths, weight flags).
2. Confirm adapter reads seed/growth channel lists from YAML (no FDb-only hardcoding in `run_ws` / `trainable`).
3. Stub or document FDcDn `search_space` preset (unimplemented OK).
4. Optional: note how to switch to true Optuna multi-objective later.

**Exit gate.** Someone else can tune a new FDb run from this doc alone; adding FDcDn is “new search space + Stage 0 preds,” not a new pipeline.

---

### Stage dependency diagram

```text
[0] val per_image preds
        │
        ▼
[1] paths / load / dry_run watershed
        │
        ▼
[2] multi-metric score + composite
        │
        ▼
[3] Ray smoke (2–3 CPU trials)
        │
        ▼
[4] FDb Ray sweep → CSV + ws_winner.yaml  (dice & skel)
        │
        ▼
[5] BiaPy test certify (REUSE_PREDICTIONS)
        │
        ▼
[6] Usage docs + non-FDb adapter hooks
```

---

### Parallelism / what not to parallelize

- **Do in parallel:** Stage 0 predict jobs for dice and skel; Stage 4 sweeps for dice and skel once Stage 3 is green.
- **Do not skip:** Stage 2 before Ray (bad composite = meaningless search); Stage 5 before declaring winners “done.”
- **Do not** tune on test (Stage 5 is evaluate-only).
