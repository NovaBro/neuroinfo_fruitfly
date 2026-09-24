# Ray Tune: Dice vs Skeleton Recall (FDb, L40S)

Plan for a fair hyperparameter comparison between **pure Dice on F** and
**Skeleton Recall aux loss**, both under the same training regime.

Related: [`raytune_biapy_scheduler_only.md`](raytune_biapy_scheduler_only.md),
[`instance_seg_dice_loss.md`](instance_seg_dice_loss.md),
[`skeleton_recall_loss.md`](skeleton_recall_loss.md).

---

## Locked decisions

| Choice | Value |
|--------|--------|
| Channel setup | **FDb only** (`biapy-aug-zarr-seunet-FDb-dice` vs `…-FDb-skel`) |
| Scheduler | **warmupcosine** (fixed; not searched) |
| Early stopping | **no patience** (`--search-patience -1`) |
| GPU | **L40S** (`--gpu l40s` → constraint `l40s`, mem `250g`) |
| Batch size | **fixed 12** (not in search space; L40S bs=15 OOMed before) |
| Sweep wall clock | **whole Optuna sweep &lt; 3 hours** (not per-trial) |
| Loss-weight meaning | see [Weight knobs](#weight-knobs-what-we-tune) below |

Final performance comparison uses **full-epoch winner retrains + test**, not
short-tune `val_loss` alone. Short sweeps only rank hparams.

---

## Weight knobs (what we tune)

There is **no** YAML key named `dice_weight`. Two different scalars matter:

### Channel weights (both bases)

```yaml
PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS: (w_F, w_Db)
```

Scales each head’s loss: `w_F * L_F + w_Db * L_Db`.

- **Dice base:** `L_F` is soft Dice (`DATA_CHANNELS_LOSSES: ['dice', 'l1']`).
- **Skel base:** `L_F` is the default FG loss (typically BCE) unless also set to dice.

Ray hparam key: `channel_weights` → written by `write_trial_config.apply_hparams`.

### Skeleton-recall weight (skel base only)

```yaml
LOSS.SKELETON_RECALL.WEIGHT: skel_weight
```

Adds `skel_weight * SoftSkeletonRecall(F, Sk)` on top of the channel losses.
No dice counterpart exists (Stage B `LOSS.INSTANCE_DICE.W_*` is not required for
this experiment).

### Recommendation (clean comparison)

| Sweep | Optuna params | Fixed |
|-------|---------------|--------|
| **FDb-dice** | `lr`, `warmup_cosine_epochs` (± optional `min_lr`), **F channel weight** | `scheduler=warmupcosine`, `batch_size=12`, Db weight fixed (e.g. `2.0` to match current dice YAML, or `1.0`) |
| **FDb-skel** | `lr`, `warmup_cosine_epochs` (± optional `min_lr`), **`skel_weight`** | `scheduler=warmupcosine`, `batch_size=12`, channel weights left at base `(1.0, 1.0)` |

Do **not** mix dice and skel into one Tuner — two separate `--exp` names.

---

## Code changes needed before launch

Current `build_search_space` (`biapy_work_folder/raytune/search_space.py`) samples
`batch_size`, `scheduler ∈ {warmupcosine, onecycle}`, and `skel_weight` (skel
only). It does **not** match this plan. Update it to something like:

```python
# L40S branch (illustrative)
space = {
    "lr": tune.loguniform(1e-4, 1e-3),
    "scheduler": "warmupcosine",  # fixed, not tune.choice
    "warmup_cosine_epochs": tune.randint(3, search_epochs),  # must be ≤ EPOCHS
    # optional: "min_lr": tune.loguniform(1e-6, 1e-4),
}
# dice: include F-weight via channel_weights helper, e.g. (w_F, 2.0)
# skel: include skel_weight = tune.uniform(0.05, 0.3)
# omit batch_size from space; set 12 in base YAML or inject as fixed hparam
```

Also ensure:

1. Dice base ends up on warmupcosine (YAML still defaults to `onecycle`; space or
   base must force `NAME: warmupcosine` + `MIN_LR` / `WARMUP_COSINE_DECAY_EPOCHS`).
2. `warmup_cosine_epochs` upper bound **≤ `--search-epochs`**.
3. `--search-patience -1` so trial YAMLs do not set `TRAIN.PATIENCE`.

Optional: pin `batch_size: 12` in both base YAMLs (already true for FDb-dice /
FDb-skel L40S defaults).

---

## Suggested short-sweep budget (&lt; 3 h wall)

L40S FDb-skel history: ~**1 min/epoch** train + ~2–3 min staging after data is hot.

| Flag | Suggested |
|------|-----------|
| `--search-epochs` | **12** |
| `--search-patience` | **-1** |
| `--num-samples` | **8** |
| `--max-concurrent` | **3** (hard cap per controller; do not raise) |
| `--gpu` | **l40s** |
| controller `--time` | **6:00:00** (override SBATCH default; enough for &lt;3 h sweep) |

Rough wall: `(8 / 3) * ~15–18 min ≈ 45–70 min` if L40S queue is free — comfortably
under 3 h. Raise epochs only if you need a longer warmup range.

Controller job walltime can stay long (`raytune_controller_sbatch.sh` default);
the 3 h target is **nested GPU sweep** wall clock.

---

## Launch (after search_space patch)

From repo root. Use **new** `--exp` names (dirs under `metrics/biapy/raytune/<exp>/`
must be empty unless `--resume`). Pass **`--space wc_nopat`** so Optuna uses the
warmupcosine / no-batch / weight space (default `--space legacy` keeps older
sweeps stable).

```bash
# Optional: dry-run first (no GPU) — 6h controller walltime
sbatch --time=6:00:00 sbatch/biapy/raytune_controller_sbatch.sh \
  --base biapy-aug-zarr-seunet-FDb-dice \
  --exp biapy_fdb_dice_l40s_wc_nopat_short_dry \
  --search optuna --gpu l40s --space wc_nopat \
  --num-samples 4 --dry-run \
  --search-epochs 12 --search-patience -1

# Dice sweep (max 3 nested L40S trains at once)
sbatch --time=6:00:00 sbatch/biapy/raytune_controller_sbatch.sh \
  --base biapy-aug-zarr-seunet-FDb-dice \
  --exp biapy_fdb_dice_l40s_wc_nopat_short \
  --search optuna --gpu l40s --space wc_nopat \
  --num-samples 8 --max-concurrent 3 \
  --search-epochs 12 --search-patience -1

# Skeleton-recall sweep (max 3 nested L40S trains at once)
sbatch --time=6:00:00 sbatch/biapy/raytune_controller_sbatch.sh \
  --base biapy-aug-zarr-seunet-FDb-skel \
  --exp biapy_fdb_skel_l40s_wc_nopat_short \
  --search optuna --gpu l40s --space wc_nopat \
  --num-samples 8 --max-concurrent 3 \
  --search-epochs 12 --search-patience -1
```

Queue hygiene: controller refuses to start if too many `*-train` jobs are already
in `squeue` (`MAX_USER_TRAINS`, default 4). Wait for an in-flight FDcDn/L40S sweep
or raise the env var.

---

## After Tune: fair comparison

Each sweep writes under `metrics/biapy/raytune/<exp>/`:

- `results_table.csv` — trial ranking
- `best_config.yaml` — winning hparams
- `winner_pointer.yaml` — path + suggested chain command
- `biapy_work_folder/configs/trials/<exp>_winner.yaml` — **full** base epochs (no search cap)

Retrain + test winners:

```bash
./sbatch/biapy/biapy-py_sbatch_chain.sh -r <exp>_winner <stem> train test
```

Compare on the same test metrics (instance IoU / F1 / whatever the eval path
already reports). Do not declare a winner from short-sweep `val_loss` alone —
Dice and Skeleton Recall scale the loss differently.

---

## Prerequisites checklist

- [ ] Patch `search_space.py` (and dice base scheduler if not forced by space)
- [ ] `env/BiaPy_env.ext3` has Ray + Optuna; dice Stage A wired (`BiaPy-novabro`)
- [ ] Channel GT caches present:
  - dice: `…/label_F.…_Db.…` (no Sk/I)
  - skel: `…/label_F.…_Db.…_Sk.dilation-1_I`
- [ ] New `--exp` names; queue not over `MAX_USER_TRAINS`
- [ ] Dry-run samples look sane (`scheduler`, warmup ≤ epochs, weights)

---

## Implementation jobs (ordered)

Do these in order. Each job has a **Done when** gate — do not start the next
job until the previous gate passes. Jobs 0–4 are local/code; 5–8 are HPC.

```text
Job 0  Preconditions / inventory
Job 1  Patch search_space.py (+ run_tune wiring)
Job 2  Align FDb-dice / FDb-skel base YAMLs
Job 3  Unit-check write_trial_config outputs
Job 4  Optuna dry-run (no GPU)
Job 5  Launch FDb-dice L40S short sweep
Job 6  Launch FDb-skel L40S short sweep
Job 7  Winner full-epoch retrain + test (both)
Job 8  Compare metrics + note results in this guide
```

### Job 0 — Preconditions / inventory

**Goal:** Confirm env, data, and queue are ready so later jobs do not fail mid-sweep.

**Steps:**

1. Confirm Ray + Optuna in overlay:
   ```bash
   # inside BiaPy_env (container) — or read prior Stage 5 install log
   python -c 'import ray, optuna; from ray.tune.search.optuna import OptunaSearch; print(ray.__version__, optuna.__version__)'
   ```
2. Confirm dice Stage A is live (`BiaPy-novabro` editable; `DATA_CHANNELS_LOSSES: dice` already trained once if possible).
3. Confirm channel GT dirs exist under `fisbe/biapy-no-aug-zarr/`:
   - dice: `train/label_F.…_Db.…` (no `_Sk`)
   - skel: `train/label_F.…_Db.…_Sk.dilation-1_I` (and matching `val/`)
4. `squeue -u $USER` — note in-flight `*-train` / raytune controllers; wait or plan `MAX_USER_TRAINS` so Jobs 5–6 can start.
5. Pick empty `--exp` names (default suggestion below).

**Done when:** Overlay imports Ray/Optuna; both channel caches exist; you know whether the queue is clear enough for two sweeps.

---

### Job 1 — Patch `search_space.py` (+ `run_tune` wiring)

**Goal:** Optuna space matches this plan (not the old lr/batch/scheduler/skel-only space).

**Files:**

- `biapy_work_folder/raytune/search_space.py` (primary)
- `biapy_work_folder/raytune/run_tune.py` (pass flags / search_epochs into space builder)
- Optionally `write_trial_config.py` only if a new hparam key is added (prefer reuse `channel_weights`)

**Steps:**

1. For L40S: keep `lr = tune.loguniform(1e-4, 1e-3)`; **remove** `batch_size` from the space (fixed 12 via base YAML).
2. Set `scheduler` to the constant `"warmupcosine"` (not `tune.choice`).
3. Add `warmup_cosine_epochs` as an integer sampler with **hi ≤ search_epochs** (e.g. `tune.randint(3, search_epochs)`). Thread `search_epochs` into `build_search_space(...)`.
4. Optional: `min_lr` loguniform; if omitted, rely on base YAML `MIN_LR: 1e-5`.
5. Skel branch (`include_skel_weight=True`): keep `skel_weight = tune.uniform(0.05, 0.3)`.
6. Dice branch: add F-weight sampling that becomes `channel_weights`, e.g.
   ```python
   # Db fixed at 2.0 to match current FDb-dice YAML
   "channel_weights": tune.sample_from(
       lambda _: [float(tune.uniform(0.5, 2.0).sample()), 2.0]
   )
   ```
   Prefer a cleaner API: sample `f_weight` then map to `channel_weights` in `apply_hparams` / trainable — only if `f_weight` is added to `HPARAM_MAP` or converted before write. Simplest path: `tune.sample_from` returning a 2-list under key `channel_weights`.
7. Auto-detect dice vs skel in `run_tune.py` (already has `base_uses_skeleton_recall`); add a parallel `include_f_channel_weight` (or invert: include channel F weight when skel is off / when base has `dice` in `DATA_CHANNELS_LOSSES`).
8. Update the Stage-6 blurb in `raytune_biapy_scheduler_only.md` only if you want docs in sync (optional).

**Done when:** `build_search_space(gpu="l40s", include_skel_weight=False, search_epochs=12)` returns lr + warmupcosine + warmup_cosine_epochs + channel_weights; skel=True returns skel_weight and **no** channel_weights search (or leaves channel weights at base). No `batch_size` / `onecycle` in either space.

---

### Job 2 — Align FDb base YAMLs

**Goal:** Bases match fixed regime so trial overlays only change searched knobs.

**Files:**

- `biapy_work_folder/configs/biapy-aug-zarr-seunet-FDb-dice.yaml`
- `biapy_work_folder/configs/biapy-aug-zarr-seunet-FDb-skel.yaml` (sanity only)

**Steps:**

1. **FDb-dice:** switch `LR_SCHEDULER` from `onecycle` to `warmupcosine` with `MIN_LR` and a placeholder `WARMUP_COSINE_DECAY_EPOCHS` (overwritten by search). Keep `BATCH_SIZE: 12`. Leave `PATIENCE` commented out.
2. **FDb-skel:** confirm already warmupcosine + `BATCH_SIZE: 12` + no patience; leave winner LR/skel_weight as defaults (trials overwrite).
3. Do **not** change `DATA_CHANNELS` / skel ENABLE (avoids re-preprocessing).

**Done when:** Both bases have `BATCH_SIZE: 12`, warmupcosine-capable scheduler blocks, no `PATIENCE`, and dice still has `DATA_CHANNELS_LOSSES: ['dice', 'l1']`.

---

### Job 3 — Unit-check trial YAML writer

**Goal:** Prove hparams land in the right nested keys without submitting GPU jobs.

**Steps:**

```bash
# From repo root — skel-like
python biapy_work_folder/raytune/write_trial_config.py \
  --base biapy-aug-zarr-seunet-FDb-skel \
  --trial-id smoke_skel_wc \
  --lr 5e-4 --scheduler warmupcosine --warmup-cosine-epochs 8 \
  --skel-weight 0.15 \
  --search-epochs 12 --search-patience -1 \
  --dry-run

# Dice-like (channel weights as JSON list)
python biapy_work_folder/raytune/write_trial_config.py \
  --base biapy-aug-zarr-seunet-FDb-dice \
  --trial-id smoke_dice_wc \
  --lr 5e-4 --scheduler warmupcosine --warmup-cosine-epochs 8 \
  --channel-weights '[1.2, 2.0]' \
  --search-epochs 12 --search-patience -1 \
  --dry-run
```

**Check printed YAML for:** `EPOCHS: 12`, no `PATIENCE` (or absent), `NAME: warmupcosine`, `WARMUP_COSINE_DECAY_EPOCHS: 8`, `BATCH_SIZE: 12`, skel `WEIGHT` / dice `DATA_CHANNEL_WEIGHTS` as expected. Dice must **not** grow `LOSS.SKELETON_RECALL`.

**Done when:** Both dry-run dumps look correct; unknown-key errors are fixed.

---

### Job 4 — Optuna dry-run (controller, no nested GPU)

**Goal:** End-to-end Ray sampling without burning L40S.

```bash
sbatch --time=6:00:00 sbatch/biapy/raytune_controller_sbatch.sh \
  --base biapy-aug-zarr-seunet-FDb-dice \
  --exp biapy_fdb_dice_l40s_wc_nopat_short_dry \
  --search optuna --gpu l40s --space wc_nopat \
  --num-samples 4 --dry-run \
  --search-epochs 12 --search-patience -1

sbatch --time=6:00:00 sbatch/biapy/raytune_controller_sbatch.sh \
  --base biapy-aug-zarr-seunet-FDb-skel \
  --exp biapy_fdb_skel_l40s_wc_nopat_short_dry \
  --search optuna --gpu l40s --space wc_nopat \
  --num-samples 4 --dry-run \
  --search-epochs 12 --search-patience -1
```

**Done when:** Controller logs show 4 sampled configs each; dice samples include `channel_weights` and no `skel_weight`; skel samples include `skel_weight`; all have `scheduler=warmupcosine` and `warmup_cosine_epochs ≤ 12`.

---

### Job 5 — Launch FDb-dice L40S short sweep

**Goal:** Real nested GPU Optuna for dice (&lt; 3 h whole sweep).

```bash
# From repo root; wait until Job 0 queue check allows it
# Cap: --max-concurrent 3 (do not raise); controller walltime 6h
sbatch --time=6:00:00 sbatch/biapy/raytune_controller_sbatch.sh \
  --base biapy-aug-zarr-seunet-FDb-dice \
  --exp biapy_fdb_dice_l40s_wc_nopat_short \
  --search optuna --gpu l40s --space wc_nopat \
  --num-samples 8 --max-concurrent 3 \
  --search-epochs 12 --search-patience -1
```

Monitor: `squeue -u $USER`, controller `.out`, nested `…-rrt_*-train-*.out`.

**Done when:** `metrics/biapy/raytune/biapy_fdb_dice_l40s_wc_nopat_short/` has `results_table.csv`, `best_config.yaml`, `winner_pointer.yaml`, and a winner trial YAML under `configs/trials/`.

---

### Job 6 — Launch FDb-skel L40S short sweep

**Goal:** Same budget for skeleton recall. Can run **after** or **alongside** Job 5 if `MAX_USER_TRAINS` and L40S slots allow (prefer sequential if queue is tight).

```bash
# Cap: --max-concurrent 3 (do not raise); controller walltime 6h
sbatch --time=6:00:00 sbatch/biapy/raytune_controller_sbatch.sh \
  --base biapy-aug-zarr-seunet-FDb-skel \
  --exp biapy_fdb_skel_l40s_wc_nopat_short \
  --search optuna --gpu l40s --space wc_nopat \
  --num-samples 8 --max-concurrent 3 \
  --search-epochs 12 --search-patience -1
```

**Done when:** Same artifacts as Job 5 under `…/biapy_fdb_skel_l40s_wc_nopat_short/`.

---

### Job 7 — Winner full-epoch retrain + test

**Goal:** Fair test-set comparison at full base `EPOCHS` (not search_epochs=12).

**Steps:**

1. Read each `winner_pointer.yaml` → `chain_command` / winner config stem.
2. Submit both chains (L40S via chain script / sbatch overrides as you usually do):
   ```bash
   ./sbatch/biapy/biapy-py_sbatch_chain.sh -r <dice_exp>_winner biapy-aug-zarr-seunet-FDb-dice train test
   ./sbatch/biapy/biapy-py_sbatch_chain.sh -r <skel_exp>_winner biapy-aug-zarr-seunet-FDb-skel train test
   ```
3. Confirm winner YAMLs did **not** inherit search epoch/patience caps (`search_epochs=None` in `write_winner_config`).

**Done when:** Both train and test jobs finish; checkpoints + test metrics exist under `metrics/biapy/` for both run ids.

---

### Job 8 — Compare and record

**Goal:** Document the outcome next to this plan.

**Steps:**

1. Pull the same test metrics for both winners (instance metrics from BiaPy test logs / eval helpers you already use).
2. Note short-sweep best `val_loss` only as hparam context — **not** as the method ranking.
3. Append a short **Results** subsection to this file (date, exp names, winning hparams, test metrics, any OOMs/retries).

**Done when:** This guide has a Results blurb; you can answer “dice vs skel on FDb under warmupcosine / no-patience / L40S” with test numbers.

---

### Progress tracker

| Job | Status | Notes |
|----:|--------|-------|
| 0 Preconditions | **Done** | Caches + overlay OK; FDcDn L40S controller may still hold queue |
| 1 search_space + run_tune | **Done** | `--space wc_nopat` preset; `f_weight` → channel_weights |
| 2 Base YAMLs | **Done** | FDb-dice → warmupcosine; batch 12 |
| 3 write_trial_config smoke | **Done** | EPOCHS=12, no PATIENCE, weights OK |
| 4 Optuna dry-run | Pending | |
| 5 Dice L40S sweep | Pending | needs `--space wc_nopat` |
| 6 Skel L40S sweep | Pending | needs `--space wc_nopat` |
| 7 Winner train+test | Pending | |
| 8 Compare + Results | Pending | |

---

## Out of scope for this plan

- FDcDn variants
- Searching `batch_size` or `onecycle`
- Stage B `bce_dice` / `LOSS.INSTANCE_DICE.*`
- Joint dice+skel loss in one model (orthogonal combo is possible later; not this A/B)
