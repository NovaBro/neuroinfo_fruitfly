> **Historical / archived.** Current PatchPerPix ops and canonical presets: [`ppp_overview.md`](../ppp_overview.md). Keep this file for background only; do not treat recipes here as the live path.
>
> ---

# PatchPerPix: unify val/test list mode + multi-dataset + log layout

Plan to remove the val-vs-test launcher mismatch, treat `completely` /
`partly` / `synthetic` the same way via dataset lists, then move per-sample
logs under `base/log/<dataset>/<sample>/`.

**Status:** Phases 1–3 implemented.
**Phase 2 policy:** synthetics remain **test-only** until `fisbe/synthetic/{train,val}/` exist.

Companion docs:

- [`ppp_config_defaults.md`](ppp_config_defaults.md) — `[data]` list mode, `*_datasets`
- [`ppp_sbatch_consolidation.md`](ppp_sbatch_consolidation.md) — `ppp_submit.sh` / `ppp_job.sh`
- [`PatchPerPix/CODEBASE.md`](../../../PatchPerPix/CODEBASE.md) — pipeline architecture

Submit from the **repo root**.

---

## 1. Problem

PPP itself treats val and test the same: `train_datasets` / `val_datasets` /
`test_datasets` under `fisbe_root`, sample IDs `{dataset}/{stem}`, outputs under
`metrics/ppp/<exp>/{val,test}/…`.

The **sbatch launcher** does not. Today:

| | Val jobs | Test jobs |
|--|----------|-----------|
| `--test-data` | Passed (`fisbe/completely/val`) | Omitted |
| List mode | Off (CLI override) | On |
| `--sample` | Flat stem (`R22C03-…`) | Namespaced (`completely/R22C03-…`) |

That came from history: everything used one folder + flat names; synthetics were
added only under `fisbe/synthetic/test/`, so **test** switched to list mode while
**val** kept the old recipe. It is not a PPP requirement.

Side effects:

- Per-sample logs land at `base/<sample>/run.log` (flat) or
  `base/completely/<sample>/run.log` (nested) — next to real `train/` / `val/` /
  `test/` dirs, not under a dedicated log tree.
- Opting into `partly` / `synthetic` on val is awkward; train staging still
  assumes a single `completely` folder.

**Goal:** one launch recipe for val and test; explicit namespaced samples only;
then consistent `base/log/<dataset>/<sample>/` logging. Support train and infer
on `partly` and `synthetic` the same way as `completely`.

---

## 2. Contract (target)

### Sample IDs

- Always `{dataset}/{stem}` when `--sample` is set (e.g. `completely/R22C03-20180918_66_J2`).
- **No flat stems.** No silent `completely/` prefix. Flat → hard error in
  `ppp_job.sh` and `run_ppp.py`.
- No `--sample` → full split (or train); OK.

### Infer / array jobs (val and test)

- Omit `--test-data` so TOML `val_datasets` / `test_datasets` stay active.
- Array tasks index `DATASETS × SPLIT` → already `{dataset}/{stem}`.
- `--datasets` must match the TOML list for that split (sbatch does not parse TOML).

### Experiment layout

```text
metrics/ppp/<exp_id>/
  config.toml
  run.log                          # train / no --sample only
  log/
    <dataset>/
      <stem>/
        run.log                    # per-sample infer (after Phase 3)
  train/
  val/{processed,instanced,evaluated}/…
  test/{processed,instanced,evaluated}/…
```

Predictions stay namespaced, e.g.:

`val/processed/<ckpt>/completely/<stem>.zarr`

### Namespaced infer/eval contract (legacy flat dropped)

Sample ID is always `{dataset}/{stem}`. On-disk outputs:

```text
{val|test}/processed/<ckpt>/<dataset>/<stem>.zarr
{val|test}/instanced/<ckpt>/<vi_params>/<dataset>/<stem>.hdf
{val|test}/evaluated/<ckpt>/<vi_params>/<dataset>/<stem>_vote_instances_*.toml
{val|test}/evaluated/<ckpt>/<vi_params>/summary.csv   # sample = dataset/stem
```

No flat `*.zarr` / `*.hdf` / `*_vote_*.toml` siblings next to `<dataset>/`.
`run_ppp.py` errors if both coexist. Infer/eval require list mode
(`fisbe_root` + `train_datasets`); `--test-data` / `--val-data` are rejected
for predict/label/evaluate.

**Purge flat leftovers** (keeps nested `{dataset}/` trees): deletes
`*.zarr` / `*.hdf` / `*_vote_instances*.toml` whose parent is not
`completely|partly|synthetic` (pure-flat leaves and flats beside a dataset dir).

```bash
./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp ppp_basic_fmaps24_400k --purge-flat-outputs --dry-run
./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp ppp_basic_fmaps24_400k --purge-flat-outputs --execute
```

Then re-label if nested instanced is missing, and re-evaluate with enough
mem/time (full test eval previously OOM'd at 64G / 1h):

```bash
./sbatch/ppp/ppp_submit.sh array <preset> --stage label --split test --ckpt N
./sbatch/ppp/ppp_submit.sh eval  <preset> --split test --ckpt N --mem 128g --time 4:00:00
```

**Web Aggregate** partitions into one table per `(FISBe split × dataset)`,
e.g. `test/completely`, `val/completely`, `test/synthetic` (row labels =
stem only).

---

## 3. Current data layout

| Dataset | train | val | test | Notes |
|---------|-------|-----|------|--------|
| `completely` | ✓ | ✓ | ✓ | Default everywhere |
| `partly` | ✓ | ✓ | ✓ | Name containing `partly` auto-sets `add_partly` / `add_partly_val` |
| `synthetic` | ✗ | ✗ | ✓ (brains) | Test-only until train/val splits exist |

Prep all folders the same way:

```bash
python PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py \
  --fisbe-root fisbe --opening-radius 1
```

---

## 4. Phased plan

Do **not** change logging first. Unify the launcher, then multi-dataset train,
then log paths — otherwise flat val samples fight a hard namespaced-log rule.

```text
Phase 1  Unify val = test launcher (list mode, namespaced sample)
Phase 2  Multi-dataset completely / partly / synthetic (data + config + train staging)
Phase 3  base/log/<dataset>/<sample>/run.log
```

---

### Phase 1 — Val jobs use test job format

**Status:** done.

**Outcome:** same recipe for `SPLIT=val` and `SPLIT=test`.

| File | Change |
|------|--------|
| [`PatchPerPix/experiments/run_ppp.py`](../../../PatchPerPix/experiments/run_ppp.py) | `--split val\|test` retargets `test_data` for list-mode discovery + output folder **without** `_cli_test_data`; rejects flat `--sample` |
| [`sbatch/ppp/ppp_job.sh`](../../../sbatch/ppp/ppp_job.sh) | List mode for both val and test; no SAMPLE strip; omit `--test-data`; pass `--split`; namespaced default sample |
| [`sbatch/ppp/ppp_submit.sh`](../../../sbatch/ppp/ppp_submit.sh) | Default `--sample completely/R22C03-…`; reject flat `--sample` |

**Smoke**

```bash
./sbatch/ppp/ppp_submit.sh array <preset> --split val  --ckpt <N> …
./sbatch/ppp/ppp_submit.sh array <preset> --split test --ckpt <N> …
```

Check:

- `--sample` is `completely/…` for both
- Zarrs under `{val,test}/processed/<ckpt>/completely/<stem>.zarr`
- Argv has `--split val|test` and no `--test-data`

**Out of Phase 1:** log path change; multi-dataset train staging.

---

### Phase 2 — Partly / synthetic like completely

**Status:** done (synthetic **test-only** by policy).

Defaults stay `completely`-only; opt in via overlays + `--datasets`.

**2a. Data**

| Dataset | Action |
|---------|--------|
| `partly` | Prep PPP arrays, then build stage tars |
| `synthetic` | Infer/test only (`fisbe/synthetic/test/`); no train/val yet |

```bash
# Prep partly (skip already-ready completely)
DATASETS=partly sbatch --partition=cs sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh

# Per-dataset train/val tars
DATASETS=completely,partly sbatch --partition=cs sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh
```

**2b. Config overlays**

- [`datasets_completely_partly.toml`](../../../PatchPerPix/experiments/flylight/setups/setup01/datasets_completely_partly.toml)
- [`datasets_test_synthetic.toml`](../../../PatchPerPix/experiments/flylight/setups/setup01/datasets_test_synthetic.toml)

```bash
./sbatch/ppp/ppp_submit.sh train <preset> --datasets completely,partly \
  --config datasets_completely_partly.toml

./sbatch/ppp/ppp_submit.sh array <preset> --split test --ckpt <N> \
  --datasets completely,synthetic --config datasets_test_synthetic.toml
```

`--datasets` must match the TOML list for that job. Naming `"partly"` auto-sets
`add_partly` / `add_partly_val`.

**2c. Train staging (list mode preserved)**

`STAGE_DATA=1` extracts `fisbe/<ds>/{train,val}_ppp.tar` →
`$SLURM_TMPDIR/ppp_data/<ds>/{train,val}/` and passes
`run_ppp --fisbe-root $SLURM_TMPDIR/ppp_data` (no `--train-data` / `--val-data`).

Infer/eval arrays still read scratch `fisbe/` with matching `--datasets`.

---

### Phase 3 — Logging under `base/log/<dataset>/<sample>/`

**Status:** done.

| File | Change |
|------|--------|
| [`run_ppp.py`](../../../PatchPerPix/experiments/run_ppp.py) | No `--sample` → `base/run.log`; with `--sample` → `base/log/<dataset>/<stem>/run.log` |
| [`cleanup_ppp_infer.sh`](../../../sbatch/ppp/tools/cleanup_ppp_infer.sh) | Never deletes `log/`; optional `--purge-legacy-sample-logs` for old dirs |

```bash
# List legacy sample-only log dirs (pre-Phase-3); does not touch processed/
./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp ppp_basic_8h --purge-legacy-sample-logs --dry-run
```

`ppp_train_track.py` still reads root `run.log` only.

---

## 5. Why not logging first?

If `run_ppp.py` requires `dataset/stem` for logs while val arrays still pass a
flat stem + `--test-data`, every val task fails at startup even though predict
path resolution would have worked under the old recipe.

Namespaced SAMPLE + `--test-data` (legacy single dir) is also a bad mix: input
resolution can disagree with sample ID shape. Phase 1 removes both sides of the
mismatch together; Phase 3 only formalizes the log path.

---

## 6. Checklist

**Phase 1**

- [x] `run_ppp.py`: `--split val|test` (list mode, no `_cli_test_data`); reject flat `--sample`
- [x] `ppp_job.sh`: list mode for val and test; no SAMPLE strip; no `--test-data`; pass `--split`
- [x] Defaults / submit help: namespaced `--sample`; reject flat
- [x] Docs: update list-mode / val-vs-test notes
- [x] Smoke: val + test array dry-run → nested sample, `--split`, no `--test-data`

**Phase 2**

- [x] Synthetic policy: **test-only** (no train/val splits yet)
- [x] `prepare_fisbe_for_ppp.py --datasets` + `prepare_fisbe_for_ppp_sbatch.sh` (partly submitted)
- [x] Opt-in overlays `datasets_completely_partly.toml` / `datasets_test_synthetic.toml`
- [x] Multi-dataset stage tars + `STAGE_DATA` → `--fisbe-root` (list mode)
- [x] Smoke: tar builder, train dry-run export, synthetic test listing

**Phase 3**

- [x] `run_ppp.py` → `base/log/<dataset>/<stem>/run.log`
- [x] `cleanup_ppp_infer.sh` keeps `log/`; `--purge-legacy-sample-logs`
- [x] Docs + smoke

**Namespaced layout (legacy flat dropped)**

- [x] `get_list_samples` prefers `{dataset}/{stem}`; errors on flat siblings
- [x] predict/label/evaluate require list mode; nested evaluated tomls
- [x] Aggregate = split × dataset tables; `--purge-flat-outputs`

---

## 7. Out of scope

- Rewriting archived scripts under `sbatch/ppp/archive/`
- Moving train logs into `log/train/` (keep root `run.log`)
- Silent flat→namespaced coercion
- Forcing `partly` / `synthetic` into default TOMLs (opt-in only)
