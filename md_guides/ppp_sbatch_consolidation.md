# PatchPerPix sbatch consolidation plan

Plan to collapse the many near-duplicate scripts under [`sbatch/ppp/`](../sbatch/ppp/)
into a small submit surface (job body + CLI), following the pattern already used for
BiaPy.

**Status:** implemented — use [`sbatch/ppp/ppp_submit.sh`](../sbatch/ppp/ppp_submit.sh) +
[`sbatch/ppp/ppp_job.sh`](../sbatch/ppp/ppp_job.sh). Per-experiment scripts remain valid
until you validate a train/array run and archive them (migration step 5).

Companion docs: [`ppp_performance_plan.md`](ppp_performance_plan.md),
[`ppp_recommendations.md`](ppp_recommendations.md). Architecture:
[`PatchPerPix/CODEBASE.md`](../PatchPerPix/CODEBASE.md).

Submit from the **repo root** (`/scratch/wmz2007/neuroinfo_fruitfly`), not from
inside `sbatch/`.

---

## 1. Problem

[`sbatch/ppp/`](../sbatch/ppp/) has ~40 `.sh` files. Most are copy-paste clones across
three experiment families, differing mainly by:

- experiment id (`-id`)
- overlay TOML under `PatchPerPix/experiments/flylight/setups/setup01/`
- `-d` task list (`train`, `predict`, `label`, `evaluate`, …)
- SLURM resources (cpus, mem, time, partition)
- checkpoint / sample / split knobs

Maintaining N copies means every fix (GPU logger, staging, singularity flags) must be
re-applied by hand.

---

## 2. Current inventory

### Experiment families

| Family | Exp id | Overlay TOML |
|--------|--------|--------------|
| `ppp_basic_8h_*` | `ppp_basic_8h` | `basic_8h.toml` |
| `ppp_basic_long8h_*` | `ppp_basic_long8h` | `basic_long8h.toml` |
| `ppp_basic_fmaps24_*` | `ppp_basic_fmaps24` | `basic_fmaps24.toml` |
| `ppp_basic_fmaps24_400k_*` | `ppp_basic_fmaps24_400k` | `basic_fmaps24_400k.toml` |

Plus one-offs: [`ppp_sbatch.sh`](../sbatch/ppp/ppp_sbatch.sh) (legacy / hardcoded),
[`ppp_array_sbatch.sh`](../sbatch/ppp/ppp_array_sbatch.sh), diagnose scripts.

### Job kinds (repeated per family)

| Kind | Role | Typical resources |
|------|------|-------------------|
| **train** | Stage train/val to `$SLURM_TMPDIR`, `-d train` | 192g, 8–48h, GPU, cpus 16 or 32 |
| **infer** | One val sample, `-d predict label evaluate` + VI/eval overlays | 96g, 2–4h, GPU |
| **ckpt_predict** | Predict at `CKPT=` | 96g, ~2h, GPU |
| **ckpt_label_eval** | Label + evaluate at `CKPT=` (reuse processed zarr) | 96g, 2–4h, GPU |
| **array** | One volume per SLURM array task | 96g, 1.5–4h, GPU; `--array` set at submit |
| **eval_agg / eval_rm0** | Evaluate-only (often CPU) | 32–64g, `cpu_short` or GPU-for-queue |
| **chain** | Login-node helper: `sbatch --dependency=afterok:…` | no `#SBATCH` body |

### Representative scripts

- Train: [`ppp_basic_fmaps24_train_sbatch.sh`](../sbatch/ppp/ppp_basic_fmaps24_train_sbatch.sh),
  [`ppp_basic_long8h_train_sbatch.sh`](../sbatch/ppp/ppp_basic_long8h_train_sbatch.sh)
- Infer: [`ppp_basic_fmaps24_infer_sbatch.sh`](../sbatch/ppp/ppp_basic_fmaps24_infer_sbatch.sh)
- Array + chain: [`ppp_basic_fmaps24_array_sbatch.sh`](../sbatch/ppp/ppp_basic_fmaps24_array_sbatch.sh),
  [`ppp_basic_fmaps24_full_array_chain.sh`](../sbatch/ppp/ppp_basic_fmaps24_full_array_chain.sh)
- Ckpt sweep: [`ppp_basic_fmaps24_ckpt_sweep_chain.sh`](../sbatch/ppp/ppp_basic_fmaps24_ckpt_sweep_chain.sh)
- VI sweep (parameterized env): [`ppp_basic_long8h_vi_sweep_sbatch.sh`](../sbatch/ppp/ppp_basic_long8h_vi_sweep_sbatch.sh)

---

## 3. Evidence of duplication

Train scripts for fmaps24 vs long8h differ almost only in:

- `#SBATCH --job-name` / `--output` / `--error`
- `--cpus-per-task` (32 vs 16)
- GPU log CSV name
- `--config …/basic_*.toml` and `-id ppp_basic_*`

Same pattern for infer, ckpt predict/label_eval, and many chains. The shared core is:

1. `set -euo pipefail` + `module purge` + `cd` repo root
2. GPU logger (`nvidia-smi … -l 3`) + `trap cleanup EXIT`
3. Optional `rsync` of `fisbe/completely/{train,val}` → `$SLURM_TMPDIR/ppp_data`
4. `singularity exec --nv --overlay env/ppp.ext3:ro …` → `conda activate ppp` →
   `python3 -u run_ppp.py --setup setup01` with layered `--config`s

Hardcoded values that should become args: `CKPT=70000` / `80000`, sample
`R22C03-20180918_66_J2`, `SPLIT=val|test`, VI overlay `vi_th_0_3.toml`,
`eval_rm0.toml`.

---

## 4. Precedent: BiaPy consolidation

BiaPy already uses a two-script design:

| Script | Role |
|--------|------|
| [`sbatch/biapy/biapy-py_sbatch.sh`](../sbatch/biapy/biapy-py_sbatch.sh) | Single job body; config/mode from args; minimal `#SBATCH` |
| [`sbatch/biapy/biapy-py_sbatch_chain.sh`](../sbatch/biapy/biapy-py_sbatch_chain.sh) | Submitter sets `--job-name`, cpus, time, mem, GPU; chains `afterok` |

SLURM CLI flags override `#SBATCH` directives, so one job file can serve many
resource profiles without a file per experiment.

PPP should mirror this: **job body vs submit/chain CLI**.

---

## 5. Recommended design (~2–3 scripts)

```text
┌─────────────────────┐         ┌──────────────────────┐
│  ppp_submit.sh      │ sbatch  │  ppp_job.sh          │
│  (login / chain)    │────────▶│  (compute node)      │
│  presets + modes    │ export  │  GPU log, stage?,    │
│  resource defaults  │ +deps  │  singularity +       │
│  afterok chains     │         │  run_ppp.py          │
└─────────────────────┘         └──────────────────────┘
         │
         ▼
   diagnose_* stay separate (different Python entrypoints)
   legacy per-exp *.sh → archive after validation
```

### 5.1 `ppp_job.sh` — single SLURM job body

Minimal `#SBATCH` (account, `%x-%j` logs). Behavior from env / CLI:

| Variable | Purpose |
|----------|---------|
| `EXP_ID` | `-id` under `metrics/ppp/` |
| `MODE` / `DO_TASKS` | `-d` list (`train`, `predict`, `label`, `evaluate`, …) |
| `CONFIGS` | Space-separated overlay basenames (or preset-resolved list) |
| `CKPT` | `--checkpoint N` or empty → `--test-checkpoint last` |
| `SAMPLE` | Optional `--sample` |
| `SPLIT` | `val` / `test` → `--test-data fisbe/completely/$SPLIT` |
| `STAGE_DATA` | `1` for train (rsync to `$SLURM_TMPDIR`) |

Shared logic once: GPU logger + trap, optional staging, singularity launch,
layered `--config flylight/setups/setup01/<name>.toml`.

### 5.2 `ppp_submit.sh` — CLI + presets + chains

Presets → overlay + train defaults:

| Preset | Overlay | Train cpus | Train wall |
|--------|---------|------------|------------|
| `8h` | `basic_8h.toml` | 16 | 8h |
| `long8h` | `basic_long8h.toml` | 16 | 8h |
| `fmaps24` | `basic_fmaps24.toml` | 32 | 8h |
| `fmaps24_400k` | `basic_fmaps24_400k.toml` | 32 | 48h |

Mode → resource defaults:

| Mode | Typical resources |
|------|-------------------|
| `train` | 192g, stage data, GPU, `l40s_public` |
| `predict` / `infer` / `label` | 96g, 2–4h, GPU |
| `evaluate` (CPU) | `cpu_short`, 32–64g |
| `array` | 96g, 1.5–4h; submitter sets `--array=0-N%K` |

Subcommands fold today’s chain scripts:

- `train` / `infer` / `evaluate`
- `chain <preset> train infer` — `afterok`
- `ckpt-sweep <preset> --ckpts 50000,60000,…`
- `array <preset> --split val|test --ckpt N --pct 2` then optional eval-agg

### 5.3 Keep separate / archive

| Keep separate | Why |
|---------------|-----|
| `diagnose_ppp_*.sh` | Different Python entrypoints, CPU partitions |
| Legacy `ppp_sbatch.sh`, `ppp_array_sbatch.sh` | Older `ppp_experiments` layout — archive once presets cover use cases |

Historical `.out`/`.err` and `sbatch/ppp/archieve/` stay as scratch; only the
submit surface shrinks.

---

## 6. Parameterization map

| Knob | Today (hardcoded per file) | Consolidated |
|------|----------------------------|--------------|
| Exp id | `-id ppp_basic_*` | `--id` / preset default |
| Overlay TOMLs | `basic_*.toml` + `vi_*` + `eval_*` | preset + `--config` extras |
| `-d` tasks | train / predict / label / evaluate | `--mode` or `--do` |
| Checkpoint | `CKPT=…` / `last` | `--ckpt` |
| Sample vs full split | hardcoded sample or `--test-data` | `--sample` / `--split` |
| Staging | train scripts only | `STAGE_DATA=1` for train |
| Partition | `l40s_public` / `rtx6000` / `cpu_short` | `--partition` on submit |
| Array concurrency | `%2` in chain | `--array 0-N%2` |

Experiment **content** stays in TOML under
`PatchPerPix/experiments/flylight/setups/setup01/`. The sbatch layer only picks
preset + mode + resources.

---

## 7. CLI examples (implemented)

```bash
# From repo root
./sbatch/ppp/ppp_submit.sh train  fmaps24
./sbatch/ppp/ppp_submit.sh train  fmaps24_400k          # 48h / 32 cpus preset
./sbatch/ppp/ppp_submit.sh infer  long8h
./sbatch/ppp/ppp_submit.sh chain  fmaps24 train infer
./sbatch/ppp/ppp_submit.sh ckpt-sweep fmaps24 --ckpts 50000,60000,70000,80000
./sbatch/ppp/ppp_submit.sh array  fmaps24 --split val --ckpt 80000 --pct 2 --agg
./sbatch/ppp/ppp_submit.sh label-array long8h --splits val,test --ckpt 70000 --pct 2 --agg
./sbatch/ppp/ppp_submit.sh vi-sweep long8h --ckpt 30000
./sbatch/ppp/ppp_submit.sh --dry-run train fmaps24      # print sbatch; do not submit
```

Equivalent legacy:

```bash
sbatch sbatch/ppp/ppp_basic_fmaps24_train_sbatch.sh
bash  sbatch/ppp/ppp_basic_fmaps24_chain.sh
bash  sbatch/ppp/ppp_basic_fmaps24_ckpt_sweep_chain.sh
bash  sbatch/ppp/ppp_basic_fmaps24_full_array_chain.sh
```

`CONFIGS` and `DO_TASKS` use `+` separators in SLURM `--export` (spaces are unsafe).

### Evaluate summaries (val / test)

Predict/label/evaluate write under `metrics/ppp/<exp>/{val|test}/` matching the
FISBe `--test-data` split (`completely/val` or `completely/test`). After
`-d evaluate`, [`ppp_job.sh`](../sbatch/ppp/ppp_job.sh) copies
`summary.csv` → `summary_<split>.csv` beside it under
`metrics/ppp/<exp>/<split>/evaluated/<ckpt>/<VI params>/`.

| Command | Result |
|---------|--------|
| `evaluate <preset> --split val --ckpt N` | under `val/evaluated/…`: `summary.csv` + `summary_val.csv` |
| `evaluate <preset> --split test --ckpt N` | under `test/evaluated/…`: `summary.csv` + `summary_test.csv` |
| `evaluate <preset> --ckpt N` (no `--split`) | val then test → `summary_val.csv` (in val/) + `summary_test.csv` (in test/) + consolidated **`summary.csv`** under test/evaluated (mean/sum via [`merge_ppp_summary_csvs.py`](../sbatch/ppp/merge_ppp_summary_csvs.py)) |

```bash
./sbatch/ppp/ppp_submit.sh evaluate fmaps24 --ckpt 80000              # both splits
./sbatch/ppp/ppp_submit.sh evaluate long8h --split val --ckpt 70000   # val only
```

### Pred affs disk policy

After `-d evaluate`, [`ppp_job.sh`](../sbatch/ppp/ppp_job.sh) runs
[`strip_pred_affs.py`](../sbatch/ppp/strip_pred_affs.py) unless `KEEP_PRED_AFFS=1`
(or `./sbatch/ppp/ppp_submit.sh … --keep-pred-affs`). That deletes
`volumes/pred_affs` under `metrics/ppp/<exp>/{val,test}/processed/<ckpt>/` while keeping
`volumes/pred_numinst` (web numinst overlays). Samples listed in
`KEEP_AFFS_SAMPLE` (comma-separated) retain full affs for VI reuse. Default:

`R22C03-20180918_66_J2.zarr` (FISBe val) and `JRC_SS04989-20160318_24_A2.zarr` (FISBe test).

If none of the keep names exist in a ckpt dir, the first `*.zarr` is kept instead.
Override: `KEEP_AFFS_SAMPLE=a.zarr,b.zarr sbatch …` (or export before submit).

One-time cleanup (dry-run then apply):

```bash
python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp --dry-run
python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp
```

Ckpt-sweep skip-predict and `DISCARD_PARTIAL` treat “prediction exists” as **complete
`pred_affs`**, not merely a sample `.zarr` directory (so numinst-only trees re-predict).
Legacy runs that mixed FISBe-val stems under `test/` are not migrated; re-run array
with `--splits val,test` to populate the correct trees.

### Array / re-run: incomplete vs overwrite

`array` submits `predict+label+evaluate` with `DISCARD_PARTIAL=1` and stacks
`default_l40s.toml` (`overwrite = true`) plus `eval_rm0.toml` (`from_scratch = true`).

| Situation | Behavior |
|-----------|----------|
| Incomplete `pred_affs` (zarr exists but chunk coverage short) | Discarded before predict (`check_prediction_complete.py`); `pred_numinst` kept if present; then predict rewrites |
| Complete existing predict / label / eval for that sample | **Overwritten** — not skipped. Array does not resume finished volumes |
| `label-array` only | Uses `label_resume.toml` (`overwrite = false`) so finished instance HDFs are skipped |
| `ckpt-sweep` | Skips predict when `pred_affs` is already **complete**; still runs label+evaluate |

Re-running:

```bash
./sbatch/ppp/ppp_submit.sh array fmaps24 --splits val,test --ckpt 80000 --pct 2 --agg
```

therefore cleans up broken affs and redoes everything for each sample under the matching
`val/` or `test/` tree. It does **not** reuse older mixed outputs that still sit under
`test/` from before split routing (those are left as legacy).

---

## 8. Migration steps

1. **Done** — `ppp_job.sh` + `ppp_submit.sh` added next to existing scripts.
2. **Done (dry-run)** — `fmaps24` / `long8h` train+infer(+chain) args verified via `--dry-run`.
   Submit a real train/infer when ready to validate on-cluster.
3. **Done** — `ckpt-sweep` and `array` (+ `--agg` evaluate) ported into `ppp_submit.sh`.
4. **Done** — `vi-sweep` and `label-array` ported into `ppp_submit.sh`.
5. **Pending (you)** — after one successful train and one array via the new path, archive
   per-experiment clones under `sbatch/ppp/archieve/`.
6. **Done** — diagnose scripts and scratch logs left alone.

---

## 9. Out of scope

- Deleting or moving existing `sbatch/ppp/*.sh` before your validation (step 5)
- Changing PPP Python / TOML experiment configs

---

## 10. Bottom line

Go from ~40 scripts to roughly **2–3** (`ppp_job.sh`, `ppp_submit.sh`, plus
diagnose). Use the same job-body + chain-submitter pattern as BiaPy. TOML overlays
remain the experiment definitions; SLURM scripts become a thin, parameterized
launcher.
