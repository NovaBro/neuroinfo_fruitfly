> **Historical / archived.** Current PatchPerPix ops and canonical presets: [`ppp_overview.md`](../ppp_overview.md). Keep this file for background only; do not treat recipes here as the live path.
>
> ---

# PatchPerPix sbatch consolidation plan

Plan to collapse the many near-duplicate scripts under [`sbatch/ppp/`](../../../sbatch/ppp/)
into a small submit surface (job body + CLI), following the pattern already used for
BiaPy.

**Status:** implemented — use [`sbatch/ppp/ppp_submit.sh`](../../../sbatch/ppp/ppp_submit.sh) +
[`sbatch/ppp/ppp_submit_lib.sh`](../../../sbatch/ppp/ppp_submit_lib.sh) (presets/plumbing) +
[`sbatch/ppp/ppp_job.sh`](../../../sbatch/ppp/ppp_job.sh). Per-experiment scripts remain valid
until you validate a train/array run and archive them (migration step 5).

Companion docs: [`ppp_performance_plan.md`](ppp_performance_plan.md),
[`ppp_recommendations.md`](ppp_recommendations.md). Architecture:
[`PatchPerPix/CODEBASE.md`](../../../PatchPerPix/CODEBASE.md).

Submit from the **repo root** (`/scratch/wmz2007/neuroinfo_fruitfly`), not from
inside `sbatch/`.

---

## 1. Problem

[`sbatch/ppp/`](../../../sbatch/ppp/) has ~40 `.sh` files. Most are copy-paste clones across
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

Plus one-offs: [`ppp_sbatch.sh`](../../../sbatch/ppp/ppp_sbatch.sh) (legacy / hardcoded),
[`ppp_array_sbatch.sh`](../../../sbatch/ppp/ppp_array_sbatch.sh), diagnose scripts.

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

- Train: [`ppp_basic_fmaps24_train_sbatch.sh`](../../../sbatch/ppp/ppp_basic_fmaps24_train_sbatch.sh),
  [`ppp_basic_long8h_train_sbatch.sh`](../../../sbatch/ppp/ppp_basic_long8h_train_sbatch.sh)
- Infer: [`ppp_basic_fmaps24_infer_sbatch.sh`](../../../sbatch/ppp/ppp_basic_fmaps24_infer_sbatch.sh)
- Array + chain: [`ppp_basic_fmaps24_array_sbatch.sh`](../../../sbatch/ppp/ppp_basic_fmaps24_array_sbatch.sh),
  [`ppp_basic_fmaps24_full_array_chain.sh`](../../../sbatch/ppp/ppp_basic_fmaps24_full_array_chain.sh)
- Ckpt sweep: [`ppp_basic_fmaps24_ckpt_sweep_chain.sh`](../../../sbatch/ppp/ppp_basic_fmaps24_ckpt_sweep_chain.sh)
- VI sweep (parameterized env): [`ppp_basic_long8h_vi_sweep_sbatch.sh`](../../../sbatch/ppp/ppp_basic_long8h_vi_sweep_sbatch.sh)

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
| [`sbatch/biapy/biapy-py_sbatch.sh`](../../../sbatch/biapy/biapy-py_sbatch.sh) | Single job body; config/mode from args; minimal `#SBATCH` |
| [`sbatch/biapy/biapy-py_sbatch_chain.sh`](../../../sbatch/biapy/biapy-py_sbatch_chain.sh) | Submitter sets `--job-name`, cpus, time, mem, GPU; chains `afterok` |

SLURM CLI flags override `#SBATCH` directives, so one job file can serve many
resource profiles without a file per experiment.

PPP should mirror this: **job body vs submit/chain CLI**.

---

## 5. Recommended design (~2–3 scripts)

```text
┌─────────────────────┐         ┌──────────────────────┐
│  ppp_submit.sh      │ sbatch  │  ppp_job.sh          │
│  + ppp_submit_lib   │────────▶│  (compute node)      │
│  (CLI / presets /   │ export  │  GPU log, stage?,    │
│   modes / chains)   │ + deps  │  singularity +       │
└─────────────────────┘         │  run_ppp.py          │
         │                      └──────────────────────┘
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
| `SAMPLE` | Optional `--sample {dataset}/{stem}` (flat rejected); per-sample log → `metrics/ppp/<exp>/log/<dataset>/<stem>/run.log` |
| `SPLIT` | `val` / `test` → `run_ppp --split` (omit `--test-data`; list mode) |
| `DATASETS` | FISBe folder names for array discovery **and** train staging (must match TOML `*_datasets`) |
| `STAGE_DATA` | `1` for train: extract `fisbe/<ds>/{train,val}_ppp.tar` → `$SLURM_TMPDIR/ppp_data/<ds>/{train,val}/`, pass `--fisbe-root` (list mode) |

Shared logic once: GPU logger + trap, optional multi-dataset staging, singularity launch,
layered `--config flylight/setups/setup01/<name>.toml`.

See [`ppp_list_mode_unify.md`](ppp_list_mode_unify.md) Phases 1–3 (list mode, multi-dataset, `log/` layout).
### 5.2 `ppp_submit.sh` + `ppp_submit_lib.sh` — CLI + presets + chains

Presets live as a `|`-delimited table in `ppp_submit_lib.sh` → overlay + train defaults:

| Preset (= TOML stem) | Overlay | Exp id / metrics | Train cpus | Train wall |
|----------------------|---------|------------------|------------|------------|
| `basic_8h` | `basic_8h.toml` | `ppp_basic_8h` | 16 | 8h |
| `basic_long8h` | `basic_long8h.toml` | `ppp_basic_long8h` | 16 | 8h |
| `basic_fmaps24` | `basic_fmaps24.toml` | `ppp_basic_fmaps24` | 32 | 8h |
| `basic_fmaps24_400k` | `basic_fmaps24_400k.toml` | `ppp_basic_fmaps24_400k` | 32 | 48h |

Mode → resource defaults:

| Mode | Typical resources |
|------|-------------------|
| `train` | 192g, stage data, GPU, `l40s_public` |
| `run` (one sample; `--do` tasks) | 96g, 2–4h, GPU |
| `evaluate` (CPU) | `cpu_short`, 32–64g |
| `array --stage …` | 96g, 1.5–4h; submitter sets `--array=0-N%K` |

Primary subcommands:

- `train` / `run` / `evaluate`
- `chain <preset> train run` — `afterok`
- `ckpt-sweep <preset> --ckpts 50000,60000,…`
- `array <preset> --ckpt N --stage all|predict|label|legacy` (`--split` comma-capable)

**Breaking changes:**

1. Modes `infer` / `predict` / `label` / `infer-array` / `predict-array` /
   `label-array` and flag `--splits` are gone. Use:

| Old | New |
|-----|-----|
| `infer` | `run` (default `--do predict+label+evaluate`) |
| `predict` | `run --do predict` |
| `label` | `run --do label+evaluate` |
| `infer-array` | `array` / `array --stage all` |
| `predict-array` | `array --stage predict` |
| `label-array` | `array --stage label` |
| `--splits val,test` | `--split val,test` |
| bare `array` (GPU predict+label) | `array --stage legacy` (+ optional `--agg`) |

2. CLI presets match TOML stems (short names removed). Exp ids / `metrics/ppp/`
   folders unchanged:

| Old preset | New preset |
|------------|------------|
| `8h` | `basic_8h` |
| `long8h` | `basic_long8h` |
| `fmaps24` | `basic_fmaps24` |
| `fmaps24_400k` | `basic_fmaps24_400k` |
| `fmaps24_400k_bs5_aug` | `basic_fmaps24_400k_bs5_aug` |
| `fmaps24_400k_aug_cheap` | `basic_fmaps24_400k_aug_cheap` |

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
# From repo root (preset = TOML stem; writes metrics/ppp/ppp_basic_* by default)
./sbatch/ppp/ppp_submit.sh train  basic_fmaps24
./sbatch/ppp/ppp_submit.sh train  basic_fmaps24_400k    # 48h / 32 cpus preset
./sbatch/ppp/ppp_submit.sh run    basic_long8h --ckpt 70000   # one sample; default --do predict+label+evaluate
./sbatch/ppp/ppp_submit.sh chain  basic_fmaps24 train run
./sbatch/ppp/ppp_submit.sh ckpt-sweep basic_fmaps24 --ckpts 50000,60000,70000,80000
# Preferred full-split pipeline (GPU predict → GPU graph → CPU label → CPU evaluate):
./sbatch/ppp/ppp_submit.sh array basic_fmaps24_400k --ckpt 400000 --pct 2         # --stage all
./sbatch/ppp/ppp_submit.sh array basic_fmaps24 --stage predict --split val --ckpt 80000 --pct 2
./sbatch/ppp/ppp_submit.sh array basic_long8h --stage label --split val,test --ckpt 70000 --pct 2
# Legacy (predict+label on GPU; optional --agg evaluate):
./sbatch/ppp/ppp_submit.sh array basic_fmaps24 --stage legacy --split val --ckpt 80000 --pct 2 --agg
./sbatch/ppp/ppp_submit.sh vi-sweep basic_long8h --ckpt 30000
./sbatch/ppp/ppp_submit.sh --dry-run train basic_fmaps24   # print sbatch; do not submit
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

Predict/label/evaluate write under `metrics/ppp/<exp>/{val|test}/` matching
`--split` (list mode: TOML `val_datasets` / `test_datasets`, no `--test-data`).
After `-d evaluate`, [`ppp_job.sh`](../../../sbatch/ppp/ppp_job.sh) copies
`summary.csv` → `summary_<split>.csv` beside it under
`metrics/ppp/<exp>/<split>/evaluated/<ckpt>/<VI params>/`.

See also [`ppp_list_mode_unify.md`](ppp_list_mode_unify.md) (list mode +
**namespaced** `processed` / `instanced` / `evaluated` under `{dataset}/`;
Aggregate = one table per `split × dataset`).

| Command | Result |
|---------|--------|
| `evaluate <preset> --split val --ckpt N` | under `val/evaluated/…`: `summary.csv` + `summary_val.csv` |
| `evaluate <preset> --split test --ckpt N` | under `test/evaluated/…`: `summary.csv` + `summary_test.csv` |
| `evaluate <preset> --ckpt N` (no `--split`) | val then test → `summary_val.csv` (in val/) + `summary_test.csv` (in test/) + consolidated **`summary.csv`** under test/evaluated (mean/sum via [`merge_ppp_summary_csvs.py`](../../../sbatch/ppp/merge_ppp_summary_csvs.py)) |

Full test evaluate is memory-heavy (prefer `--mem 128g --time 4:00:00`). If flat
outputs sit beside `{dataset}/` dirs, purge first:

```bash
./sbatch/ppp/tools/cleanup_ppp_infer.sh --exp <exp> --purge-flat-outputs --execute
```

```bash
./sbatch/ppp/ppp_submit.sh evaluate basic_fmaps24 --ckpt 80000              # both splits
./sbatch/ppp/ppp_submit.sh evaluate basic_long8h --split val --ckpt 70000   # val only
```

### Pred affs / pred_code disk policy

After `-d evaluate`, [`ppp_job.sh`](../../../sbatch/ppp/ppp_job.sh) runs
[`strip_pred_affs.py`](../../../sbatch/ppp/strip_pred_affs.py) to delete large prediction
arrays under `metrics/ppp/<exp>/{val,test}/processed/<ckpt>/`:

- `volumes/pred_affs` unless `KEEP_PRED_AFFS=1` (or `--keep-pred-affs`)
- `volumes/pred_code` unless `KEEP_PRED_CODE=1` (or `--keep-pred-code`)

`volumes/pred_numinst` is always kept (web numinst overlays). Samples listed in
`KEEP_AFFS_SAMPLE` (comma-separated) retain **both** arrays for VI / debug reuse.
Default:

`R22C03-20180918_66_J2.zarr` (FISBe val) and `JRC_SS04989-20160318_24_A2.zarr` (FISBe test).

If none of the keep names exist in a ckpt dir, the first `*.zarr` is kept instead.
Override: `KEEP_AFFS_SAMPLE=a.zarr,b.zarr sbatch …` (or export before submit).
Missing arrays (e.g. no `pred_code` on non-train_code runs) are skipped quietly.

One-time cleanup (dry-run then apply; defaults to both arrays):

```bash
python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp --dry-run
python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp
# Affs only:
python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp --arrays volumes/pred_affs
# Codes only:
python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp --arrays volumes/pred_code
```

Ckpt-sweep skip-predict and `DISCARD_PARTIAL` treat “prediction exists” as a
**complete primary array**, not merely a sample `.zarr` directory: `pred_affs`
for direct-affinity models, `pred_code` when `NEEDS_DECODE=1` (so numinst-only
trees still re-predict). Label readiness still requires complete `pred_affs`.
Legacy runs that mixed FISBe-val stems under `test/` are not migrated; re-run
array with `--split val,test` to populate the correct trees.

### Array / re-run: incomplete vs overwrite

Prefer **`array`** (default `--stage all`): GPU predict → [GPU decode if
`train_code`] → afterok **split label** → afterok CPU `evaluate`. Decode is
inserted when the merged base+overlay+extras have `model.train_code` /
`training.train_code` (`sbatch/ppp/needs_decode.py` → `NEEDS_DECODE=1`). For
`basic-ppp-decoder` use `--base original_default_train_code.toml` (overlay also
sets `train_code = true`). The label is split because blockwise `vote_instances`
needs CUDA for the patch graph, but the final `affGraphToInstances` is CPU-only
and idles the GPU for half the job (`l40s_public` low-util kills). `vi_cpu.toml`
alone does **not** work with `blockwise = true` (asserts cuda).

- GPU array `ppp_<preset>_decode_<split>` (`DO_TASKS=decode`, train_code only):
  expands `volumes/pred_code` → `volumes/pred_affs` in-place in the processed zarr.
- GPU array `ppp_<preset>_labgraph_<split>` (`vi_split_gpu.toml`: `termAfterStitchGraph`):
  per-block + inter-block patch graph cached in `<instanced>/<sample>.zarr` `volumes/blocks`,
  marks `stitch_graph_done`, writes no HDF.
- CPU array `ppp_<preset>_label_<split>` (`vi_split_cpu.toml`: `cuda=false`, `graphToInst`),
  `--dependency=aftercorr` (task i waits for GPU task i): reads the cached graph and labels.
  Raises if `stitch_graph_done` is missing.

Both label phases use `label_resume.toml`. Escape hatch: `--vi-gpu` runs the label
stage as one GPU job.

`--stage decode` runs decode-only (resume when `pred_code` already exists).
`--stage legacy` submits `predict[+decode]+label` on GPU with `DISCARD_PARTIAL=1`
and stacks `default_l40s.toml` (`overwrite = true`) plus `eval_rm0.toml`
(`from_scratch = true`); optional `--agg` for CPU evaluate.

| Situation | Behavior |
|-----------|----------|
| Incomplete primary predict array (zarr exists but chunk coverage short) | Discarded before predict (`check_prediction_complete.py`); primary is `pred_code` when `NEEDS_DECODE=1`, else `pred_affs`; `pred_numinst` kept if present; then predict rewrites |
| Complete existing predict / label / eval for that sample | **Overwritten** on `--stage legacy` / `--stage predict` — not skipped |
| `--stage decode` | GPU decode only; requires complete `pred_code`; writes `pred_affs` |
| `--stage label` | GPU graph → CPU label split by default; `label_resume.toml` (`overwrite = false`) skips finished HDFs; **always** chains CPU evaluate afterok |
| `--stage all` | predict → [decode] → label → evaluate per split (see above) |
| `ckpt-sweep` | Skips predict(+decode) when `pred_affs` is already **complete**; with train_code, skips predict only when `pred_code` is complete but still decodes if affs missing; still runs label+evaluate |

Re-running label+evaluate after predicts exist (e.g. idle-killed GPU label):

```bash
./sbatch/ppp/ppp_submit.sh array basic_fmaps24_400k --stage label --split test --ckpt 400000 --pct 2
```

Decode then label for a train_code preset when `pred_code` is already on disk:

```bash
./sbatch/ppp/ppp_submit.sh array basic-ppp-decoder --base original_default_train_code.toml \
  --ckpt 15000 --split val,test --stage decode --pct 2
./sbatch/ppp/ppp_submit.sh array basic-ppp-decoder --base original_default_train_code.toml \
  --ckpt 15000 --split val,test --stage label --pct 2
```

Full redo via legacy GPU array:

```bash
./sbatch/ppp/ppp_submit.sh array basic_fmaps24 --stage legacy --split val,test --ckpt 80000 --pct 2 --agg
```

therefore cleans up broken affs and redoes everything for each sample under the matching
`val/` or `test/` tree. It does **not** reuse older mixed outputs that still sit under
`test/` from before split routing (those are left as legacy).

---

## 8. Migration steps

1. **Done** — `ppp_job.sh` + `ppp_submit.sh` added next to existing scripts.
2. **Done (dry-run)** — `basic_fmaps24` / `basic_long8h` train+run(+chain) args verified via `--dry-run`.
   Submit a real train/run when ready to validate on-cluster.
3. **Done** — `ckpt-sweep` and `array` (+ `--agg` on `--stage legacy`) ported into `ppp_submit.sh`.
4. **Done** — `vi-sweep` and `array --stage predict|label|all` (CPU `vi_cpu` label by
   default; evaluate afterok for `all`/`label`) ported into `ppp_submit.sh`.
5. **Pending (you)** — after one successful train and one array via the new path, archive
   per-experiment clones under `sbatch/ppp/archive/`.
6. **Done** — diagnose scripts and scratch logs left alone.
7. **Done** — CLI simplify: `--split` comma-capable; `array --stage`; `run --do`;
   helpers in `ppp_submit_lib.sh`.
8. **Done** — deprecated mode/flag aliases removed (hard break; see §5.2 table).
9. **Done** — CLI presets renamed to TOML stems (`basic_fmaps24`, …); exp ids unchanged.

---

## 9. Out of scope

- Deleting or moving existing `sbatch/ppp/*.sh` before your validation (step 5)
- Changing PPP Python / TOML experiment configs

---

## 10. Bottom line

Go from ~40 scripts to roughly **2–3** (`ppp_job.sh`, `ppp_submit.sh` + lib, plus
diagnose). Use the same job-body + chain-submitter pattern as BiaPy. TOML overlays
remain the experiment definitions; SLURM scripts become a thin, parameterized
launcher.
