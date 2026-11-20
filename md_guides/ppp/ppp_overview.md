# PatchPerPix overview (current)

How to run FISBe PatchPerPix on NYU Greene. Index of all PPP guides: [`README.md`](README.md).

- Architecture: [`PatchPerPix/CODEBASE.md`](../../PatchPerPix/CODEBASE.md)
- Configs: [`setup01/configs/README.md`](../../PatchPerPix/experiments/flylight/setups/setup01/configs/README.md)
- Launchers: [`ppp_submit_train.sh`](../../sbatch/ppp/ppp_submit_train.sh) and
  [`ppp_submit_test.sh`](../../sbatch/ppp/ppp_submit_test.sh), which both call
  [`ppp_job.sh`](../../sbatch/ppp/ppp_job.sh)

Submit everything from the **repo root** (`/scratch/wmz2007/neuroinfo_fruitfly`).

---

## 1. Pipeline

Learning (PyTorch / gunpowder) and assembly (CUDA vote_instances) talk **only through Zarr/HDF on disk**.
One entry point, `PatchPerPix/experiments/run_ppp.py`, with `-d` tasks:

```text
train → predict → [decode if train_code] → label → evaluate
```

| Stage | Output |
|-------|--------|
| predict | Affinities / patches / numinst under `processed/` |
| decode | Patches → affinities (only when `train_code = true`) |
| label | Instance IDs in `instanced/.../*.hdf`, key `vote_instances` |
| evaluate | `evalinstseg` → TOML + `summary.csv` under `evaluated/` |

An array run (`--stage all`, the default) splits label into a GPU and a CPU job so the GPU isn't held for
CPU work:

```mermaid
flowchart LR
  pred[GPU predict] --> dec{train_code?}
  dec -->|yes| decode[GPU decode]
  dec -->|no| graph[GPU labgraph]
  decode --> graph
  graph --> cpuLabel[CPU label]
  cpuLabel --> eval[CPU evaluate]
```

Worked example, `train_code = false` preset (no decode), from the repo root:

```bash
./sbatch/ppp/ppp_submit_test.sh array our-noenc-c1k_v1 --base original_default.toml --ckpt 50000 --split test
```

1. **Predict (GPU).** Writes affinities and the instance-count prediction to
   `metrics/ppp/<exp_id>/test/processed/<ckpt>/<dataset>/<sample>.zarr`.
2. **Patch graph (GPU, job name `labgraph`).** Blockwise vote-instances builds the per-block and inter-block
   patch graph (`vi_split_gpu.toml`, `termAfterStitchGraph`), cached in
   `test/instanced/<ckpt>/.../<sample>.zarr` under `volumes/blocks`. It sets `stitch_graph_done` and stops
   before assigning instance ids. A retry skips blocks that already have `patch_pairs` and `aff_graph_mat`.
   Walltime 4 h. Performance write-up: [`ppp_labgraph_speedup.md`](ppp_labgraph_speedup.md).
3. **Label (CPU).** `vi_split_cpu.toml` reads that graph and runs `affGraphToInstances`. Task i waits only for
   GPU task i (`aftercorr`) and refuses to start without `stitch_graph_done`. Output:
   `test/instanced/<ckpt>/.../<sample>.hdf` (key `vote_instances`). Samples with a valid HDF are skipped when
   resume is on (`label_resume.toml`).
4. **Evaluate (CPU).** After every label task succeeds, one job scores the HDFs against GT (`eval_rm200.toml`
   drops components under 200 voxels). Per-sample log lines are `general.avg_f1_cov_score`. Table:
   `test/evaluated/<ckpt>/patch_threshold_0_3/fc_threshold_0_3/mws_True/skeletonize_foreground_True/numinst_threshs__0_9_0_1_/summary.csv`,
   copied to `summary_test.csv`. Volumes with no HDF are omitted. Evaluate then strips `volumes/pred_affs`
   and `volumes/pred_code` from the prediction Zarrs (`pred_numinst` is kept).

---

## 2. Data prep

Datasets live under `fisbe/<name>/{train,val,test}/` (Zarr). Names used by live presets: `completely`,
`partly`, `synth_vh`, `1k_v1`, `1k_v1_u8`.

Order when a dataset changes (details and every variable: [`ppp_prep.md`](ppp_prep.md)):

1. Dataset-specific build (synth from SquashFS, himanshu), if the zarrs don't exist yet.
2. PPP arrays (`raw_normalized`, `gt_instances_rm_5`, `gt_numinst`, `gt_fg_rm_5`):
   `prepare_fisbe_for_ppp_sbatch.sh`, or `pack_ppp_sqf_sbatch.sh` for large sqf-backed sets (`1k_v1`, `1k_v1_u8`).
3. FG point CSVs (`prepare_fg_points_sbatch.sh`), when the preset sets `sample_fg_points = true`.
4. Stage tars (`prepare_ppp_stage_tars_sbatch.sh`) for `STAGE_DATA=1` train jobs. Not for sqf-packed sets.

```bash
DATASETS=completely,partly,synth_vh sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh
SPLITS=completely/train,completely/val,completely/test sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh
DATASETS=completely,partly,synth_vh sbatch sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh
```

Inference-only data with no GT (e.g. himanshu): [`ppp_new_dataset_inference.md`](ppp_new_dataset_inference.md).

---

## 3. How to launch

Train and inference are separate commands. Both set SLURM resources and call `ppp_job.sh`. Presets and the
`sbatch` wrapper live in `ppp_submit_common.sh`; `ppp_submit.sh` only prints the two commands and exits.
`--dry-run` prints the `sbatch` line without submitting; `--help` lists every flag.

```bash
./sbatch/ppp/ppp_submit_train.sh [--dry-run] <preset> [flags…]
./sbatch/ppp/ppp_submit_test.sh [--dry-run] <mode> <preset> [flags…]
```

Logs: `sbatch/ppp/logs/<job_name>-<jobid>.{out,err}`. Array tasks:
`sbatch/ppp/logs/array/<preset>_<split>_<arrayjob>_<task>.{out,err}`.

### Train

One GPU job. It unpacks `fisbe/<dataset>/{train,val}_ppp.tar` into `$SLURM_TMPDIR` (`STAGE_DATA=1`), or mounts
the `fisbe/sqf/<dataset>/*.sqf` images for sqf-packed sets, and runs `-d train`. CPUs, memory, walltime and
partition come from the preset unless overridden.

```bash
./sbatch/ppp/ppp_submit_train.sh our-noenc-c1k_v1 \
  --base original_default.toml --id our-noenc-c1k_v1 --time 24:00:00
```

Common flags: `--base`, `--id`, `--config` (repeatable), `--datasets`, `--time`, `--partition`, `--cpus`,
`--mem`, `--account`, `--dependency`.

For H200, see [§6](#6-train-ops). Short smoke train: add `--config smoke.toml` (plus `smoke_noval.toml` when
timing) and `--time 00:30:00`.

### Resume and fine-tune

- **Resume** an experiment with the same `--id`. `train.py` picks up the latest checkpoint, and the run keeps the
  `config.toml` saved in its experiment dir, so newly added config defaults don't apply.
- **Fine-tune** a checkpoint on another dataset with
  [`exp_v1_finetune.sh`](../../sbatch/ppp/exp_v1_finetune.sh). It copies the source checkpoint (model + Adam
  state) into a new experiment, writes an overlay under `configs/exp/`, and resumes at iteration N to
  N + `FT_ITERS` (default 50000). Checkpoint numbers continue from the source.

  ```bash
  ./sbatch/ppp/exp_v1_finetune.sh [--dry-run] our-noenc-c1k_v1 our-noenc-c1k_v1 50000
  # → exp id exp3-noenc-c1k_v1-ft50000-completely
  ```

### Test

| Mode | When to use it |
|------|----------------|
| `array` | A whole val or test split. Default `--stage all`: GPU predict → [GPU decode] → GPU labgraph → CPU label → CPU evaluate. |
| `run` | One sample: predict[+decode]+label+evaluate. |
| `evaluate` | CPU rescore. `--split val` or `--split test`; omit `--split` (or pass `val,test`) to score both and write a merged `summary.csv`. |
| `ckpt-sweep` | Each checkpoint: predict → [decode] → label+evaluate. Skips a checkpoint whose prediction Zarr is complete. |
| `vi-sweep` | One sample, chained across VI overlays (`vi_th_0_3` … `vi_th_0_7` unless `--overlays`). |

```bash
./sbatch/ppp/ppp_submit_test.sh array our-noenc-c1k_v1 \
  --base original_default.toml --ckpt 50000 --split test

./sbatch/ppp/ppp_submit_test.sh run base1-noenc-c \
  --base original_default.toml --ckpt 400000 \
  --sample completely/R22C03-20180918_66_J2 --split val
```

Single array stages: `--stage predict`, `--stage decode` (only with `train_code`), `--stage label` (GPU graph
then CPU label, chains evaluate), or `--stage legacy` (predict[+decode]+label on one GPU; add `--agg` to chain
evaluate). Standalone evaluate: `./sbatch/ppp/ppp_submit_test.sh evaluate <preset> --split test --ckpt <N>`.
`--pct` is array concurrency (default 2). `--datasets completely` limits the array to `fisbe/completely/test`
when the preset also lists a synth test set.

Wait for a training job with `--dependency <train_job_id>`.

Other flags: `--vi` (default `vi_th_0_3.toml`), `--eval` (default `eval_rm200.toml`), `--do` (override `-d`
for `run`), `--keep-pred-affs`, `--keep-pred-code`, plus the train resource flags.

### Several checkpoints in a row (disk-limited)

`pred_affs` for one val/test pipeline is large and is only removed by evaluate. To validate many checkpoints
without filling `/scratch`, chain full pipelines so each predict waits for the previous evaluate:

```bash
./sbatch/ppp/ppp_submit_chain.sh [--dry-run] [--after JOBID] \
  <preset> <exp_id> <split> <ckpt,ckpt,...> [extra ppp_submit_test.sh flags…]

./sbatch/ppp/ppp_submit_chain.sh our-noenc-c1k_v1 our-noenc-c1k_v1 val 50000,40000 \
  --base original_default.toml --pct 3 --account torch_pr_61_tandon_advanced
```

It prints `ckpt=<N> evaluate=<jobid>` per pipeline and `chain_last=<jobid>`. Two chains in parallel keep at
most two pipelines' `pred_affs` on disk. A val pipeline takes ~2.5–3 h per checkpoint.

### Sample IDs and outputs

Sample IDs are always `{dataset}/{stem}` (e.g. `completely/R22C03-20180918_66_J2`); flat stems are rejected.

```text
metrics/ppp/<exp_id>/
  config.toml
  run.log                    # train / no --sample
  log/<dataset>/<stem>/run.log
  train/
  val/{processed,instanced,evaluated}/…
  test/{processed,instanced,evaluated}/…
```

Infer and array jobs omit `--test-data`, so the TOML `*_datasets` lists stay active. `--datasets A,B`
overrides them for that job; `run_ppp` errors if a listed directory is missing.

### Python helpers

The submitters call these themselves. Run one by hand only for the checks below.

| Script | What it does |
|--------|--------------|
| [`ppp_config_path.py`](../../sbatch/ppp/ppp_config_path.py) | Resolves bare config names to `setup01/configs/<role>/<name>.toml`. `python3 sbatch/ppp/ppp_config_path.py NAME...` checks a stack. |
| [`needs_decode.py`](../../sbatch/ppp/needs_decode.py) | Prints `1` when the merged TOMLs set `training.train_code`; inserts the decode stage. |
| [`ppp_datasets.py`](../../sbatch/ppp/ppp_datasets.py) | Dataset names and `{dataset}/{stem}` lists from the merged TOMLs. Sizes the SLURM array. |
| [`check_prediction_complete.py`](../../sbatch/ppp/check_prediction_complete.py) | A prediction is finished only when every chunk is on disk. Array predict drops incomplete volumes; label/decode-only jobs refuse them; `ckpt-sweep` skips finished checkpoints. |
| [`merge_ppp_summary_csvs.py`](../../sbatch/ppp/merge_ppp_summary_csvs.py) | Concatenates `summary_val.csv` and `summary_test.csv` and recomputes the mean/sum rows. |
| [`strip_pred_affs.py`](../../sbatch/ppp/strip_pred_affs.py) | Deletes `volumes/pred_affs` and `volumes/pred_code` after evaluate (keeps `pred_numinst` and two named keep-samples). |

```bash
python3 sbatch/ppp/ppp_datasets.py --split test --count --datasets himanshu_20x \
  original_default.toml datasets_test_himanshu_20x.toml

python3 sbatch/ppp/strip_pred_affs.py \
  --root metrics/ppp --exp <exp_id> --split test --ckpt <N> --dry-run
```

---

## 4. Presets and config stacking

Always pass `--base` explicitly. The submitter fallback (`default_l40s.toml`) is **not** the canonical base.

| Preset | `--base` | Train / val+test data | Notes |
|--------|----------|-----------------------|-------|
| `our-noenc-c1k_v1` | `original_default.toml` | `1k_v1` / `completely` | noenc, fmaps20, `fast_affinities`. experiment_v1 model. |
| `our-noenc-c1k_v1_u8` | `original_default.toml` | `1k_v1_u8` / `completely` | noenc, fmaps24, `fast_affinities` |
| `our-decoder-c1k_v1`, `_u8` | `original_default_train_code.toml` | `1k_v1` (`_u8`) / val `completely`, test same synth set | PPP+dec, fmaps20, bs30; preset defaults to H200 |
| `our-noenc-cs` | `original_default.toml` | `synth_vh` / `completely`+`synth_vh` | noenc, fmaps24, 400k |
| `base1-noenc-c` | `original_default.toml` | `completely` / `completely` | noenc baseline, fmaps24, `fast_affinities` |
| `base1-decoder-c` | `original_default_train_code.toml` | `completely` / `completely` | PPP+dec baseline |
| `basic-ppp-noenc` | `original_default.toml` | `completely` / `completely` | noenc, fmaps24, 400k, all augs (cheapened) |
| `basic-ppp-noenc-cp` | `original_default.toml` | `completely`+`partly` / same | noenc, fmaps24, 400k |
| `basic-ppp-decoder` | `original_default_train_code.toml` | `completely` / `completely` | PPP+dec, 400k |

`PRESET_TABLE` (in `sbatch/ppp/ppp_submit_common.sh`) holds exactly these presets. The old rows `basic_8h`,
`basic_long8h`, `basic_fmaps24*` and `basic-ppp-noenc-cps` were removed on 2026-11-06 because their overlays no
longer exist. An unknown preset now prints the list of valid names. Adding a preset means adding the overlay
to `configs/presets/` and adding a row to the table.

Configs merge left to right (later wins):

```text
--base  →  preset overlay  →  --config … (data / train / exp overlays)  →  --vi / --eval  →  CLI flags
```

Configs live in role folders under `setup01/configs/` (`base/`, `presets/`, `data/`, `train/`, `exp/`, `vi/`,
`eval/`, `archive/`). Pass **bare file names**; `ppp_config_path.py` finds the one match (`archive/` is
skipped). The folder-by-folder catalog is
[`configs/README.md`](../../PatchPerPix/experiments/flylight/setups/setup01/configs/README.md).
The per-key dictionary is the archived [`ppp_config_defaults.md`](historical/ppp_config_defaults.md) (written
against an older defaults file; trust the live TOMLs for values).

---

## 5. Evaluation protocol

PPP doesn't score itself: `evaluate` calls [`evaluate-instance-segmentation`](../../evaluate-instance-segmentation/)
(`evalinstseg`). Matching is by **clDice overlap**, not raw ID equality, and assignment for official runs is
**greedy**, not Hungarian.

| Symbol | Meaning |
|--------|---------|
| avF1 | Mean F1 over clDice thresholds after 1-to-1 assignment |
| C | Mean clRecall coverage of GT by preds |
| S (`avg_f1_cov_score`) | 0.5 · avF1 + 0.5 · C |
| FS / FM | False split / false merge |
| skel cov | Skeleton coverage on matched pairs |

Size filters (don't confuse them):

| Knob | Where | Role |
|------|-------|------|
| `vote_instances.ignore_small_comps` | stitch FG clean | Drop tiny FG components before the graph |
| `vote_instances.remove_small_comps` | post-label | Drop tiny instances in stitch |
| `evaluation.remove_small_components` | eval preprocess | Drop tiny preds **before** metrics |

While debugging instance quality, prefer a soft eval overlay (`eval_rm0` / `eval_rm200`). An aggressive rm500
can hide over-fragmentation and inflate F1. Metric walkthrough: archived
[`ppp_evaluation.md`](historical/ppp_evaluation.md).

---

## 6. Train ops

- **Container:** `singularity exec --nv --overlay env/ppp.ext3:ro` + CUDA sif, then
  `source /ext3/env.sh; conda activate ppp`. Pin `pycuda<=2024.1.2` in the overlay (env is Python 3.9).
- **Staging:** train sets `STAGE_DATA=1` and unpacks `*_ppp.tar` into `$SLURM_TMPDIR`; sqf-packed datasets are
  mounted instead.
- **Prefetch:** `ThreadedPrefetch` after `gp.Stack`, TOML `prefetch_batches` (default 2). It fixes batch-handoff
  idle; more workers alone don't.
- **FG sampling:** with `sample_fg_points = true`, prepare FG CSVs first. That beats integral-mask
  RandomLocation (long 0% GPU startup). `oversample_complete` multiplies mask-setup cost.
- **GPU util:** public partitions cancel low-util jobs. `ppp_job.sh` backgrounds `nvidia-smi … -l 1` into
  `gpu_log/ppp/gpu_usage_log_<exp_id>_<jobid>.csv`. Don't put `@fork` on train.
- **Train/val curves:** TensorBoard is not in the overlay, so curves come from text files. Train loss is the
  `Train process: iteration=N loss=…` lines in `<exp>/run.log`. Every `val_log_step` iterations
  (presets: 2000, so val points line up with checkpoints) `setup01/validation.py` evaluates a fixed val subset
  (`val_num_batches` x `batch_size` crops, seeds `val_seed + i`, cached in RAM, ~5 s per val point after a ~1 min
  first load) and appends to `<exp>/train/val_metrics.csv` plus a `Validation: iteration=N …` line in `run.log`.
  Columns: `loss`/`loss_patch`/`loss_fg`; fg `dice`/`iou`/`precision`/`recall`; `numinst_acc_fg`;
  `overlap_precision`/`overlap_recall`; `aff_acc`/`aff_precision`/`aff_recall`/`aff_f1` (on gt-fg voxels). These are
  voxel-level only; instance metrics (avF1, C, FM, FS, …) still come from `-d validate_checkpoints`.
  Plot with `ipynb/ppp_training_log_tracker.ipynb` or
  `python PatchPerPix/experiments/ppp_train_track.py metrics/ppp/<exp>/run.log --val --val-plot out.png`.
  Runs from before 2026-11-06 logged one random val batch as an ordinary `Train process` line at
  `iteration % val_log_step == 2` (a spike in the train curve); pass `legacy_val_log_step=2000`
  (`--legacy-val-log-step 2000`) to split those off.
- **H200:** `h200_public` allows at most **16 CPUs per GPU**. For noenc, stack `exp_v1_h200.toml` (14 workers,
  fp16 raw) and pass `--partition h200_public --cpus 16`: 0.63–0.66 s/iter.
- **L40S:** stack `compile.toml` (`torch.compile` of UNet + heads + loss, noenc only): 1.45 s/iter vs 1.63
  eager (2.24 before `fast_affinities`). The first iteration compiles for ~2 min. `channels_last.toml` on top
  gives 1.36 s/iter but peaks at 45.1 of 46.1 GB.
- **Speed numbers, settings per GPU, and how to diagnose a slow step:**
  [`ppp_train_performance.md`](ppp_train_performance.md).

---

## 7. Quality strategy

1. **Prediction quality first.** Weak `pred_affs` / `pred_numinst` won't be fixed by VI merge knobs. Check
   mid-aff FG metrics and MIPs before long sweeps.
2. **Checkpoints.** `last` ≠ best; sweep or pick from val logs.
3. **VI overlays.** Knob changes must appear in the validation product path, or they overwrite the baseline HDF.
   `DO_TASKS` in SLURM `--export` must use `+`, not commas.
4. **Dead keys.** `do_nms` and `affinity_graph_voting*` are unused; changing them is a no-op.
5. **Report** Num Pred/GT, soft F1 / S, skel coverage, FS/FM, and a MIP, not a single F1 under a hard size
   filter. A Num Pred ≫ Num GT with near-zero avF1 means over-segmentation (experiment_v1: ~530–600 preds
   for ~5 GT per volume).
