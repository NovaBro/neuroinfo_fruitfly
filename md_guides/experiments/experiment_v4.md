# experiment_v4: Synthetic Data Scaling (simple_v1)

Plan written 2026-11-18; implementation approved the same day; launched 2026-11-18 11:26 (see "Phase 0 and launch"). Status page
`md_guides/experiments/expv4_status.md` gets created at launch.

## Goal

Two questions about the new synthetic dataset `simple_v1`
([`simple_dataset_plan.md`](../dataset/simple_dataset_plan.md): 50,000 / 100 / 100 samples of branching tubes,
400×600×600, 4-bit raw, haze + distractor blobs baked in, confetti added on the GPU in training):

1. **Effectiveness.** Does pretraining on simple_v1 followed by an encoder-only LoRA finetune on completely beat
   training on completely alone?
2. **Scaling.** How does the result change with the number of synthetic training samples,
   N ∈ {1k, 5k, 10k, 50k}?

## Decisions (user, 2026-11-18)

| Question | Decision |
|---|---|
| Model | noenc PatchPerPix, `num_fmaps = 15` (`expv2_e1_common.toml`) |
| Pretrain | 50k iterations, peak lr 4e-4, linear warmup 10 % + cosine to 0 |
| Finetune | 6k iterations, peak lr 5e-5, linear warmup 10 % + cosine to 0 |
| Finetune method | LoRA on the **encoder only** (`targets = ["l_conv"]`, which includes the bottleneck), rank 4 (alpha 4), output heads trainable, decoder and upsamplers frozen |
| Scaling arms | N = 1,000 / 5,000 / 10,000 / 50,000 simple_v1 train samples |
| Baseline | No pretraining: train on completely from scratch |
| Data mixing | None. Pretraining sees only simple_v1, finetuning (and the baseline) only completely. |
| Tracking | Completely **and** synthetic validation loss and metrics at every checkpoint, incl. `fg_dice`, `fg_precision`, `fg_recall` |
| Selection | **Completely val only:** highest mean of `fg_dice`, `fg_precision`, `fg_recall` (see "Selection rule") |
| Baseline schedule | 50k its, peak 4e-4, same warmup + cosine as pretraining |
| Checkpoint / val interval | Every 2.5k for pretraining and the baseline (20 points); every 1k for finetuning |
| Tests | After pretraining: the selected pretrain checkpoint on **simple_v1 test**. After finetuning: the selected finetune checkpoint on **simple_v1 test and completely test**. The baseline is tested like a finetuned model (both). simple_v1 test = its **first 5 volumes** (`max_test_samples = 5`). |

## Arms

| EXP | Stage 1 | N | Stage 2 | Tests |
|---|---|---|---|---|
| 4.0 | train on completely from scratch (no pretraining) | – | – | best of 4.0 on simple_v1 + completely |
| 4.1 | pretrain on simple_v1 | 1,000 | LoRA finetune on completely (4.1F) | best of 4.1 on simple_v1; best of 4.1F on simple_v1 + completely |
| 4.2 | pretrain on simple_v1 | 5,000 | LoRA finetune on completely (4.2F) | same pattern |
| 4.3 | pretrain on simple_v1 | 10,000 | LoRA finetune on completely (4.3F) | same pattern |
| 4.4 | pretrain on simple_v1 | 50,000 | LoRA finetune on completely (4.4F) | same pattern |

Test chains: 4 pretrain × simple_v1, 4 finetune × (simple_v1 + completely), baseline × (simple_v1 + completely)
= 9 simple_v1 chains (first 5 test volumes, `tube_50100`–`tube_50104`) and 5 completely chains (7 volumes
each).

Exp ids: `expv4-b0-completely`, `expv4-pre-n{1k,5k,10k,50k}`, `expv4-ft-n{1k,5k,10k,50k}`.

The N subsets are nested: `max_train_samples = N` keeps the first N sorted train files, i.e.
`tube_00000` … `tube_{N−1}`. All arms run the same 50k iterations, so smaller N repeats each volume more often
(N = 1k: ~900 crops per volume at batch 18; N = 50k: ~18).

## Common setup

- **Base / preset:** `original_default.toml` + `our-noenc-c1k_v1.1.toml` (140³ valid crops, batch 18,
  fast affinities, sampling on fg points), + `expv2_e1_common.toml` (fmaps 15, `raw_transport_fp16`, `compile`).
  The preset caps `max_train_samples = 1000`; every arm sets its own value.
- **Pretrain 4.1–4.4:**
  - train `simple_v1` (first N), GPU confetti on (`dsexp_simple_v1.toml`, `[training.augmentation.gpu_confetti]`);
  - 50k its, peak 4e-4, WarmupCosine (warmup 0.10, min ratio 0), as `dsexp_pre.toml`;
  - val on `simple_v1` val, first 5 (`val_metrics.csv`), plus `completely` val, 5 volumes (extra val,
    `val_metrics_completely.csv`);
  - checkpoint + val every **2.5k** (20 points; `dsexp_pre.toml` uses 5k).
- **Baseline 4.0:** train `completely` (18 volumes), same 50k / 4e-4 schedule and interval, val `completely`
  (`val_metrics.csv`) plus `simple_v1` first 5 as extra val (`val_metrics_simple_v1.csv`). No confetti.
- **Finetune 4.1F–4.4F:**
  - seed from the selected pretrain checkpoint with `sbatch/ppp/expv2/seed_finetune_ckpt.py`
    (`--lr 5e-5 --warmup-frac 0.1 --ft-iters 6000`), as `expv3.py start_finetune`;
  - `[training.lora]`: `enabled = true`, `rank = 4`, `alpha = 4.0`, `targets = ["l_conv"]`,
    `train_heads = true`, `train_bias = false` (`setup01/lora.py`);
  - train `completely` (18) only, 6k its, peak 5e-5, WarmupCosine (`start_iter` = seed iteration), as
    `dsexp_ft.toml`;
  - val `completely` 5 (`val_metrics.csv`, selection) + `simple_v1` first 5 as extra val; checkpoint + val
    every 1k;
  - no confetti: the finetune stack must **not** include `dsexp_simple_v1.toml` (its `gpu_confetti` section
    refuses non-simple_v1 train data and would stop the job).
- **Tests** (sparse prediction pipeline, `pred_sparse_affs.toml`; standard summary metrics S, avF1, C,
  clDiceTP, tp, FS, FM, nPred / nGT):
  - **completely test** (7 volumes), `remove_small_components = 1500` as in all earlier experiments: selected
    checkpoints of 4.0 and 4.1F–4.4F;
  - **simple_v1 test**, first 5 of the 100 volumes (confetti baked in), `remove_small_components = 500` (the eval default;
    every simple_v1 neuron has ≥ 500 voxels by construction, so 1500 would delete true small neurons): selected
    checkpoints of 4.1–4.4 (pretrain), 4.1F–4.4F and 4.0. `predict_no_gp.py` reads the uint8 raw directly and
    evaluation reads `gt_instances_rm_5` (a link to `gt_instances`); `run_ppp.py` knows `simple_v1`.
- **Account:** `torch_pr_61_tandon_priority`; L40S (`l40s_public`).

## Selection rule

```
score(ckpt) = mean(fg_dice, fg_precision, fg_recall)   on completely val; NaN counts as 0
best        = highest score; ties -> lower completely val loss, then the later iteration
```

- **Pretrain 4.1–4.4:** read from `val_metrics_completely.csv` (completely is the extra val set there); the best
  checkpoint seeds the finetune.
- **Finetune 4.xF and baseline 4.0:** read from `val_metrics.csv`; the best checkpoint is tested.
- Synthetic val loss and metrics are recorded at every checkpoint and reported next to the score, but never
  used to select.
- `fg_precision` is NaN when a checkpoint predicts no foreground (seen at iteration 20 of the simple_v1 smoke);
  counting it as 0 keeps such checkpoints from winning.
- Completely val uses crop set A (checksum line in each train log). Record it per run; scores are only
  comparable when it matches.

## Reporting

`expv4_status.md`, one row per arm:

- pretrain: best iteration, score, its three components, completely val loss, simple_v1 val loss and fg metrics
  at that iteration;
- finetune: the same for the finetune checkpoints;
- tests: S, avF1, C, clDiceTP, tp, FS, FM, nPred / nGT on each test set the run gets.

Final figures:
- **Scaling:** completely test S and avF1 of 4.1F–4.4F against N (log axis), with baseline 4.0 as a horizontal
  line.
- **Synthetic skill:** simple_v1 test S / avF1 after pretraining vs after finetuning, per N (how much the
  encoder-only LoRA finetune forgets), with 4.0 as the no-synthetic reference.
- **Curves:** completely and simple_v1 val score / loss per arm over iterations.
- **Transfer:** pretrain completely-val score against N (how far synthetic-only training transfers before
  finetuning).

## To build (show the change list before coding)

1. **Phase 0: data pipeline at 50k files.** `train.py get_sources` opens every train zarr and builds a
   `ZarrSource` + `CsvPointsSource` (+ `Pad`, `RandomLocation`, …) per file. It has only ever run with ≤ ~1k files.
   - Smoke (40 its, `dsexp_smoke.toml`) with `max_train_samples = 50000` and with `1000`.
   - Measure time to the first iteration, host memory (job RSS; PreCache forks 28 workers) and s/it.
   - Go: startup < ~30 min and memory within the job's 250 GB.
   - No-go: propose a lazy source (one node that opens a random zarr per request). That is a `train.py` change
     with its own approval, and it must not alter the completely val pipeline (val-crop checksum).
2. **Driver** `sbatch/ppp/expv4/expv4.py` (+ `expv4_tick.sh`, state `metrics/ppp/expv4/state.json`, status
   page). Copy of the `expv3.py` lane pattern:
   - submit pretrain / baseline;
   - select with the score above (replaces `select_best`, which takes the lowest val loss);
   - seed and submit the finetune (`start_finetune`);
   - chain the tests (`ppp_submit_test.sh`): simple_v1 after pretrain selection, both sets after finetune
     selection;
   - write the status table.
3. **Overlays** `configs/exp/expv4/`:
   - `expv4_pre.toml`: `dsexp_pre.toml` settings with checkpoints / `val_log_step` = 2500.
   - `expv4_n{1k,5k,10k,50k}.toml`: `max_train_samples`.
   - `expv4_b0.toml`: completely train/val, `extra_val_datasets = ["simple_v1"]`.
   - `expv4_ft.toml`: `dsexp_ft.toml` settings, `extra_val_datasets = ["simple_v1"]`, the `[training.lora]` block.
   - Per finetune, a generated overlay with `max_iterations` / `start_iter`.

   Stacks:

   | Run | Stack (after `original_default.toml`) |
   |---|---|
   | pretrain | `our-noenc-c1k_v1.1` + `expv2_e1_common` + `expv4_pre` + `dsexp_simple_v1` + `expv4_n<N>` |
   | baseline | `our-noenc-c1k_v1.1` + `expv2_e1_common` + `expv4_pre` + `expv4_b0` |
   | finetune | `our-noenc-c1k_v1.1` + `expv2_e1_common` + `expv4_ft` + generated |
4. **LoRA check:** `sbatch/ppp/tools/test_lora.py` with rank 4 and `targets = ["l_conv"]`, to record the
   trainable parameter count and confirm the decoder stays frozen.

## Compute

| Stage | Per run | Runs | GPU-h |
|---|---|---|---|
| Pretrain (L40S ~1.2 s/it incl. 4.7 ms confetti; + 20 × 2 val passes) | ~17–18 h | 4 | ~70 |
| Baseline 4.0 | ~15–17 h | 1 | ~16 |
| Finetune (6k its + 6 val passes) | ~2 h | 4 | ~8 |
| Completely test chains (7 volumes) | ~1–2 h | 5 | ~8 |
| simple_v1 test chains (5 volumes) | ~1 h | 9 | ~9 |
| **Total** | | | **~110** |

- Run time: `--time 24:00:00` for pretrain and baseline.
- Parallelism: the five stage-1 runs go in parallel. The per-user cap (16 GPUs, `QOSMaxGRESPerUser`) is shared
  with any running expv3 jobs.

## Launch order

1. Phase 0: smoke at N = 50k (go / no-go); smoke of the finetune (LoRA encoder) and of a simple_v1 test
   (uint8 raw through prediction, evaluation through the links) on a smoke checkpoint.
2. LoRA check.
3. Driver dry run.
4. Launch 4.0–4.4 together.
5. Each finetune starts when its pretrain is selected.
6. Tests chain after each selection (pretrain: simple_v1; finetune and baseline: simple_v1 + completely).
7. Write the report.

## Risks and caveats

- **Pipeline startup at 50k files** (Phase 0) is the main unknown.
- **One seed per arm.** Earlier sweeps differed by ~±0.02 in S between similar arms (OP-EXP 6/7), so smaller
  gaps between N values may be noise.
- **5-volume completely val** makes selection noisy; the score mixes three thresholded fg metrics and ignores
  instance separation (patch loss), which the test measures.
- **LoRA rank 4 at 5e-5:** OP-EXP 6/7 ran LoRA at 5e-5 to 8e-4; the `lora_r8.toml` notes say LoRA usually
  wants 5e-4 to 1e-3. An encoder-only rank-4 adapter at 5e-5 may barely move. The completely-val curves of 4.xF
  will show it; a higher-lr follow-up would be a separate arm.
- **Baseline length:** training 50k its on 18 volumes overfits (EXP 0.1 peaked at 10k of 35k); selection
  picks the early checkpoint, so the baseline is not disadvantaged.
- **Confetti differences:** drawn per 140³ crop during training, per volume in the baked val/test samples
  (parity checked, `check_gpu_confetti.py`).

## Phase 0 and launch (2026-11-18)

- **N = 50k startup (no-go with the original loader):** job 23600906 loaded 50,010 zarrs in ~10 min, then built
  one RandomLocation per file at ~715 files/min (~85–90 min before the first iteration, 13.7 GB RSS). Cancelled.
  User: fix the loader, then launch all arms on the same code.
- **Lazy training sources** (`setup01/lazy_sources.py`, `[data] lazy_sources = true` in `expv4_pre.toml` and
  `expv4_ft.toml`; `lazy_max_num_inst = 30` in `expv4_n*.toml`): one node picks a file uniformly at random per
  request and builds that file's existing chain on first use (LRU cache per process). Training only; the
  validation sources are built as before. Smoke 23604600 (N = 50k): first iteration after ~1.5 min (mostly
  torch.compile), 1.40 s/it, the same as the old loader at N = 1k (1.41 s/it). Val crop checksums unchanged:
  simple_v1 5.597435e+06, completely 3.516400e+06. Baseline stack smoke 23604601 OK (scan: 12 instances max).
- **Finetune smoke** (23601725, from the N = 1k smoke @ 40): LoRA rank 4 on 8 convs under `l_conv` only, 171,948
  trainable params (90,984 LoRA + heads) = 0.71 % of 24.1 M; ~1.0 s/it; `val_metrics.csv` (completely) and
  `val_metrics_simple_v1.csv` written; the driver's selection reads both.
- **Selection logic** checked on smoke CSVs: NaN precision counts as 0; ties go to the lower completely loss.
- **Test path:** prediction on uint8 simple_v1 test volumes works (5 × 7–11 min). A barely trained smoke model
  predicts no foreground, so evaluate finds no instances (pre-existing pipeline behaviour); the evaluation
  check uses the expv2 EXP 1.3 checkpoint copied into `metrics/ppp/expv4-smoke-evalcheck`.
- **Launch 11:26:** train jobs 4.0 23605206, 4.1 23605210, 4.2 23605212, 4.3 23605213, 4.4 23605215
  (tandon_priority, 24 h); tick loop `sbatch/ppp/expv4/expv4_tick.sh` (every ~10 min, stop: `scancel -n
  expv4_tick`). Status: [expv4_status.md](expv4_status.md).

## Interim notes (2026-11-19 09:20)

- **Test-path check passed:** expv2 EXP 1.3 @ 23000 copied to `expv4-smoke-evalcheck`, tested through the driver:
  simple_v1 (first 5) S 0.589, then completely S 0.3045 (n = 7), matching its known rm1500 result (0.305). The two
  tests of one checkpoint do not mix.
- **Synthetic-only pretrains barely transfer to completely:** across all 20 checkpoints of every N, completely
  val dice 0.12–0.22 at recall 0.06–0.12 and precision ≈ 0.997; completely val loss rises with training. The
  selection score (mean of dice / precision / recall) is 0.39–0.44 everywhere, propped up by precision, so the
  selected pretrain checkpoints differ from the rest by noise-level margins: 4.1 @ 27.5k, 4.2 @ 30k,
  **4.3 @ 2.5k**, 4.4 @ 22.5k. Baseline 4.0 reaches 0.748 (dice 0.729) at 15k.
- **4.1F** (encoder LoRA r4, 5e-5): best 0.576 (dice 0.502, recall 0.35) at 33k, well below the baseline.

## Accounts (2026-11-19 13:15)

User: use `general` and `tandon_advanced` to finish sooner. Slurm refuses in-place account changes for users
(`scontrol update ... Account=` → "Unspecified error"), so the driver now alternates new submissions over
`ACCOUNTS = [tandon_advanced, general]` and `expv4.py move` cancelled + resubmitted the 7 test pipelines whose GPU
work had not started (4.2/4.3/4.4-Ts from `--stage label`, predictions kept; 4.1F/4.2F/4.3F/4.4F-Ts in full).
4.0-Tc and 4.1-Ts were partly done and stay on tandon_priority. Priority after the move: 14,100
(tandon_advanced) / 13,280 (general) vs ~12,130 (tandon_priority).

## Open questions

None (resolved 2026-11-18, see Decisions). One default set by the plan, not by you:
`remove_small_components = 500` on simple_v1 test (1500 on completely test as before).
