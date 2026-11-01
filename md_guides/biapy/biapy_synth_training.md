# Experiment: Effect of staged synth training.

# Exp1: Baseline
- Train baseline setup on completely dataset, and test on completely dataset

# Exp2: Staged Training
- Pre-trian on 1k_v1_u8 dataset, and then fine-tune on completely.
- Test on the test splits for 1k_v1_u8 and completely
- Concerns, potentially takes a long time. See what optimizations, or methods can be done to do training in less than 24 hours
- If training does go fast enough, may not train on all images fully, and lr scheduler may not complete

# Exp3: 
- Train on 1k_v1_u8 dataset
- Test on the test splits for 1k_v1_u8
- Concerns, potentially takes a long time. See what optimizations, or methods can be done to do training in less than 24 hours
- If training does go fast enough, may not train on all images fully, and lr scheduler may not complete

---

# Implementation details

Everything below is measured from existing logs unless marked *verify*.
Model/config for all three experiments is the raytune winner
`biapy_work_folder/configs/biapy_fcdcdn_skel_l40s_c3_winner.yaml` (seunet, F+Dc+Dn, skeleton recall,
patch `(80,128,128,3)`, batch 12, AdamW, one-cycle, 240 epochs). Only data, schedule length, and checkpoint
loading differ between experiments.

## 0. What the logs say about time (the real constraint)

| | completely (train) | 1k_v1_u8 (train) |
|---|---|---|
| Patches per epoch (BiaPy "Loaded train data shape") | 4,169 | 18,000 |
| Iterations per epoch (batch 12) | ~348 | 1,500 |
| Iteration time, L40S | ~1.0 s (winner run on FDb: 0.65-1.03 s) | 1.005 s |
| Full 240 epochs | ~84k iters, **~23-24 h** | 360k iters, **~100 h** |
| Val pass | 134 patches, seconds | 3,600 patches, ~1.5 min/epoch (0.31 s/it) |

- The existing 1k run (job 21468397) hit the 24 h limit at **epoch 54 of 240**, with the one-cycle LR still at
  2.8e-4 (peak 3.3e-4). Its `checkpoint-best.pth` is a mid-schedule, high-LR checkpoint and should not be used
  for any reported number. Treat that run as a throughput measurement only.
- GPU utilization in that run averaged 98% (median 100%, 1.3% of samples under 30%). The job is **GPU-bound, not
  input-bound**, so data-loader tuning will not help. Speed comes only from fewer iterations or cheaper
  iterations.
- Staging train+val to `$SLURM_TMPDIR` took 641 s (~11 min) and used ~25 GB of the 800 GB node SSD. Disk is not
  a problem; the 11 min is a fixed cost per train job.
- `metrics/biapy/` currently has only the 1k directory. **No completely baseline checkpoint exists** in this
  tree, so Exp1 has to be (re)run.

**Consequence:** plan in **iterations**, not epochs. A 24 h job at ~1.0 s/it is ~80k iterations after staging,
validation, and test-time slack. Everything below is sized against that number.

## 1. Rules shared by all three experiments

1. **Always let the one-cycle schedule finish.** Set `TRAIN.EPOCHS` to what fits in the walltime. Never rely on
   a 240-epoch schedule being cut off. BiaPy's one-cycle is a function of `EPOCHS`, so shortening epochs
   compresses the schedule and it completes (this is the "lr scheduler may not complete" concern, solved by
   choosing EPOCHS up front rather than by early stopping).
2. **Same preprocessing, same normalization, same watershed/post-processing across experiments** (see section 6).
3. **Report the final-epoch or best-val checkpoint by one fixed rule** (best val loss, as BiaPy saves it). Val
   loss for synthetic and for completely are different scales; pick the val split of the dataset being trained
   on at that stage.
4. **Seeds.** `SYSTEM.SEED` exists (default 0). Leave it identical across arms. With 7 completely test volumes,
   one run per arm is still a noisy comparison, so report per-volume deltas, not just means, and say so.

## 2. Exp1 - Baseline (completely only)

Goal: the reference number for completely test.

- **Config:** copy of the winner, unchanged data paths (`fisbe/biapy-no-aug-zarr`). Name it
  `biapy-exp1-baseline-completely.yaml` (job name = YAML stem, so outputs go to a clean
  `metrics/biapy/biapy-exp1-baseline-completely/`).
- **Budget:** full 240 epochs is ~84k iters, ~23.5 h plus ~11 min staging. That is too close to the 24 h
  limit on an L40S. Two safe options, pick one and write it down:
  - (a) Run on **H100/H200** (the chain script has these constraints commented out; throughput on them is
    *unmeasured*, so run a 3-epoch speed probe first), keep 240 epochs.
  - (b) Keep L40S and set `TRAIN.EPOCHS: 200` (~19.5 h). Also confirm the earlier short runs
    (e.g. the 40-epoch raytune trials) are not the config you actually want to compare against.
- **Preprocessing:** `fisbe/biapy-no-aug-zarr` already has the `label_F…_Sk*_I` caches (the 1k run staged them
  fine), so **skip preprocessing**.
- **Commands (repo root):**

  ```bash
  ./sbatch/biapy/biapy-py_sbatch_chain.sh biapy-exp1-baseline-completely train test
  ```

- **Test:** completely test (7 vols, `IN_MEMORY: true`), scored with
  `evaluate-instance-segmentation/biapy_eval.py`.

## 3. Exp3 - Synthetic-only (1k_v1_u8), and it doubles as Exp2 stage 1

**Exp3 and Exp2's pre-training stage are the same training run.** Train once on 1k_v1_u8, test it on the
1k test split (Exp3), and also use the same checkpoint as the starting point for Exp2. This halves the
compute. Call it the "pretrain run".

### 3.1 Choosing the budget

At 1,500 iters/epoch the schedule cannot be 240 epochs. Options:

| Choice | Epochs | Iterations | Wall time (L40S) | Passes over the 18k patches |
|---|---|---|---|---|
| Full-ish | 50 | 75k | ~21 h + val ~1.2 h = ~22.5 h | 50 |
| Safer | 40 | 60k | ~16.7 h + val ~1 h | 40 |
| Quick probe | 10 | 15k | ~4.2 h | 10 |

Recommendation: **EPOCHS 40, WARMUP_COSINE_DECAY_EPOCHS 4** (the copied 20 would be half the schedule). Run a
10-epoch probe first (same YAML with `EPOCHS: 10`) to confirm the loss trajectory and the throughput before
spending a full day.

Why not use all 797 volumes "fully"? 40 epochs x 18k patches is already 720k patch presentations on a dataset
where the 18k patches are random crops of 797 volumes, so each volume is seen ~900 times. More epochs on the
same data is the likely cause of diminishing returns, not a lack of data coverage. A short, complete schedule
is the better trade than a 240-epoch schedule that is cut off at epoch 54.

### 3.2 Cheaper iterations (try in this order; measure each with a 3-epoch probe)

The job is GPU-bound at 98%, so only these help:

1. **Faster GPU.** `--constraint='h100|h200'` instead of `l40s`. Expected gain is large for this conv-heavy
   3D U-Net but is not measured here; probe first. Check the account/queue; the chain script lists the h200
   lines commented out.
2. **Mixed precision / TF32.** `BiaPy-novabro/biapy/engine/train_engine.py` has no `torch.autocast` or
   `GradScaler` (the only "GradScaler" hit is a comment), so training runs in **fp32**. This is the largest
   untried lever, and the GPU is already saturated, so it directly shortens iterations. BiaPy is the in-repo
   fork `BiaPy-novabro/`, so the change is a code edit, not a YAML key. Cheapest first step:
   `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` (what PPP's
   `train.py` does for H100/H200), then bf16 autocast around the forward pass and loss. Test the
   skeleton-recall and Dc/Dn L1 losses under bf16, because they have sqrt/log terms (the engine already guards
   non-finite gradients). Any precision change must be applied to **all three experiments**, or Exp1 is no
   longer comparable.
3. **Smaller per-step cost.** Lower `DATA.PATCH_SIZE` in Z (e.g. `(48,128,128,3)`) or `FEATURE_MAPS`. This
   changes the model and breaks checkpoint compatibility with Exp1 (BiaPy checks patch size when loading), so
   it is **not recommended** for this experiment.
4. **Cut validation cost.** Val is 3,600 patches, ~1.5 min per epoch (~6% of the time). Use a subset of the val
   split for the pretrain run (e.g. first 30 of 110 volumes, a copy of the val tree). This saves ~1 h of a
   40-epoch run.
5. **Subsample training volumes** (e.g. 400 of 797) instead of fewer epochs. Equivalent compute, but worse
   diversity per iteration than fewer passes over everything, so prefer fewer epochs.

Not worth doing: more dataloader workers (GPU already saturated), moving data to local SSD (already done).

### 3.3 Preprocessing and data

- `fisbe/biapy-1k_v1_u8-zarr/{train,val}` and the `label_F…_Sk*_I` caches **already exist** (train/val dirs show
  the cache folder). Test has no cache (not needed for inference).
- The prep script is `sbatch/biapy/biapy_prep_1k_sbatch.sh`; it uses `--unique-ids`, so the binary-mask issue is
  already handled for 1k. Do not re-run it.
- Train with `SRC_ROOT` pointed at the 1k tree. The sbatch now takes it from the environment:

  ```bash
  SRC_ROOT=fisbe/biapy-1k_v1_u8-zarr \
    ./sbatch/biapy/biapy-py_sbatch_chain.sh biapy-exp3-pretrain-1k train test
  ```

  The YAML is `biapy-1k_v1_u8-seunet-FDcDn-skel.yaml` copied to
  `biapy-exp3-pretrain-1k.yaml` with the epochs/warmup changes above.
  **Warning:** the chain script exports nothing about `SRC_ROOT`; forgetting it trains on completely while the
  YAML says 1k. Check the log line `Training on PATH=` at the start of every job.
- Exp3 test uses the 1k test split with `IN_MEMORY: false` (95 volumes).

### 3.4 Scoring the 1k test split

`biapy_eval.py` always reads GT from `fisbe/completely/{split}`, so it cannot score this. Needs a small change
or a wrapper: GT folder `fisbe/1k_v1_u8/test`, key `volumes/gt_instances`, and the container must mount
`/scratch/vh2308/data/synth_MCFO/1k_v1_u8.sqf:ro`, because those arrays are symlinks into the SquashFS. Plan
this as a task before the test job finishes, not after.

## 4. Exp2 - Staged training (pretrain, then fine-tune)

### 4.1 Pretrain stage
The Exp3 pretrain run, unchanged. Do not take checkpoints from the old timed-out run.

### 4.2 Fine-tune stage on completely

- **Config:** copy of the winner, named `biapy-exp2-finetune-completely.yaml`. Add:

  ```yaml
  MODEL:
    LOAD_CHECKPOINT: true
    LOAD_CHECKPOINT_ONLY_WEIGHTS: true   # weights only: fresh AdamW, epoch counter starts at 0
  PATHS:
    CHECKPOINT_FILE: metrics/biapy/biapy-exp3-pretrain-1k/checkpoints/biapy-exp3-pretrain-1k_0-checkpoint-best.pth
  ```

- **Code change required:** `apply_mode_overrides` in `biapy_work_folder/run_biapy-py.py` sets
  `MODEL.LOAD_CHECKPOINT = False` for every train launch, so the YAML above is silently ignored. Change the
  train branch to keep a YAML `true` **only when `PATHS.CHECKPOINT_FILE` is set**, e.g.

  ```python
  if mode == 'train':
      cfg['TRAIN']['ENABLE'] = True
      cfg['TEST']['ENABLE'] = False
      if not cfg.get('PATHS', {}).get('CHECKPOINT_FILE'):
          cfg['MODEL']['LOAD_CHECKPOINT'] = False
  ```

  This leaves every existing config's behavior unchanged. Verify in the train log: `Loading checkpoint from
  file` and `Model weights loaded!`, and that the "Mode overrides (train)" line no longer prints
  `MODEL.LOAD_CHECKPOINT=False`.
- **Schedule for fine-tuning:** do not reuse 240 epochs at 3.3e-4. The pretrain weights would be partly
  overwritten at that LR. Starting suggestion, to be treated as a hyperparameter:
  `LR: 1e-4` (about 3x lower), `EPOCHS: 60-80` (~21-28k iters, ~6-8 h), `WARMUP_COSINE_DECAY_EPOCHS: 5`,
  `MIN_LR` unchanged. Because the budget is small, you can afford a **2-point LR sweep** (3.3e-5 and 1e-4) as two
  parallel jobs; they use different job names so there is no checkpoint collision.
- **Disk/data:** `SRC_ROOT` must stay at the default completely tree (`fisbe/biapy-no-aug-zarr`). Do not export
  the 1k value from the previous step.
- **Checkpoint safety:** `PATHS.CHECKPOINT_FILE` is read-only. Do not hardlink the pretrain file onto the
  fine-tune job's own `…_0-checkpoint-best.pth`; they would share an inode and the pretrain checkpoint would be
  overwritten when validation improves.
- **Commands:**

  ```bash
  ./sbatch/biapy/biapy-py_sbatch_chain.sh biapy-exp2-finetune-completely train
  ```

### 4.3 Testing Exp2 on both test splits

Two separate test jobs, because a test run reads one `DATA.TEST` from the YAML:

1. **completely test:** a copy of the fine-tune YAML named `…-test-completely.yaml` with
   `PATHS.CHECKPOINT_FILE` **removed** (otherwise the test job loads the pretrain weights, since
   `CHECKPOINT_FILE` takes priority over the job directory). Submit with `-j biapy-exp2-finetune-completely -r 0`.
2. **1k test (forgetting check):** a copy with `DATA.TEST` on `fisbe/biapy-1k_v1_u8-zarr/test`
   (`IN_MEMORY: false`), same `-j … -r 0`, but with a distinct output name so the TIFFs don't overwrite
   the completely test. Use a different `-r` (e.g. `1k_test`) and hardlink the checkpoint to
   `{job}_1k_test-checkpoint-best.pth` after the fine-tune has exited.
   Score it with the same 1k-capable scorer as in 3.4.

## 5. Evaluation matrix

| Model | completely test (7) | 1k test (95) |
|---|---|---|
| Exp1 baseline (completely only) | main number | not run (optional) |
| Exp3 pretrain (1k only) | zero-shot transfer | in-domain number |
| Exp2 fine-tune (1k then completely) | **compare against Exp1** | forgetting check |

Interpretation: Exp2 vs Exp1 on completely test answers "does staging help". Exp3 on completely test
(zero-shot) shows how much transfers before fine-tuning. Exp3 on 1k test is the in-domain sanity check of the
synthetic pipeline.

Scoring uses `per_image_instances` (before the BiaPy 1000-voxel post-processing) and then drops components under
500 voxels, via `biapy_eval.py`. Keep that identical across arms. For the zero-shot Exp3-on-completely test,
use the hardlink approach from `biapy_1k_v1_u8.md` section 6 (new run id, same job name).

## 6. Confounds to control before trusting the comparison

1. **Intensity scale.** 1k raw is uint8; completely raw is uint16. The YAMLs do not set `DATA.NORMALIZATION`,
   so BiaPy's default `zero_mean_unit_variance` with `MEAN_VAL`/`STD_VAL = -1` is used, and no
   `normalization_*.npy` sits next to the checkpoint. It is unclear which statistics a test run on a different
   dataset would use. Fix this before Exp2 and the zero-shot test: either rescale both to the same range in prep
   (`CLIPMAX` for the 1k set already exists in the PPP prep) or set explicit
   `DATA.NORMALIZATION` values in all three YAMLs. Check by logging the mean/std actually applied in a test job.
2. **Equal compute is not equal.** Exp2 gets pretrain (~60k iters) + fine-tune (~25k) iters; Exp1 gets ~84k.
   That is roughly matched, which is convenient, but say it explicitly in the write-up.
3. **Channel order/semantics.** Check that the three channels of the synthetic data (HA, V5, FLAG) correspond to
   the same channel indices as completely; `CHANNEL_SHUFFLE: true` in the augmentation reduces sensitivity, but
   confirm.
4. **Watershed/post-processing were tuned on completely** (Dc/Dn seed thresholds, `remove_small_objects: 1000`).
   They are applied unchanged to the 1k test, so 1k numbers may be pessimistic.
5. **Overlapping neurons in synth data.** The unique-id merge keeps only one label per voxel (last channel wins),
   while the 1k eval GT keeps channels separate. 1k test scores are therefore capped by overlap.
6. **7 test volumes.** Report per-volume results with the baseline paired to each fine-tune volume.

## 7. Suggested order of work

1. Fix the `LOAD_CHECKPOINT` override in `run_biapy-py.py` (section 4.2) and add the 1k-capable scoring path
   (section 3.4), then do a 1-epoch smoke test of each.
2. Resolve normalization (section 6.1).
3. Submit Exp1 baseline (independent, no dependencies). Submit the Exp3 10-epoch probe at the same time on
   another node.
4. Submit the full Exp3 pretrain (40 epochs) once the probe looks sane and the speed is known.
5. After the pretrain job exits, submit the Exp2 fine-tune(s) and then the three test jobs in section 5.

## 8. Checklist

- [ ] Throughput probe on L40S vs H100/H200 (3 epochs each) recorded.
- [ ] Exp1 YAML: epochs set so the run fits 24 h; skip preprocessing; submitted.
- [ ] Exp3 YAML: `EPOCHS: 40`, `WARMUP_COSINE_DECAY_EPOCHS: 4`, `SRC_ROOT` = 1k tree; first log lines show 1k paths.
- [ ] Old timed-out 1k checkpoint is **not** used as a starting point or in any table.
- [ ] `run_biapy-py.py` keeps `LOAD_CHECKPOINT` only when `PATHS.CHECKPOINT_FILE` is set; train log shows
      `Model weights loaded!`.
- [ ] Exp2 YAML: lower LR/epochs, `CHECKPOINT_FILE` on the pretrain checkpoint, `SRC_ROOT` left at completely.
- [ ] Test YAMLs for Exp2 have no `CHECKPOINT_FILE`; one for completely test, one for 1k test.
- [ ] Normalization is explicit and identical in all YAMLs.
- [ ] 1k test scorer mounts the SquashFS and reads `fisbe/1k_v1_u8/test`.
- [ ] Per-volume results table for the three arms on completely test.
