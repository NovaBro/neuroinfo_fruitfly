# PPP noenc training speed-up (`our-noenc-c1k_v1`, L40S → H200)

Written 2026-11-05. Work lives in the worktree `.claude/worktrees/ppp-noenc-speed`
(branch `worktree-ppp-noenc-speed`, uncommitted at time of writing).
Related: [`PPP_TRAIN_SPEEDUP_GUIDE.md`](PPP_TRAIN_SPEEDUP_GUIDE.md) (decoder affinity vectorization),
[`ppp_h200_gpu_util.md`](ppp_h200_gpu_util.md) (H200 16-CPU cap, fp16 raw transport),
[`historical/ppp_gpu_util_prefetch.md`](historical/ppp_gpu_util_prefetch.md) (ThreadedPrefetch).

## TL;DR

- On L40S, noenc training ran at **2.18 s/iter** (bs18). The step is bound by GPU 3D convolutions,
  not by data loading. 200k iterations ≈ **121 h per stage**.
- Fixes: `training.fast_affinities` (exact, on by default in the noenc configs) and an optional
  `training.channels_last`. On L40S they give −15% on the synthetic step. bf16 and bigger batches don't help.
- The big lever is hardware. On **H200, bs18, real data: 0.658 s/iter, GPU util ~89%**, with the data
  pipeline keeping up. 200k iterations ≈ **37 h per stage** (3.3× faster than L40S).

Recommended training setup for the two-stage plan:

| Setting | Value |
|---|---|
| Partition | `h200_public` (≤16 CPUs per GPU) |
| Account | `torch_pr_61_tandon_advanced` |
| `batch_size` | 18 (keeps "200k iterations" equal to the L40S runs; per-sample cost is flat from bs18 to bs63) |
| `num_workers` | 14 |
| `raw_transport_fp16` | true |
| `fast_affinities` | true |
| `channels_last` | false (no gain on H200) |

## Goal

1. Train PPP noenc for 200k iterations on `completely` + `1k_v1`, and pick the best checkpoint by
   `completely` validation.
2. Fine-tune that checkpoint on `completely` only for another 200k iterations, and pick the best by
   `completely` validation again.

At 2.18 s/iter on L40S this is about 10 days of training, so the training step had to get faster first.

## Jobs referenced

| Job | GPU | What | Result |
|---|---|---|---|
| 21773195 | L40S | Real training, `our-noenc-c1k_v1` (bs18, 28 workers) | 2.18 s/iter, GPU util 85.5% |
| 21758017 | L40S | Real-data 300-iteration profile run (`profile_noenc_l40s.toml`) | ~2.3 s/iter |
| 21787759 | L40S | Synthetic step profile: base / fast_aff / bf16 / no_amp / bs24 | base 1.689 s; fast_aff 1.510; bf16 no gain; no_amp & bs24 OOM |
| 21867382 | L40S | Synthetic profile + channels_last test | fast_aff+cl 1.449 s; test OK |
| 21867383 | H200 | Synthetic profile at bs18/36/63 + channels_last test | fast_aff 0.565 s at bs18; ~31 ms/sample at every batch size |
| 21902285 | H200 | Real-data 300-iteration smoke run (`profile_noenc_h200.toml`) | **0.658 s/iter**, util ~89% |

Logs: `sbatch/ppp/logs/ppp_profile_noenc-<job>.out` (profiles), `sbatch/ppp/logs/ppp_our-noenc-c1k_v1_train-<job>.out`
(training). Profile outputs and traces: `metrics/ppp/profile_noenc/<job>/` (in the worktree for 21787759 onward).

## Analysis

### 1. The L40S step is conv-bound; the data pipeline is hidden

From the gunpowder profiling stats of the real run 21773195:

- `TorchTrainExt` process: **2.18 s** mean per batch.
- `ThreadedPrefetch` wait: **0.00 s**. Stack (unpickle + stack 18 samples) costs 0.60 s, but it overlaps
  the GPU step.
- Validation, checkpoint and snapshot stalls: ~0.8% of total time.

So data loading and augmentation were not the bottleneck. Don't re-chase them for L40S.

### 2. Where the GPU step goes (synthetic, L40S, bs18, job 21787759)

| Section | s/step |
|---|---|
| backward | 0.984 |
| UNet forward | 0.410 |
| compute_affinities (gt targets) | 0.187 |
| loss | 0.077 |
| total step | 1.689 |

Kernel view: convolution backward ~51% of CUDA time. Weight-gradient, input-gradient and forward
implicit-GEMM kernels are 12–14% each. cuDNN NCHW↔NHWC layout-conversion kernels are **~17%** in total.
`binary_cross_entropy_with_logits` is ~8%.

The real step (2.18 s) is ~0.5 s above the synthetic one (1.69 s). The L40S logs don't attribute that
gap. On H200, with fast_aff and fp16 raw transport, the gap shrank to 0.09 s.

### 3. Dead ends

- **bf16 autocast:** 1.724 s vs 1.689 (no gain; fp16 tensor cores are already saturated).
- **fp32 (no AMP), bs24:** OOM on the 44 GB L40S (bs18 peaks at 35.8 GB).
- **Bigger batch on H200:** 31.4 / 31.7 / 32.1 ms per sample at bs18 / 36 / 63, so no per-sample gain.
  At bs63, 200k iterations would be 3.5× the samples (~112 h).
- **channels_last on H200:** 0.563 vs 0.565 s at bs18; ≤3% at bs63.

## Fixes implemented

### `training.fast_affinities` (exact)

`torch_model.seg_to_affgraph_3d_multi_torch_cropped` computes the noenc ground-truth affinities directly
on the cropped output region. It slices with Python ints instead of indexing a CUDA neighborhood tensor,
which removes ~12 device→host syncs per edge, and it skips the zero-init and crop of the padded volume.
It only activates for noenc + overlapping instances + 3D + `add_affinities = "loss"`.

- Verified bit-identical to the old path by `sbatch/ppp/tools/test_fast_affinities.py` ("ALL EQUAL").
- Affinity time: 0.187 → 0.022 s (L40S); 0.234 → 0.030 s (H200).
- Turned on (`fast_affinities = true` under `[training]`) in `our-noenc-c1k_v1.toml`,
  `our-noenc-c1k_v1_u8.toml` and `base1-noenc-c.toml`. Experiments resumed with `-id` keep their saved
  config, so they don't pick it up.

### `training.channels_last` (optional, off by default)

- `UnetModelWrapper.use_channels_last()` stores 5D weights in `channels_last_3d`. `forward()` converts
  `raw` to match. `train.py` calls it after `.to(device)` and before building the optimizer.
- Only strides change: `state_dict` values are identical, so checkpoints stay compatible.
- Overlay: `setup01/channels_last.toml`.
- Verified by `sbatch/ppp/tools/test_channels_last.py`. It builds the model twice from identical weights
  and compares fwd + loss + backward:

  | GPU | fp32 loss rel | fp32 grad rel (all params) | AMP fp16 grad rel |
  |---|---|---|---|
  | L40S | 0 | 2.1e-8 | 1.9e-4 |
  | H200 | 0 | 2.3e-7 | 3.4e-5 |

- **Effect:** on L40S it removes the layout-conversion kernels (~265 ms/step). It then adds strided
  elementwise kernels: `add_` rises, a generic `elementwise_kernel` appears, and loss goes 0.076 → 0.092 s.
  Peak memory rises 4.4 GB. The net gain is only ~0.08 s/step (−5%), and none on H200. Use it only on L40S.

### Profiling tooling

- `sbatch/ppp/tools/profile_ppp_step.py`:
  - Variants compose with `+` (`fast_aff`, `cl`, `bf16`, `no_amp`, `bsN`, `inN`), e.g. `fast_aff+cl+bs63`.
  - Variant flags override the TOML (so `base` stays the reference even though the TOML now sets
    `fast_affinities = true`).
  - `--profile-variants` chooses which variants get a torch.profiler table and chrome trace.
  - The summary has an `ms/smp` column.
- `sbatch/ppp/tools/profile_ppp_step_noenc_sbatch.sh`:
  - Runs `test_fast_affinities.py` (`RUN_AFF_TEST`) and `test_channels_last.py` (`RUN_CL_TEST`) first.
  - Runs one process per variant (`VARIANTS`), logged to `<OUT_DIR>/variant_<v>.log`.
  - Prints a combined summary table at the end.
- `sbatch/ppp/ppp_job.sh`: when submitted from a worktree or copy (`SLURM_SUBMIT_DIR` contains
  `sbatch/ppp/ppp_job.sh`), it runs that tree's code instead of the main checkout's.
- Overlays: `profile_noenc_l40s.toml` (300 iterations, no val/ckpt/snapshot noise),
  `profile_noenc_completely_l40s.toml` (same, trains on `completely`), and
  `profile_noenc_h200.toml` (same, plus 14 workers and `raw_transport_fp16`).

## Results

### Synthetic GPU step (bs18 unless noted)

| Variant | L40S s/step | L40S ms/sample | H200 s/step | H200 ms/sample |
|---|---|---|---|---|
| base | 1.696 | 94.2 | 0.771 | 42.8 |
| cl | 1.624 | 90.2 | — | — |
| fast_aff | 1.527 | 84.8 | 0.565 | 31.4 |
| fast_aff+cl | 1.449 | 80.5 | 0.563 | 31.3 |
| fast_aff, bs36 | — | — | 1.143 | 31.7 |
| fast_aff+cl, bs36 | — | — | 1.145 | 31.8 |
| fast_aff, bs63 | — | — | 2.024 | 32.1 |
| fast_aff+cl, bs63 | — | — | 1.969 | 31.3 |

Peak memory on H200: 39 GB at bs18, 65 GB at bs36, 111 GB at bs63.

### Real data

| | L40S (21773195) | H200 (21902285) |
|---|---|---|
| s/iter, steady state | 2.18 | **0.658** |
| TorchTrainExt | 2.18 | 0.645 |
| Wait for next batch (ThreadedPrefetch) | 0.00 | 0.00 |
| Stack | 0.60 s (fp32 raw) | 0.13 s (fp16 raw) |
| Workers | 28 | 14 (~74% load) |
| GPU util during training | 85.5% | ~89% |
| Peak GPU memory | 35.8 GB | 41.7 GB |
| 200k iterations | ~121 h | **~37 h** |

The H200 run warms up for two iterations (55 s, 16 s for cache fill and cuDNN autotune), then stays at
0.65–0.67 s/iter.

Note on reading `nvidia-smi` logs: the mean over the whole job (54%) includes data staging and startup.
Average only over the training window.

## How to run

H200 synthetic profile:

```bash
cd /scratch/wmz2007/neuroinfo_fruitfly/.claude/worktrees/ppp-noenc-speed
RUN_AFF_TEST=0 VARIANTS="base fast_aff fast_aff+cl fast_aff+bs36" \
EXTRA="--profile-variants base fast_aff+cl" \
sbatch --account=torch_pr_61_tandon_advanced --partition=h200_public --time=01:30:00 \
  sbatch/ppp/tools/profile_ppp_step_noenc_sbatch.sh
```

H200 real-data smoke run (300 iterations):

```bash
./sbatch/ppp/ppp_submit_train.sh our-noenc-c1k_v1 --base original_default.toml \
  --id prof-noenc-c1k_v1-h200 --config profile_noenc_h200.toml \
  --partition h200_public --cpus 16 --account torch_pr_61_tandon_advanced --time 01:00:00
```

`h200_public` rejects more than 16 CPUs per GPU. The submit fails with
"partition 'h200_public' is not valid for this job" for `--cpus 32` (the train default) or 24.

## Caveats

- The smoke run trained on `1k_v1` only. Stage 1 adds the `completely` volumes. With ~25% worker headroom
  it should still keep up, but this is not measured.
- `raw_transport_fp16` feeds the model raw at fp16 precision, which differs from the existing L40S runs.
  The earlier H200 setup in `ppp_h200_gpu_util.md` already used it.
- The main checkout has an uncommitted decoder change in `torch_model.py` (`upsample_conv_first`,
  `_ConvFirstSequential`) that this worktree lacks. Merge both when bringing this branch back. Noenc
  doesn't use that code.

## Next to-dos

1. **Commit and merge** the worktree changes into `main`, together with the main checkout's decoder
   `upsample_conv_first` change.
2. **Stage 1 config:** noenc on `completely` + `1k_v1`, 200k iterations, val on `completely`, with the H200
   settings above. Check how two `train_datasets` are mixed (`oversample_complete`, `max_train_samples`).
3. **Stage 2 config:** fine-tune from the stage-1 best checkpoint on `completely` only, another 200k
   iterations. Check how `run_ppp.py` / `TorchTrainExt` start from an existing checkpoint (and whether the
   optimizer state and iteration counter carry over).
4. **Measure checkpoint selection cost.** Picking the best checkpoint needs predict + decode + label +
   evaluate on `completely/val` per checkpoint. At `checkpoints = 2000` that is 100 checkpoints per stage,
   which may cost as much as training. Options: checkpoint every 10k (20 per stage), or validate a coarse
   subset first and then refine around the best.
5. **Optional:** a real-data H200 smoke run on `completely` (`profile_noenc_completely_l40s.toml` with H200
   settings) to confirm worker headroom with those volumes.
6. **Optional (L40S only):** convert head outputs back to contiguous before the loss under `channels_last`,
   to recover the extra elementwise and loss cost (a few % at most).
