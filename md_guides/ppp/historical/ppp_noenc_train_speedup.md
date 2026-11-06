> **Archived 2026-11-06.** Superseded by [`../ppp_train_performance.md`](../ppp_train_performance.md). Kept verbatim; relative links below point to the file's old location in `md_guides/ppp/`.

# PPP noenc training speed-up (`our-noenc-c1k_v1`, L40S → H200)

Written 2026-11-05. Developed in the worktree `.claude/worktrees/ppp-noenc-speed` (branch
`worktree-ppp-noenc-speed`) and merged into `main` as **`01c946a`** (fast-forward, 2026-11-05).
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
(training). Profile outputs and chrome traces: `metrics/ppp/profile_noenc/<job>/`. H200 smoke-run GPU log:
`gpu_log/ppp/gpu_usage_log_prof-noenc-c1k_v1-h200_21902285.csv`. These jobs ran from the worktree; their logs
and outputs were copied into the main checkout (gitignored). The smoke run's experiment dir
(`metrics/ppp/prof-noenc-c1k_v1-h200`, 7 GB of checkpoints) is still only in the worktree.

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
- Overlay: `setup01/configs/train/channels_last.toml`.
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

## Why H200 is faster than L40S (same batch size, fewer CPUs)

Short answer: the step is GPU-bound. The H200 runs the fp16 3D convolutions ~2.7× faster and has ~5.5×
the memory bandwidth. CPUs only feed data, and on both GPUs the data was ready before the GPU needed it
(ThreadedPrefetch wait 0.00 s). So fewer CPUs cost nothing as long as the pipeline keeps up, and fp16 raw
transport is what makes it keep up on 16 CPUs.

### CPU count doesn't set the speed

- Wait for the next batch was 0.00 s on both GPUs, so more CPUs can't speed up the step.
- Required supply: L40S ~8 samples/s (bs18 / 2.18 s); H200 ~27 samples/s (bs18 / 0.658 s). Workers spend
  ~0.07 s per sample, about 2 cores of real work. That is why 14 workers on 16 CPUs ran at ~74% load.
- L40S was over-provisioned anyway: job 20330947 used 55 CPU-h over 24 h on 32 cores (~2.3 cores busy;
  see [`ppp_h200_gpu_util.md`](ppp_h200_gpu_util.md)).
- The real supply limit is the single main process that unpickles and stacks each batch. It is limited
  by bytes, not cores. fp16 raw cut Stack from 0.60 s to 0.13 s. Not oversubscribing CPUs matters too:
  16 workers on 16 CPUs gave 11.1 samples/s, 12 workers gave 29.2.

### Where the GPU time goes, and why H200 handles it better

| Part of the step (L40S synthetic, bs18) | Share | Limited by |
|---|---|---|
| Convolution forward + backward (implicit GEMM) | ~51% of CUDA time in backward alone | tensor-core FLOPs |
| cuDNN NCHW↔NHWC layout conversion | ~17% | memory bandwidth |
| BCE loss + elementwise ops | ~8%+ | memory bandwidth |
| `compute_affinities` (before fast_aff) | 0.187 s | host↔device syncs (latency) |

| Spec (approximate) | L40S | H200 |
|---|---|---|
| Dense fp16 tensor throughput | ~362 TFLOPS | ~989 TFLOPS (~2.7×) |
| Memory bandwidth | ~0.86 TB/s GDDR6 | ~4.8 TB/s HBM3e (~5.5×) |
| SMs | 142 | 132 |

The L40S has more SMs. The H200's edge is ~4× tensor-core work per SM per clock, plus HBM. The measured
~2.7× per sample with fast_aff (84.8 vs 31.4 ms/sample) matches the compute ratio. Convolutions
dominate; the bandwidth-bound parts shrink too.

The other results fit this picture:
- **bf16: no gain on L40S.** The fp16 tensor cores are already saturated, so L40S really is compute-bound.
- **fast_aff: −27% on H200 vs −10% on L40S.** A host↔device sync costs the same fixed time on either GPU.
  Once everything else is fast, the syncs are a bigger share of the step.
- **channels_last: −5% on L40S, ~0 on H200.** It removes bandwidth-bound layout conversions, and those
  are already cheap with HBM.
- **Bigger batch on H200: no per-sample gain** (31.4 / 31.7 / 32.1 ms at bs18 / 36 / 63). The GPU is
  already full at bs18. The extra memory (141 GB) is not a source of speed.

### Part of the 3.3× real-data gap isn't hardware

The headline 2.18 vs 0.658 s/iter compares runs that weren't set up the same way:

| | L40S 21773195 | H200 21902285 |
|---|---|---|
| fast_aff | off (predates it) | on |
| Raw transport | fp32 | fp16 |
| Training data | `completely` + `1k_v1` | `1k_v1` only |
| Synthetic → real gap | 1.69 → 2.18 s (+0.5 s, not explained) | 0.565 → 0.658 s (+0.09 s) |

Like-for-like hardware ratio: synthetic base 1.696 / 0.771 = **2.2×**; with fast_aff on both,
1.527 / 0.565 = **2.7×**. The rest of the 3.3× comes from fast_aff and fp16 transport, which only the H200
run had, plus the unexplained L40S gap.

**Hypothesis (not measured):** the +0.5 s on L40S is main-process overhead from fp32 transport. Stacking
18 × 31 MiB (0.60 s of mostly Python work) runs in the prefetch thread. It can hold the GIL and compete
for memory bandwidth with the training thread that launches GPU kernels, even though it "overlaps" the
step. fp16 halves those bytes, and the gap shrank to 0.09 s on H200. To test: run a real-data L40S job
with `fast_affinities` + `raw_transport_fp16` and compare its synthetic → real gap.

## How to run

H200 synthetic profile:

```bash
cd /scratch/wmz2007/neuroinfo_fruitfly   # submit from the repo root
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
- The decoder conv-first change in `torch_model.py` (`upsample_conv_first`, `_ConvFirstSequential`) is
  still an **uncommitted** edit in the main checkout, re-applied on top of `01c946a`. It is not part of
  this commit. Noenc doesn't use that code.

## Next to-dos

1. ~~Commit and merge the worktree changes into `main`~~ — done, `01c946a`. Still open: commit the
   decoder `upsample_conv_first` change (with `sbatch/ppp/tools/test_decoder_upsample_equiv*`), and
   remove the worktree once its 7 GB smoke-run experiment dir is no longer needed.
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
