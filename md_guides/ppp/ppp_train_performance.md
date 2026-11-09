# PPP training performance (L40S / H200, noenc / decoder)

Current as of 2026-11-06. This merges four write-ups that are now archived in [`historical/`](historical/):

| Date | Topic | Archived original |
|---|---|---|
| 2026-10-21 | H200 data handoff, fp16 raw transport (`our-noenc-cs`) | [`ppp_h200_gpu_util.md`](historical/ppp_h200_gpu_util.md) |
| 2026-10-26 | Decoder affinity vectorization (`our-decoder-c1k_v1`) | [`PPP_TRAIN_SPEEDUP_GUIDE.md`](historical/PPP_TRAIN_SPEEDUP_GUIDE.md) |
| 2026-10-29 | Decoder trilinear-upsample profile | (was only in session notes) |
| 2026-11-05 | noenc step: `fast_affinities`, `channels_last`, L40S → H200 | [`ppp_noenc_train_speedup.md`](historical/ppp_noenc_train_speedup.md) |
| 2026-11-06 | L40S speed-up investigation: fp16 × fast_aff grid, trace, compile, layer bench (§6) | (written here) |

The earlier batch-handoff / `ThreadedPrefetch` fix is in
[`historical/ppp_gpu_util_prefetch.md`](historical/ppp_gpu_util_prefetch.md).

---

## 1. Recommended settings

| | noenc, L40S | noenc, H200 | decoder (`train_code`), H200 |
|---|---|---|---|
| Partition / CPUs | `l40s_public`, 32 | `h200_public`, **16** (hard cap per GPU) | `h200_public`, 16 |
| `batch_size` | 18 | 18 | 30 |
| `num_workers` | 28 | **14** | 12 |
| `raw_transport_fp16` | optional, −3% (within noise; §6) | **true** | true |
| `fast_affinities` | true (preset default) | true (preset default) | n/a (decoder uses the vectorized code path) |
| `compile` (`compile.toml`) | **true**, −11% (§6) | not measured | n/a (noenc only) |
| `channels_last` | optional with compile, −16% total, but peaks at 45.1 of 46.1 GB | off (no gain) | off (slower) |
| Measured | **1.45 s/iter** with compile (1.63 without; 2.24 without fast_aff) | **0.63–0.66 s/iter**, util ~89% | 5.34 s/iter, util ~94% |
| 200k iterations | ~80 h | **~37 h** | ~12 days |

How to get those settings:

- **noenc on H200:** stack [`exp_v1_h200.toml`](../../PatchPerPix/experiments/flylight/setups/setup01/configs/exp/exp_v1_h200.toml)
  on the preset. It sets 14 workers and fp16 raw, plus experiment_v1's 50k iterations / checkpoint every 2k /
  snapshot every 25k. `raw_fp16.toml` alone is **not** enough on H200: it keeps the preset's 28 workers on 16 CPUs.

  ```bash
  ./sbatch/ppp/ppp_submit_train.sh our-noenc-c1k_v1 --base original_default.toml \
    --config exp_v1_h200.toml --id <exp_id> \
    --partition h200_public --cpus 16 --account torch_pr_61_tandon_advanced --time 48:00:00
  ```

- **noenc on L40S:** stack `compile.toml` (the presets don't set it yet). The first iteration compiles for
  ~2 min. Add `channels_last.toml` only for runs that never need more memory (it leaves ~1 GB free).

  ```bash
  ./sbatch/ppp/ppp_submit_train.sh our-noenc-c1k_v1 --base original_default.toml \
    --config compile.toml --id <exp_id> --account torch_pr_61_tandon_advanced --time 48:00:00
  ```

- **decoder on H200:** the `our-decoder-c1k_v1*` presets already default to `h200_public`, 16 CPUs, bs30,
  12 workers in `PRESET_TABLE`.
- `h200_public` rejects more than 16 CPUs per GPU. `--cpus 32` (the train default) or 24 fails with
  "partition 'h200_public' is not valid for this job". The noenc presets still default to L40S/32 CPUs, so
  H200 launches need `--partition h200_public --cpus 16`.
- The old `our-noenc-cs` H200 overlay (bs63, 12 workers, `cache_size = 128`, fp16) is archived as
  `configs/archive/h200-noenc.toml`. Bigger batches give no per-sample gain (§5), so bs18 is the default now.
- Experiments resumed with `-id` keep their saved `config.toml`, so they don't pick up new defaults such as
  `fast_affinities`.

---

## 2. How to diagnose a slow step

The training pipeline is:

```text
workers (gunpowder, PreCache, mp pipes) → main process: unpickle + Stack → ThreadedPrefetch → train thread: GPU step
```

Read the gunpowder profiling stats (every `profiling` iterations) in the train `.out`:

| Symptom | Meaning |
|---|---|
| `ThreadedPrefetch` wait ≈ 0, `TorchTrainExt` ≈ batch time | GPU-bound. More CPUs or workers won't help. Profile the step (§6). |
| `Stack` / `ThreadedPrefetch` large, batch time ≫ train time | Handoff-bound: bytes through the single main process (§3). |
| Same s/sample on a faster GPU | Launch/sync-bound: Python loops or host↔device syncs in the step (§4). |
| `memory.used` swings by 100 GB | Varying tensor shapes re-tune cuDNN workspaces (§4). |

Rules of thumb:

- Low CPU efficiency on one machine says nothing about whether the pipeline is CPU-bound on another. Look at
  **per-sample cost × required rate** and at **bytes/s through the single consumer**.
- Keep `num_workers` at roughly CPUs − 2 to CPUs − 4. Oversubscription hurts more than too few workers:
  16 workers on 16 CPUs gave 11.1 samples/s, 12 workers gave 29.2.
- Bigger batches don't raise supply. Supply is samples/s.
- `nvidia-smi` logs (`gpu_log/ppp/gpu_usage_log_<exp_id>_<jobid>.csv`, named by exp id, not preset) include
  data staging and startup. Average only over the training window: the H200 smoke run averaged 54% over the
  whole job and ~89% while training.
- Warm-up: the first 2–3 iterations are slow (cache fill, cuDNN autotune). Skip ~20 when timing.

---

## 3. Data handoff on H200 (`our-noenc-cs`, 2026-10-21)

**Problem.** H200 caps jobs at 16 CPUs. The first H200 setup (bs63, 16 workers) ran at ~33% GPU util, while
the L40S reference (32 CPUs, bs18) ran at 85%.

**Diagnosis.**

1. L40S was never CPU-bound: job 20330947 used 55 CPU-h over 24 h on 32 cores (~2.3 cores busy).
2. Workers are cheap: ZarrSource 0.05 s and IntensityAugment 0.02 s per sample. At the 17.5 samples/s the
   H200 can consume, that's ~3.5 cores of work.
3. The handoff is the limit. One sample pickles to ~34.7 MiB (`RAW` float32 `(3,140,140,140)` is 31.4 MiB of
   it), so a bs63 batch is ~2.2 GB per step through mp pipes into one process. `Stack` averaged 4.19 s
   (max 14 s) ≈ 63 × 0.07 s.
4. L40S hid the same cost: 18 × 0.07 = 1.3 s fits inside a 2.6 s step, overlapped by `ThreadedPrefetch`.
5. 16 workers + main process + prefetch thread + train thread on 16 CPUs caused the 6–19 s stalls.

**Fix.** `[training] raw_transport_fp16` (default false) in `train.py` casts `RAW` to float16 just before
`gp.PreCache`. `torch_model.forward` starts with `raw = raw.float()`. Plus 12 workers and `cache_size = 128`
(`cache_size = 400` held ~14 GB in flight for no benefit). `bench_ppp_dataloader.py` got a `--raw-fp16` flag.

**Results.**

| Workers | RAW | s/batch (max) | samples/s |
|---|---|---|---|
| 16 | fp32 | 9.58 (16.2) | 6.6 |
| 16 | fp16 | 5.67 (8.5) | 11.1 |
| 12 | fp16 | 2.16 (2.4) | 29.2 |

Bench: bs63, `synth_vh`, 16 CPUs, no GPU work (jobs 20488811–13). The 45-min smoke train 20535976 reached
**85.3%** util after startup. `Stack` dropped to 0.44 s (from 4.19), `ThreadedPrefetch` to 0.01 s (from 4.85).

**Caveats.**

- fp16 transport changes training inputs by ~1e-3 on values in [-1, 1] relative to fp32 L40S runs. Keep it in
  mind when comparing final metrics across the two.
- Bench pitfalls: `bench_ppp_dataloader_sbatch.sh` hardcodes partition `cs` and 32 CPUs (override on the
  `sbatch` line); `h200_public` needs `--gres=gpu:1`, so the CPU-only bench holds an idle GPU; bench step C
  re-runs with defaults (ignore it); the "handoff" line measures the uncast batch, so it never shows the fp16
  savings. Node noise is large (6.6 vs 15.4 samples/s for the same config); trust orderings, not absolutes.
- If util drops later: try 10 workers, `prefetch_batches = 3`, or shared-memory transport (not implemented).

---

## 4. Decoder: affinity vectorization (`our-decoder-c1k_v1`, 2026-10-26)

**Problem.** On H200 (bs30, AMP, `add_affinities = "loss"`, `train_code = true`) training ran at ~10.0 s/iter
(p10 8.2, p90 11.6), mean util ~60%, `memory.used` swinging 30–140 GB. Per-sample cost was the same as on L40S
(0.34 s/sample for L40S bs22 job 21007371 and H200 bs30 job 20945953), so the faster GPU bought nothing.

**Root cause.** `seg_to_affgraph_3d_multi_torch_code` (`PatchPerPix/PatchPerPix/util/train_util.py`) built
ground-truth affinities for 1024 sampled foreground locations per batch element with two Python loops: one per
location (30 × 1024 slices with GPU-tensor bounds, a host sync each) and one per edge (343 edges × ~10 tiny
kernels). That is ~10⁵ syncs/launches per step. The memory swings were probably cuDNN re-tuning for varying
decoder input shapes, since the sample count was `min(sample_cnt, n_fg)`.

**Fix.**

| File | Change |
|---|---|
| `PatchPerPix/PatchPerPix/util/train_util.py` | New `_seg_to_aff_patches_vectorized`: gathers only the centre and `nhood` voxels per patch in one advanced-indexing op, then `aff = any_C(center == off & center != 0) & fg[centre] & fg[offset]`. The 3D and 2D `*_torch_code` functions delegate to it. Output identical. |
| `setup01/torch_model.py` | Sample exactly `sample_cnt` locations per element (with replacement), so decoder shapes are constant. Elements with zero foreground are still skipped. |
| `setup01/train.py` | TF32 on for fp32 matmul/conv outside autocast. |
| `configs/train/smoke_noval.toml` | `val_log_step = 2000`; stack after `smoke.toml` so validation doesn't distort timing. |
| `sbatch/ppp/tools/test_affgraph_vectorized.py` | Randomized equivalence test vs the loop version (3D and 2D). Result: `3D ok (200, 343)`, `2D ok (200, 49)`. |

The only semantic change is the fixed sample count: elements with fewer than 1024 foreground voxels now get
1024 with-replacement samples. If the loss regresses, revert that first.

**Results** (smoke job 21045488, 150 iterations, first 20 skipped):

| | before (20945953) | after (21045488) |
|---|---|---|
| median s/iter | 10.07 | **5.34** (1.9×) |
| p10 / p90 | 8.2 / 11.6 | 4.8 / 5.9 |
| mean GPU util | ~60% | **~94%** |
| `memory.used` | 30–140 GB, swinging | ~46–49 GB, flat |
| loss at it 100 | 0.642 | 0.596 |

### Decoder: what the remaining 5–6 s is (2026-10-29)

Profiled with `profile_ppp_step.py` (job 21338403, synthetic inputs, H200, bs30, 140³, AMP): step 6.0 s =
backward 3.7 + forward 2.3. The UNet forward is only 0.25 s; the **decoder forward is 2.08 s**, and
`aten::upsample_trilinear3d` (+ backward) is **~85% of CUDA time**.

Cause: funlib's `Upsample(mode="nearest")` actually builds `nn.Upsample(mode="trilinear")` + a 1×1 conv, and the
decoder runs it on 30 × 1024 = 30,720 tiny patches (channels 128→64 and 64→1).

Dead ends: `cudnn.benchmark` off (6.30 s), `channels_last` (6.52 s, slower), no-AMP at bs30 (CUDA illegal
memory access), input 112 (fails valid-padding shape). bs16 = 3.23 s, so the step is linear in batch.

**Fix, implemented but not landed:** `model.autoencoder.upsample_conv_first` in `torch_model.py`
(`_ConvFirstSequential`) runs the 1×1 conv **before** the trilinear upsample. It is the same math with the same
weights (interpolation weights sum to 1), so checkpoints stay compatible, and the second stage upsamples
1 channel instead of 64.

- Equivalence (`sbatch/ppp/tools/test_decoder_upsample_equiv.py`, job 21768764): fp32 max output diff 5.7e-6,
  dW 1.2e-6; AMP output 1.2e-4, dW 4.5e-3 (output absmax 0.173).
- Not enabled in any TOML yet, and the speed-up is **not measured**.
- Still an uncommitted edit in `torch_model.py`, along with the two test scripts.
- Alternative: switch autoencoder upsampling to `transposed_conv` (new weights, needs retraining).

---

## 5. noenc: the step is conv-bound (`our-noenc-c1k_v1`, 2026-11-05)

Merged into `main` as `01c946a`.

### Where the L40S step goes

The real run 21773195 (L40S, bs18, 28 workers) ran at **2.18 s/iter**, util 85.5%. `ThreadedPrefetch` wait was
0.00 s; `Stack` (0.60 s) overlapped the GPU step; validation/checkpoint/snapshot stalls were ~0.8%. Data loading
was not the bottleneck on L40S. Don't re-chase it there.

Synthetic step (job 21787759, L40S, bs18):

| Section | s/step |
|---|---|
| backward | 0.984 |
| UNet forward | 0.410 |
| `compute_affinities` (GT targets) | 0.187 |
| loss | 0.077 |
| **total** | **1.689** |

Kernel view: convolution backward ~51% of CUDA time; cuDNN NCHW↔NHWC layout conversion **~17%**;
`binary_cross_entropy_with_logits` ~8%.

### `training.fast_affinities` (exact, on in the noenc presets)

`torch_model.seg_to_affgraph_3d_multi_torch_cropped` computes the noenc GT affinities directly on the cropped
output region, slicing with Python ints instead of indexing a CUDA neighborhood tensor (removes ~12 syncs per
edge) and skipping the zero-init and crop of the padded volume. It activates only for noenc + overlapping
instances + 3D + `add_affinities = "loss"`.

- Bit-identical to the old path: `sbatch/ppp/tools/test_fast_affinities.py` → "ALL EQUAL".
- Affinity time 0.187 → 0.022 s (L40S), 0.234 → 0.030 s (H200).
- `fast_affinities = true` in `our-noenc-c1k_v1.toml`, `our-noenc-c1k_v1_u8.toml`, `base1-noenc-c.toml`.

### `training.channels_last` (optional, off by default)

`UnetModelWrapper.use_channels_last()` stores 5D weights in `channels_last_3d`; `forward()` converts `raw` to
match; `train.py` calls it after `.to(device)` and before building the optimizer. Only strides change, so
checkpoints stay compatible. Overlay: `channels_last.toml`.

| GPU | fp32 loss rel | fp32 grad rel | AMP fp16 grad rel |
|---|---|---|---|
| L40S | 0 | 2.1e-8 | 1.9e-4 |
| H200 | 0 | 2.3e-7 | 3.4e-5 |

(`sbatch/ppp/tools/test_channels_last.py`: same weights, fwd + loss + backward.)

Effect on L40S: removes the layout-conversion kernels (~265 ms/step) but adds strided elementwise kernels (loss
0.076 → 0.092 s, +4.4 GB peak memory). Net ~−5%. No gain on H200.

### Synthetic step results (bs18 unless noted)

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

H200 peak memory: 39 GB at bs18, 65 GB at bs36, 111 GB at bs63.

Dead ends: bf16 autocast 1.724 s vs 1.689 (fp16 tensor cores already saturated); fp32 (no AMP) and bs24 OOM on
L40S (bs18 peaks at 35.8 GB); bigger batch on H200 (flat ms/sample; bs63 × 200k iterations is 3.5× the samples).

### Real data

| | L40S (21773195) | H200 (21902285) |
|---|---|---|
| s/iter, steady state | 2.18 | **0.658** |
| `TorchTrainExt` | 2.18 | 0.645 |
| `ThreadedPrefetch` wait | 0.00 | 0.00 |
| `Stack` | 0.60 s (fp32 raw) | 0.13 s (fp16 raw) |
| Workers | 28 | 14 (~74% load) |
| GPU util during training | 85.5% | ~89% |
| Peak GPU memory | 35.8 GB | 41.7 GB |

experiment_v1 EXP 2 (job 21975068, H200, `exp_v1_h200.toml`) trained at 0.63 s/iter, confirming the smoke run.

### Why H200 is faster with fewer CPUs

The step is GPU-bound, and on both GPUs the batch was ready before the GPU needed it. So CPUs only have to keep
up. Required supply is ~8 samples/s on L40S and ~27 samples/s on H200; workers spend ~0.07 s per sample (about
2 cores of work), which is why 14 workers on 16 CPUs ran at ~74% load. The real supply limit is the
single-process unpickle + stack, which is bound by bytes, so fp16 raw is what lets 16 CPUs keep up.

| Spec (approximate) | L40S | H200 |
|---|---|---|
| Dense fp16 tensor throughput | ~362 TFLOPS | ~989 TFLOPS (~2.7×) |
| Memory bandwidth | ~0.86 TB/s GDDR6 | ~4.8 TB/s HBM3e (~5.5×) |
| SMs | 142 | 132 |

The measured 84.8 vs 31.4 ms/sample (2.7×, fast_aff on both) matches the compute ratio. The rest fits too:
bf16 doesn't help (compute-bound); fast_aff saves −27% on H200 vs −10% on L40S (a sync costs the same fixed time
on either GPU, so it's a bigger share of a faster step); channels_last helps only L40S (it removes
bandwidth-bound work that HBM already makes cheap).

The headline 3.3× (2.18 vs 0.658) is **not** all hardware. Like-for-like synthetic is 2.2× (base) or 2.7×
(fast_aff). The H200 run also had fast_aff, fp16 raw and `1k_v1`-only data, and the L40S real step sits
~0.5 s above its synthetic step (1.69 → 2.18) while the H200 gap is 0.09 s.

**Hypothesis:** the extra 0.5 s on L40S is main-process overhead from fp32 transport. Stacking
18 × 31 MiB runs in the prefetch thread, which can hold the GIL and compete for memory bandwidth with the thread
launching kernels. Predicted L40S with fast_aff + fp16: 1.46–1.54 s/iter if true, ~1.95–2.0 if not.
Tested on 2026-11-06 (§6, step 1): partly right, and `fast_affinities` (not fp16) is what removes the gap.


## 6. L40S speed-up investigation (2026-11-06)

### Problem

noenc training on L40S (`our-noenc-c1k_v1`, bs18, 28 workers) ran at **2.18–2.34 s/iter**, about 124 h per
200k iterations, versus 0.66 s/iter on H200. Synthetic profiling (§5) put the GPU-only step at ~1.69 s, so
~0.5 s/iter was unexplained. The open question from §5 was whether `raw_transport_fp16` and `fast_affinities`
close that gap, and what else limits the L40S.

### Diagnosis, step by step

All real-data runs: `our-noenc-c1k_v1` on `original_default.toml` + `profile_noenc_l40s.toml` (300 it on
`1k_v1`, no val/ckpt/snapshot), `l40s_public`, 32 CPUs, `torch_pr_61_tandon_advanced`. Steady-state s/iter =
median of iterations 21–300 (`summarize_train_profile.py`). One job per arm, so different nodes: the same config
has measured 2.18–2.34 s/iter across nodes, so treat differences under ~3% as noise.

**Step 1: fast_aff × fp16 raw grid (real data).** Is the 0.5 s gap fp32 transport, the affinity syncs, or both?

| Arm | Job | s/iter | vs base | `Stack` | prefetch wait | peak mem |
|---|---|---|---|---|---|---|
| base (fp32, old affinities; `no_fast_aff.toml`) | 22104455 | 2.240 | — | 0.40 s | 0.00 | 37.4 GB |
| fp16 raw (`raw_fp16.toml`) | 22104466 | 2.053 | −8.4% | 0.23 s | 0.00 | 35.3 GB |
| fast_aff | 22104461 | 1.673 | −25.3% | 0.24 s | 0.00 | 42.4 GB |
| fast_aff + fp16 | 22104468 | 1.624 | −27.5% | 0.21 s | 0.00 | 42.2 GB |

What it told us: `fast_affinities` is the big win on real data (−0.57 s/step, 3.3× its synthetic −0.17 s). The
old path's ~12 host syncs per edge stall the kernel-launch thread, which on real data competes with the prefetch
thread for the GIL. fp16 raw helps only while those syncs exist: −0.19 s alone, −0.05 s (noise) on top of
fast_aff. The batch was always ready (prefetch wait 0.00). With fast_aff the real step is ~0.10 s above
synthetic, the same gap as on H200.

**Step 2: real-data trace, sync audit, clocks** (22110017, `trace_train.toml`: torch.profiler over iterations
201–210, sync audit over 231–235; `analyze_torch_trace.py`). Where does the remaining 1.65 s go?

- **GPU busy 98.8%** of the step; idle ~20 ms/iter. Feeding the GPU is no longer a problem.
- **6 host syncs/iter**, all in gunpowder `TorchTrainExt`: 3 blocking H2D input copies (`torch_train.py:247`),
  `if loss != 0.0`, the GradScaler inf check, `loss.cpu()`. The CPU waits there while the GPU is still busy,
  so they cost nothing.
- **H2D copy**: 335 MiB/iter at 8.9 GB/s (pageable, not pinned) = 39 ms/iter (2.4%) with no overlap.
- **Power cap**: `clocks_event_reasons` = SW Power Cap in 198/202 training samples; mean SM clock 2136 of
  2520 MHz (85%), 317 of 350 W. Per-card power behaviour is the likely source of the node-to-node spread.
- **Kernel time**: conv 878 ms (56%), elementwise 368 ms (23.5%), cuDNN NCHW↔NHWC layout conversion 267 ms
  (17%), reduce 34 ms. About 40% of the step is memory-bound non-conv work, the L40S's weak side.

**Step 3: synthetic variants** (22110019, 22114572; `profile_ppp_step.py`, fast_aff + fp16 raw, bs18).
Can compilation or layout remove the memory-bound 40%?

| Variant | s/step | vs eager | peak mem | note |
|---|---|---|---|---|
| eager | 1.533 | — | 33.4 GiB | |
| `+fused_adam` | 1.537 | 0% | | optimizer step is already 4 ms |
| compile UNet + heads | 1.444 | −6% | | backward 0.98 → 0.95 s |
| compile UNet + heads + loss | 1.333 | −13% | 32.8 GiB | loss 76 → 9 ms; elementwise 368 → ~166 ms (fused Triton) |
| compile UNet + heads + channels_last | 1.380 | −10% | | uncompiled loss pays the cl penalty |
| compile all + channels_last | 1.252 | −18% | 35.5 GiB | also removes the 267 ms of layout kernels |

`test_compile_equiv.py` (22114572): fp32 loss/grad rel 7e-8, outputs identical; AMP fp16 grad rel 5.7e-3
(reduction-order noise in the fused loss backward; loss rel 3.5e-7); no recompiles. **COMPILE OK.**

**Step 4: layer bench** (22115735; each block alone, fwd + bwd, eager). Is the conv work itself efficient?
Model conv FLOPs / step time = **85 TFLOPS = 23.5% of the 362 TFLOPS fp16 peak**.

| Block | Input | ms | Share | TFLOPS | % peak |
|---|---|---|---|---|---|
| `r_conv.0.0` (top decoder level) | 332 ch @ 56³ | 615 | **46%** | 142 | 39% |
| `l_conv.0` (first layer, 3 input ch) | 3 ch @ 140³ | 223 | **17%** | 10 | **2.8%** |
| `r_conv.0.1` | 435 ch @ 32³ | 137 | 10% | 160 | 44% |
| `r_up.0.0` (upsample + crop + concat) | 375 ch @ 28³ | 98 | 7% | 23 | 6% |
| `l_conv.1` | 20 ch @ 68³ | 78 | 6% | 48 | 13% |
| `r_conv.0.2` | 630 ch @ 20³ | 45 | 3% | 160 | 44% |
| `out_code_affs` (343-ch head) | 312 ch @ 52³ | 39 | 3% | 41 | 11% |
| everything else | | 100 | 7% | | |

What it told us: `fmap_dec_factors = 1.2` (vs `fmap_inc_factors = 3`) keeps the decoder wide. The top decoder
level convolves 332 → 312 channels at 56³ and holds **66% of the model's FLOPs** at ~40% of peak already, so
the only big lever left there is a narrower decoder (a model change). The 3-channel first layer is 17% of the
step at 2.8% efficiency: tensor cores pad 3 channels to 8 and its activations are bandwidth-bound.

**Step 5: `training.compile` on real data** (new opt-in flag; 22118725/26/28, same batch of jobs):

| Run | Job | s/iter | vs eager | peak GPU mem (nvidia-smi) | first iteration |
|---|---|---|---|---|---|
| eager (fast_aff + fp16) | 22118725 | 1.626 | — | 34.8 of 46.1 GB | |
| `compile.toml` | 22118726 | **1.449** | **−10.9%** | 35.2 GB | 128 s (compile) |
| `compile.toml` + `channels_last.toml` | 22118728 | **1.359** | **−16.4%** | **45.1 GB** | 117 s |

All three runs were power-capped at a similar clock (2134–2173 MHz), so the gain is not a faster card. Logs show
no recompiles or fallbacks. `train_net_checkpoint_0` from the compiled runs has the same 41 keys and shapes as
the eager run (no `_orig_mod.` prefix), so compiled checkpoints resume and predict without the flag. The H200
compile run (22118729) was cancelled before it started.

### Fixes applied and result

| Change | Where | Status | Real L40S effect |
|---|---|---|---|
| `fast_affinities` (exact GT affinities, no per-edge syncs) | `torch_model.py`; on in the noenc presets | default (since 2026-11-05) | 2.240 → 1.673 s/iter (−25%) |
| `raw_transport_fp16` | `raw_fp16.toml` / `exp_v1_h200.toml` | optional on L40S | −3% on top of fast_aff (noise) |
| **`training.compile`** (UNet + heads + loss, in place via `nn.Module.compile()`) | `train.py` `_maybe_compile_model`; overlay `configs/train/compile.toml` | **new, opt-in** (noenc only) | 1.626 → **1.449** s/iter (−10.9%) |
| compile + `channels_last` | `compile.toml` + `channels_last.toml` | opt-in, **memory-tight** | 1.626 → 1.359 s/iter (−16.4%), 45.1 of 46.1 GB |

Net: **2.24 → 1.45 s/iter (−35%, 1.55× throughput)** with fast_aff + fp16 + compile. 200k iterations go from
~124 h to ~80 h (~76 h with channels_last as well).

Ruled out (measured, no useful gain): fused Adam (0%), removing the remaining syncs (GPU already busy), more
data workers or other data-path work (prefetch wait 0.00 in every arm).

Left on the table: pinned + overlapped H2D (≤2.4%); a narrower decoder (largest lever, needs a quality
comparison); compile on H200 (not measured).

---

## 7. Tooling

| Tool | Use |
|---|---|
| `sbatch/ppp/tools/profile_ppp_step.py` | Synthetic-input step profiler. Variants compose with `+` (`fast_aff`, `cl`, `bf16`, `no_amp`, `bsN`, `inN`), e.g. `fast_aff+cl+bs63`. Variant flags override the TOML, so `base` stays the reference. `--profile-variants` picks which get a torch.profiler table and chrome trace. Summary has an `ms/smp` column. |
| `sbatch/ppp/tools/profile_ppp_step_noenc_sbatch.sh` | Runs `test_fast_affinities.py` (`RUN_AFF_TEST`) and `test_channels_last.py` (`RUN_CL_TEST`) first, then one process per variant (`VARIANTS`, logged to `<OUT_DIR>/variant_<v>.log`), then a combined table. Outputs: `metrics/ppp/profile_noenc/<job>/`. |
| `sbatch/ppp/tools/profile_ppp_step_sbatch.sh` | Same for the decoder setup. |
| `sbatch/ppp/tools/bench_ppp_dataloader.py` (+ `_sbatch.sh`) | Dataloader throughput without GPU work; `--raw-fp16`. See §3 caveats. |
| `test_fast_affinities.py`, `test_channels_last.py`, `test_affgraph_vectorized.py`, `test_decoder_upsample_equiv.py` | Equivalence tests for each change above. |
| `sbatch/ppp/tools/summarize_train_profile.py JOBID...` | Steady-state s/iter (median/mean/p90, `--skip` warm-up), train-step time, GPU util and peak memory from a training job's log and `gpu_log/ppp/` CSV. |
| `no_fast_aff.toml` | A/B overlay that turns `fast_affinities` off. |
| `trace_train.toml` (+ `train.py` `_TrainDiagnostics`) | Real-data diagnostics: `training.torch_trace_iters = [A, B]` writes a torch.profiler chrome trace + op table, `training.sync_debug_iters = [C, D]` attributes every host sync to its Python stack. Output: `<exp>/train/torch_trace/`. Off unless set. |
| `sbatch/ppp/tools/analyze_torch_trace.py TRACE [--whole]` | Per-iteration GPU busy/idle, largest idle gaps with the main-thread op running, H2D/DtoH time and bandwidth, sync calls, kernel time by category (conv / layout / elementwise / …), top kernels. `--whole` for `profile_ppp_step.py` traces. |
| `profile_ppp_step.py` variants `compile`, `compile_all`, `fused_adam`; `--layer-bench VARIANT` | `torch.compile` of UNet + heads (`compile_all`: + loss); per-block fwd/bwd ms, conv GFLOP, TFLOPS and % of peak. |
| `test_compile_equiv.py` (`RUN_COMPILE_TEST=1` in the noenc profile sbatch) | Eager vs compiled fwd + loss + backward from identical weights; fp32 hard check, AMP reported, recompile check. |
| GPU logger in `ppp_job.sh` | Now also logs `clocks.sm`, `clocks.max.sm`, `power.draw`, `power.limit` and `clocks_event_reasons.active` (`0x4` = SW power cap). |
| Overlays `profile_noenc_l40s.toml`, `profile_noenc_completely_l40s.toml`, `profile_noenc_h200.toml` | 300 real iterations with no val/ckpt/snapshot noise (`_completely` trains on `completely`; `_h200` adds 14 workers + fp16). |
| `smoke.toml` + `smoke_noval.toml` | Short smoke train without validation distorting timing. |
| `ppp_job.sh` | When submitted from a worktree or copy (`SLURM_SUBMIT_DIR` contains `sbatch/ppp/ppp_job.sh`), runs that tree's code instead of the main checkout's. |

H200 synthetic profile, from the repo root:

```bash
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

---

## 8. Open follow-ups

1. **L40S speed (§6), remaining:**
   - Decide whether `compile = true` becomes the default in the noenc presets. It is opt-in for now; existing
     experiments resumed with `-id` keep their saved config either way.
   - Measure `compile.toml` on H200 (22118729 was cancelled) and in a long run that reaches validation
     (156³ input, a second compiled graph) and checkpoint writes.
   - Pin + overlap the H2D copy (pinned buffers, `non_blocking`, side stream in `ThreadedPrefetch`): ≤2.4%.
   - Narrower decoder (`fmap_dec_factors`): top decoder level = 46% of the step, 66% of FLOPs. Quality experiment.
   - Optionally add `raw_transport_fp16` to the L40S noenc presets so their inputs match H200 runs (speed-neutral).
2. **Decoder `upsample_conv_first`:** commit it with its tests, measure the step with `profile_ppp_step.py`,
   then enable it in the decoder presets.
3. **Decoder leftovers:** the per-element `torch.nonzero` loop in `torch_model.forward` (~30 syncs/step);
   re-choose `batch_size` once the step is GPU-bound.
4. **channels_last on L40S:** convert head outputs back to contiguous before the loss to recover the extra
   elementwise/loss cost (a few % at most).
5. **Worktree cleanup:** `.claude/worktrees/ppp-noenc-speed` still holds the 7 GB smoke-run experiment dir
   `metrics/ppp/prof-noenc-c1k_v1-h200`. Remove the worktree when it's no longer needed.
6. **Long-run check on H200:** watch util around validation and checkpointing; the fp16 smoke runs never reached
   a snapshot write.

---

## Job index

| Job | GPU | What | Result |
|---|---|---|---|
| 20330947 | L40S | `our-noenc-cs` reference, 32 CPUs, bs18 | 85.2% util, 2.59 s/iter |
| 20070698 | H200 | `our-noenc-cs` old H200 setup, bs63, 16 workers | 33.5% util; batch 6–19 s |
| 20488811–13 | H200 | Dataloader bench fp32/w16, fp16/w16, fp16/w12 | 6.6 / 11.1 / 29.2 samples/s |
| 20535976 | H200 | `our-noenc-cs` 45-min smoke with fp16 + 12 workers | 85.3% util |
| 20945953 | H200 | `our-decoder-c1k_v1` before vectorization, bs30 | 10.07 s/iter, ~60% util |
| 21007371 | L40S | `our-decoder-c1k_v1_u8`, bs22 | 7.5 s/iter |
| 21045488 | H200 | Decoder smoke after vectorization | 5.34 s/iter, ~94% util |
| 21338403 | H200 | Decoder synthetic profile | trilinear upsample ~85% of CUDA time |
| 21768764 | H200 | Decoder `upsample_conv_first` equivalence | equal within fp tolerance |
| 21773195 | L40S | `our-noenc-c1k_v1` real training, bs18, 28 workers | 2.18 s/iter, 85.5% util |
| 21758017 | L40S | noenc real-data 300-it profile | ~2.3 s/iter |
| 21787759 | L40S | noenc synthetic: base / fast_aff / bf16 / no_amp / bs24 | 1.689 → 1.510 s (fast_aff) |
| 21867382 | L40S | noenc synthetic + channels_last test | fast_aff+cl 1.449 s |
| 21867383 | H200 | noenc synthetic bs18/36/63 + channels_last test | 0.565 s at bs18; ~31 ms/sample at every bs |
| 21902285 | H200 | noenc real-data 300-it smoke (`profile_noenc_h200.toml`) | **0.658 s/iter**, ~89% util |
| 22104455 | L40S | real 300-it grid: base (no fast_aff, fp32) | 2.240 s/iter |
| 22104466 | L40S | real 300-it grid: fp16 raw | 2.053 s/iter |
| 22104461 | L40S | real 300-it grid: fast_aff | 1.673 s/iter |
| 22104468 | L40S | real 300-it grid: fast_aff + fp16 | **1.624 s/iter**, ~82% util |
| 22110017 | L40S | real trace + sync audit + clocks (`trace_train.toml`) | GPU busy 98.8%; 6 syncs/iter; power-capped at 85% clock |
| 22110019 | L40S | synthetic: fused_adam, compile, compile_all, compile+cl | compile_all −13% |
| 22114572 | L40S | `test_compile_equiv` + compile_all ± cl | COMPILE OK; compile_all+cl **1.252 s** (−18%) |
| 22115735 | L40S | layer bench (fast_aff) | 23.5% of fp16 peak; top decoder conv 46% of step |
| 22118725 | L40S | real 300 it, eager (fast_aff + fp16) | 1.626 s/iter |
| 22118726 | L40S | real 300 it, `compile.toml` | **1.449 s/iter** (−10.9%) |
| 22118728 | L40S | real 300 it, `compile.toml` + `channels_last.toml` | 1.359 s/iter (−16.4%), 45.1 GB peak |
| 22118729 | H200 | real 300 it, `compile.toml` | cancelled before start |
| 21975068 | H200 | experiment_v1 EXP 2 v1 n500 train | 0.63 s/iter |

Logs: `sbatch/ppp/logs/<job_name>-<jobid>.out`. GPU logs: `gpu_log/ppp/`.
