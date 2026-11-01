# PPP training step speed-up (affinity vectorization)

Guide for the `our-decoder-c1k_v1` slowdown on H200 (observed 2026-10-26, job 20945953).
Related: `PatchPerPix/CODEBASE.md` (architecture), `ThreadedPrefetch` fix for the batch-handoff idle time
(a separate, earlier problem).

## Problem

Training `our-decoder-c1k_v1` (batch_size=30, AMP on, `add_affinities = "loss"`, `train_code = true`)
on an H200 ran at **~10.0 s/iter** (median; p10 8.2, p90 11.6), with:

- mean GPU utilization ~60% (not low enough to be cancelled, but far from saturated);
- `memory.used` swinging between ~30 GB and ~140 GB of 143 GB across training.

Measurements from the logs:

| run | GPU | batch | median s/iter | s/sample |
|---|---|---|---|---|
| `ppp_our-decoder-c1k_v1_u8_train-21007371` | L40S | 22 | 7.5 | 0.34 |
| `ppp_our-decoder-c1k_v1_train-20945953` | H200 | 30 | 10.07 | 0.34 |

- The "time per iteration went up" on H200 is only the larger batch. **Per-sample cost is identical**, so a much
  faster GPU bought nothing. The step is bound by CPU/kernel-launch/host-sync overhead, not GPU compute.
- It is not data loading: `Train process time` ≈ `Batch time`, and `ThreadedPrefetch` already hides the
  stack/unpickle handoff.

### Root cause

With `add_affinities = "loss"`, `UnetModelWrapper.forward` (`torch_model.py`) builds the ground-truth affinities on the GPU for
`num_code_samples` (1024) sampled foreground locations per batch element, via
`seg_to_affgraph_3d_multi_torch_code` (`PatchPerPix/PatchPerPix/util/train_util.py`). That function had two Python loops:

1. one iteration per sampled location (batch 30 × 1024 ≈ 30k) doing
   `patches[idx] = seg[sample_loc[0], :, sample_loc[1]:sample_loc[1]+ps, ...]`. Slicing with GPU-tensor bounds forces a
   host sync per iteration plus a small copy kernel;
2. one iteration per edge (343 for a 7³ patch) with ~10 tiny kernels each.

That is on the order of 10⁵ syncs/launches per step, which do not speed up on a faster GPU.

Memory swings (probable cause, not separately measured): the number of sampled locations per batch element was
`min(sample_cnt, n_fg)`, so the decoder input shape varied from step to step. With `cudnn.benchmark = True` and the caching
allocator, varying shapes cause re-tuning workspace spikes and fragmentation.

## Fix

1. **Vectorize the affinity computation.** Only the centre voxel and the `nhood` voxels of each patch are needed, so gather
   those directly in one advanced-indexing op (`(N, 1+E, C)`) instead of copying full patches. Then
   `same = ((center == off) & (center != 0)).any(C)` and `aff = same & fg[centre] & fg[offset]` → float `(N, E)`.
   Output is identical to the old function.
2. **Fixed sample count.** Sample exactly `sample_cnt` locations per batch element (the sampling was already with
   replacement via `torch.randint`), so decoder input shapes are constant.
3. **TF32** for any fp32 matmul/conv outside autocast.

`expandable_segments:True` was already set in `sbatch/ppp/ppp_job.sh` (~L646), so nothing changed there.

## Code changes

| file | change |
|---|---|
| `PatchPerPix/PatchPerPix/util/train_util.py` | New `_seg_to_aff_patches_vectorized(seg, nhood, psH, samples_loc)`; `seg_to_affgraph_3d_multi_torch_code` and `seg_to_affgraph_2d_multi_torch_code` now delegate to it (signatures unchanged). Other affinity variants (`seg_to_affgraph_*_torch`, `*_batch`) are untouched. |
| `PatchPerPix/experiments/flylight/setups/setup01/torch_model.py` | Both `rand_loc_ids = torch.randint(...)` calls use `size=(sample_cnt,)` instead of `min(sample_cnt, n_fg)`. Elements with zero foreground are still skipped. |
| `PatchPerPix/experiments/flylight/setups/setup01/train.py` | `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True` after `cudnn.benchmark = True`. |
| `PatchPerPix/experiments/flylight/setups/setup01/smoke_noval.toml` (new) | `[training] val_log_step = 2000`; stack after `smoke.toml` so validation does not distort s/iter timing. |
| `sbatch/ppp/tools/test_affgraph_vectorized.py` (new) | Randomized equivalence test, vectorized vs the original loop implementation (3D and 2D, includes zero background, multi-channel, corner locations). |

Behavioral note: the only semantic change is (2). For batch elements with fewer than 1024 foreground voxels the
decoder now sees 1024 (with-replacement) samples rather than `n_fg`. If the loss regresses, revert that first.

## Verification

Equivalence test (CPU is enough), from the repo root:

```bash
singularity exec --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate ppp; python sbatch/ppp/tools/test_affgraph_vectorized.py'
```

Result (2026-10-26): `3D ok (200, 343)`, `2D ok (200, 49)`.

Smoke run (separate exp id; does not touch running jobs):

```bash
./sbatch/ppp/ppp_submit_train.sh our-decoder-c1k_v1 --base original_default_train_code.toml \
  --id our-decoder-c1k_v1-smoke --config smoke.toml --config smoke_noval.toml \
  --time 00:30:00 --account torch_pr_61_tandon_advanced
```

Submitted as job **21045488** (h200_public). Log: `sbatch/ppp/logs/ppp_our-decoder-c1k_v1_train-21045488.out`;
GPU log: `gpu_log/ppp/`.

Compare against the baseline (ignore iterations 1-20):

- s/iter: baseline median 10.07 s, p90 11.6 s (`grep "Batch: iteration"`).
- `memory.used` range: baseline 30-140 GB.
- mean GPU utilization: baseline ~60%.
- loss at iterations 1, 50, 100, 150: baseline ~2.12 at it 1, ~0.64 at it 100.

### Results

Smoke job 21045488 (h200_public, COMPLETED, 150 iterations, 19 min; iterations 21-150 used, first 20 skipped as warm-up):

| | baseline (20945953) | smoke (21045488) |
|---|---|---|
| median s/iter | 10.07 | **5.34** (~1.9x faster) |
| p10 / p90 s/iter | 8.2 / 11.6 | 4.8 / 5.9 |
| mean GPU util (steady state) | ~60% | **~94%** |
| `memory.used` | 30-140 GB, swinging | **~46-49 GB, flat** (one ~134 GB spike during warm-up) |
| loss at it 100 | 0.642 | 0.596 |

- Warm-up: the first iterations took 70 s, 30 s and 8 s (startup + cudnn benchmark), then settled near 5-6 s.
- Loss at iterations 50/100/150 was 0.615/0.596/0.583; no NaNs or errors in the log. Iteration 1 differs (3.72 vs 2.12), which is
  random initial state. This is only a 150-iteration sanity check, not proof that final quality is unchanged.
- The gain is smaller than removing the loops alone might suggest. The gunpowder profile shows `Stack` averaging 4.2 s per call
  (n=10, max 15.9 s) and `ZarrSource` ~0.9 s, so data loading may be close to the limit again at the new step time. A
  `torch.profiler` pass would show how the remaining ~5.3 s splits between data wait and GPU work.

## Follow-ups (not done)

- Remove the per-batch-element `torch.nonzero` loop in `torch_model.py` forward (~30 host syncs/step) if profiling still shows it.
- `channels_last_3d` and `torch.compile` on the decoder.
- Re-choose `batch_size` once the step is GPU-bound.
