# PPP labgraph (GPU patch-graph phase): fixing low GPU utilization

Written 2026-10-22. Companion to [`ppp_overview.md`](ppp_overview.md) and
[`ppp_h200_gpu_util.md`](ppp_h200_gpu_util.md) (same symptom, training side).

## TL;DR

The GPU half of the split label step (`ppp_<preset>_labgraph_<split>`, config
`vi_split_gpu.toml`) left the GPU idle 82–98% of the time, and Greene cancelled one task for low
utilization. Each 92³ block took about **2 minutes regardless of how much work it had**. Almost
all of that was host-side work: two ~10 GB *unified-memory* consensus arrays were zero-filled on
the CPU and migrated to the GPU page by page, and an unguarded `np.min`/`np.max` then pulled one
of them back to the host on every block.

The fixes:
- keep the consensus arrays in device memory;
- guard the min/max behind DEBUG;
- cache compiled kernels;
- vectorize the Python pair loops;
- stop building a networkx graph that the GPU phase throws away;
- restore logging in the spawned child process.

With these, blocks take **~3 s (empty) to ~5–7 s (with foreground)**.

## Jobs referenced

| Job | What | Result |
|-----|------|--------|
| 20622991_[0-6] | `our-noenc-cs` labgraph, test split, ckpt 34000, old code | 47–133 min per sample; mean GPU util 0.2–9.7%; task 1 (R14A02) **cancelled** |
| 20666571_[0-6] | Same, with the fix (only R14A02 and VT011145 recomputed) | R14A02 **39.5 min** (256 blocks + 299 inter-block); VT011145 **27.8 min** (old: 47 min); others skipped in ~1 min |
| 20666572_[0-6] | CPU label array on the new caches | R14A02 8 min, VT011145 6 min |
| 20666580 | Evaluate, test split | running at time of writing |

Utilization with the fix is still low (R14A02: mean 7.6%, 87% of samples at 0%; peak 21.5 GB GPU
memory), but each sample now finishes in well under an hour, so the job no longer sits idle for
hours.

Utilization summary for 20622991, from `gpu_log/ppp/gpu_usage_log_our-noenc-cs_20622991_*.csv`
(3 s samples):

| Task | Duration | Mean util | Samples at 0% |
|------|----------|-----------|---------------|
| 0 | 92 min | 9.4% | 83% |
| 1 (R14A02, cancelled) | 133 min | 0.2% | 98% |
| 2 | 105 min | 9.7% | 82% |
| 3 | 47 min | 8.6% | 84% |
| 4 | 95 min | 8.9% | 83% |
| 5 | 112 min | 5.0% | 90% |
| 6 | 56 min | 7.4% | 85% |

## Problem

### Symptom

The label step is already split into a GPU job (per-block and inter-block patch graph, cached in
`<instanced>/<sample>.zarr` under `volumes/blocks`) and a CPU job (`vi_split_cpu.toml`,
`affGraphToInstances`). The split was meant to keep the CPU-only labelling off the GPU, but the
"GPU" job itself barely used the GPU.

### Diagnosis

1. **No per-block logs.** `run_ppp.py` runs `vote_instances_sample_seq` in a `spawn` child
   process. A spawned child starts with an unconfigured root logger, so every INFO line from
   `vote_instances` and `stitch_patch_graph` was dropped. The `.out` file stopped at
   `forking __main__.vote_instances_sample_seq`.
2. **Block cache timestamps showed a fixed cost per block.** The `aff_graph_mat` mtimes in R14A02's
   partial cache were about 2 minutes apart whether a block had 1 patch pair or 2,631. The cost was
   per block, not per unit of work.
3. **Where the fixed cost came from**, per block (block 92³ plus a 3-voxel margin = 98³,
   `patchshape` 7³, `neighshape` 14³ = 2,744):
   - **Consensus arrays.** `outCons` and `outConsCnt` have shape `(14,14,14,98,98,98)` float32,
     **10.3 GB each**. They were allocated with `pycuda.driver.managed_zeros`, which allocates
     unified memory and zero-fills it **on the host** (first touch of ~20 GB), and then the
     kernels page-fault it over to the GPU.
   - **An unguarded full-array min/max.** In `create_consensus_array_cuda`, the log line after
     normalization (its message even says "too slow") always ran `np.min(outCons)` and
     `np.max(outCons)`. That migrated 10 GB back to the host for a single-threaded scan, and the
     next kernels (`rank_patches_cuda`, `computePatchGraph_cuda`) migrated it back to the GPU again.
   - **Affinity copies.** The 343-channel fp16 block (~650 MB) went through `astype(float32)` in
     `blockwise_vote_instances`, then a zero-filled managed buffer plus another copy in
     `to_instance_seg`.
   - **Kernel builds.** `DynamicSourceModule` was called three or four times per block. The
     kernel source has the block shape baked in, so this went through pycuda's compile path on
     every block.
4. **Python hotspots on dense blocks and stitching.** `computeAndStorePatchPairs` looped over
   every k-d tree pair in Python and removed rejects with `set.remove`. In stitching,
   `remove_pairs` did the same, and `remove_intra_block_pairs` ran `np.all(block == point)`
   against every candidate for every pair (roughly O(pairs × candidates)). Stitching also filled
   a networkx graph edge by edge with `update_graph`, even when `termAfterStitchGraph` discards it.
5. **Whole-volume CPU work before the first block** (`clean_mask`, `skeletonize_3d` on the
   bounding box). It turned out to be small: 2.4 s and 1.5 s for R14A02.

## Fix

```mermaid
flowchart LR
  load["load_input fp16"] --> copy["single cast into managed buffer"]
  copy --> cons["consensus fill+norm (device memory)"]
  cons --> rank["rank_patches_cuda"]
  rank --> cover["foreground cover / thin cover"]
  cover --> pairs["computeAndStorePatchPairs (vectorized)"]
  pairs --> graph["computePatchGraph_cuda"]
  graph --> write["zarr write block cache"]
```

Measured on R14A02 with the fix (from the restored logs):

| Stage | Time |
|-------|------|
| `load_affs` (zarr read + blosc decompress, 343 ch) | 1.3–1.9 s |
| `copy_affs_to_managed` (fp16 → fp32) | 0.8–1.0 s |
| `consensus` | 0.1–0.2 s (3.2 s on the first block: kernel compile) |
| `ranking`, `foreground_cover`, `thin_cover`, `patch_pairs` | ≤ 0.13 s each |
| `patch_graph` | ~0.3 s |
| whole block (`blockwise_vote_instances`) | ~2.5–3 s empty, ~4–7 s with foreground |

The remaining per-block time is mostly CPU I/O (the affinity read). The GPU still idles between
blocks, but each sample now takes minutes instead of hours.

## Code changes

### Diagnostics

| File | Change |
|------|--------|
| `PatchPerPix/experiments/run_ppp.py` | New `_parent_log_spec()` / `_configure_child_logging()`. `fork` and `fork_return` pass the parent's root log level and `FileHandler` paths to `_fork_worker` / `_fork_return_worker`, which call `logging.basicConfig(..., force=True)`. Child INFO logs now reach `.out` and `run.log` (as `INFO:__mp_main__` / `INFO:PatchPerPix...`). |
| `PatchPerPix/PatchPerPix/vote_instances/utilVoteInstances.py` | New `timed(name)` context manager that logs `time <name>: <s>s`. |
| `.../vote_instances.py` (`to_instance_seg`) | Timers around `copy_affs_to_managed`, `filter_patches`, `consensus`, `ranking`, `foreground_cover`, `thin_cover`, `patch_pairs`, `patch_graph`. Logs the selected patch count and the pair count. |
| `.../stitch_patch_graph.py` | Timers around `load_affs`, `do_block`, `clean_mask`, `skeletonize_bb`, `stitch_vote_instances`, `inter_block_pairs`, `inter_block_load_affs`, `inter_block_do_block`, and `all-block-stitching`. Logs per-block shape, foreground voxels, and patch pairs, and per-inter-block candidates and pairs. |

### Device-memory consensus arrays and kernel cache

| File | Change |
|------|--------|
| `.../vote_instances/cuda_code.py` | `make_kernel` caches `DynamicSourceModule` objects in a per-process dict keyed by `(code, options)`; `delete_cuda` clears it. New helpers: `alloc_empty_array` (managed, no host zero-fill), `alloc_zero_device_array` (`pycuda.gpuarray.zeros`, falls back to managed memory on `MemoryError`), `to_device_array`, `cuda_arg` (passes `.gpudata` for a GPUArray), `to_host`, and `free_array`. The module logger is private (`_logger`) so `from .cuda_code import *` doesn't override importers' `logger`. |
| `.../vote_instances/consensus_array.py` | `create_consensus_array_cuda` allocates `outCons` / `outConsCnt` with `alloc_zero_device_array` and the overlap mask with `to_device_array`, and passes them via `cuda_arg`. The post-normalization min/max is behind `logger.isEnabledFor(logging.DEBUG)`. Debug min/max and `save_consensus` use `to_host`. Frees go through `free_array`. |
| `.../vote_instances/ranked_patches.py` | `rank_patches_cuda` passes `cuda_arg(consensus_vote_array)`. |
| `.../vote_instances/aff_patch_graph.py` | `computePatchGraph_cuda` passes `cuda_arg(consensus_vote_array)` in both launch paths. |

Memory: about 2 × 10.3 GB device memory per 98³ block plus ~1.3 GB of managed affinities, which
fits on an L40S (48 GB). With `num_parallel_blocks > 1`, several workers would compete for
device memory. Allocation then falls back to managed memory (logged as a warning) instead of
failing.

### Single fp16 → fp32 cast

| File | Change |
|------|--------|
| `.../vote_instances/vote_instances.py` | In the CUDA path, `pred_affs` is assigned straight into an uninitialized managed buffer (`alloc_empty_array`); the assignment does the fp16 → fp32 cast. The non-CUDA path still casts to float32 explicitly. |
| `.../vote_instances/stitch_patch_graph.py` | Removed `block.astype(np.float32)` before `do_block`, for both per-block and inter-block calls. |

### Vectorized pair code, no throwaway graph

| File | Change |
|------|--------|
| `.../vote_instances/utilVoteInstances.py` | New `query_pairs_array(tree, r, p)`: `cKDTree.query_pairs(..., output_type='ndarray')`, with a fallback for older scipy. |
| `.../vote_instances/aff_patch_graph.py` | `computeAndStorePatchPairs` builds `pts` with `np.array`, filters pairs with a single distance mask, and fills `arr` by slicing (pairs, then single-patch rows). Row order can differ from the old set-iteration order; pair content is the same. |
| `.../vote_instances/stitch_patch_graph.py` | `remove_pairs` / `remove_intra_block_pairs` take and return `(N, 2)` index arrays. Block membership is checked with `np.isin` on int64 coordinate keys (`coord_keys`, 21 bits per axis). `candidate_idx` and `overlapping_pairs` are built with array indexing. `update_graph` (networkx) is skipped when `termAfterStitchGraph` is set (`build_graph`). A `seen_nonempty_block` flag replaces `affgraph is None` as the "first non-empty block, no neighbours yet" check, so the stitching order is unchanged. |

### Verification tooling

| File | Purpose |
|------|---------|
| `sbatch/ppp/tools/compare_vi_block_cache.py` | Compares two `<sample>.zarr` block caches: block and inter-block groups, pairs as undirected row sets, and affinities aligned by pair (`--atol`). With `--hdf OLD NEW` it also checks that two instance segmentations are the same partition up to relabeling. Exits non-zero on differences. |

## Verification

- **Pair logic.** Random-data equivalence test of the old loop versions against the new
  vectorized `remove_pairs` + `remove_intra_block_pairs` (5 trials, 352–864 pairs) and
  `computeAndStorePatchPairs` (3 trials, ~10k rows): identical pair sets.
- **Comparison script.** Running `compare_vi_block_cache.py` of the old VT011145 cache against
  itself reports 292 / 292 groups matching.
- **End to end.** The old outputs for VT011145 (`.hdf`, `.png`, `.zarr`) and the partial R14A02
  cache were moved to `metrics/ppp/our-noenc-cs/test/_old_labgraph_20622991/`, and the label array
  was resubmitted:

  ```bash
  ./sbatch/ppp/ppp_submit_test.sh array our-noenc-cs --stage label --split test --ckpt 34000 \
    --datasets completely --agg --keep-pred-affs
  ```

  Samples whose `.hdf` already exists are skipped in about 20 s (`label_resume.toml`). Once
  VT011145 finishes:

  ```bash
  OLD=metrics/ppp/our-noenc-cs/test/_old_labgraph_20622991
  NEW=metrics/ppp/our-noenc-cs/test/instanced/34000/patch_threshold_0_3/fc_threshold_0_3/mws_True/skeletonize_foreground_True/numinst_threshs__0_9_0_1_/completely
  python3 sbatch/ppp/tools/compare_vi_block_cache.py \
    $OLD/VT011145-20171222_63_I1.zarr $NEW/VT011145-20171222_63_I1.zarr \
    --hdf $OLD/VT011145-20171222_63_I1.hdf $NEW/VT011145-20171222_63_I1.hdf
  ```

  Result: **`OK`**. All 292 block and inter-block groups have identical pair sets, the max
  `aff_graph_mat` difference is 2.4e-7 (float summation order), and the final segmentation is
  the **same partition** (2,153 labels, bijective mapping over 207,703 foreground voxels).

- **Where the time goes now** (R14A02, sum over all blocks, from the `time …` lines):

  | Stage | Total |
  |-------|-------|
  | `load_affs` + `inter_block_load_affs` | 571 s |
  | `copy_affs_to_managed` | 494 s |
  | `consensus` | 748 s |
  | `patch_graph` | 185 s |
  | `foreground_cover` + `ranking` + `thin_cover` | 62 s |
  | `patch_pairs`, `inter_block_pairs` | ~0 s |

  Reading the affinities and copying them to managed memory (~18 min together) is now the largest cost.

## Remaining / follow-ups

- **Per-block time is now dominated by `load_affs` (~1.8 s) and `copy_affs_to_managed`
  (~1 s).** The load reads 343 channels × 98³ from `pred_affs` chunks (343×32³), which aren't
  aligned to the bounding-box offset. A prefetch thread that loads block *i+1* while block *i*
  runs on the GPU would hide most of the load. Uploading fp16 and casting on the device would
  shrink the copy.
- `computeForegroundCover` / `thinOutForegroundCover` still loop in Python. They're ≤ 0.1 s on
  sparse blocks; check the logs on dense samples before optimizing them.
- After the comparison passes, delete `_old_labgraph_20622991/`.
