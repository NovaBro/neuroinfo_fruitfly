> **Historical / archived.** Current PatchPerPix ops and canonical presets: [`ppp_overview.md`](../ppp_overview.md). Keep this file for background only; do not treat recipes here as the live path.
>
> ---

# PPP training GPU utilization — batch-handoff prefetch

Why PPP train jobs got low-util warnings / cancellations on public GPU
partitions, what was measured, and the `ThreadedPrefetch` fix.

Related: [`ppp_performance_plan.md`](ppp_performance_plan.md) §6 (GPU util),
[`PatchPerPix/CODEBASE.md`](../../../PatchPerPix/CODEBASE.md) (training stack).

---

## 1. Symptom (Oct 2026)

| Run | Mean GPU util | Samples <10% | Steady s/iter |
|---|---|---|---|
| `basic-ppp-noenc` | 71% | 24% | 3.9 |
| `basic_fmaps24` (cancelled) | 62% | 32% | 3.7 |
| `basic_fmaps24_400k` (cancelled) | 46% | 49% | 3.9 |
| `basic-ppp-noenc-cp` (cancelled) | 19% | 79% | 4.3 median, 37% of steps >10 s |

Even "good" runs dipped below 10% for a quarter of the samples: a per-step
sawtooth, not long stalls.

## 2. Root cause: serial batch handoff in the main process

Pipeline (`setup01/train.py`): sources → augmentations → `gp.PreCache`
(32 worker processes) → `gp.Stack(batch_size)` → `TorchTrainExt`.

`PreCache` sits **before** `Stack`, so each iteration the main process alone:

1. unpickles 18 samples (~35 MiB each, ~650 MB) from the workers'
   `multiprocessing.Queue`;
2. `np.stack`s them into a ~600 MB batch;
3. copies it to the GPU, runs forward/backward;
4. reads `loss.cpu()`, which waits for the GPU to finish.

Nothing overlaps, so the GPU idles during steps 1–2:

```
main thread: [unpickle+stack ~1.2s][GPU step ~2.7s][unpickle+stack][GPU step]...
GPU:          idle                  busy             idle            busy
```

On a shared node (e.g. gl068, 4×L40S, 128 cores shared with other users) the
gap grows, and the partition's utilization check cancels the job.

## 3. What was ruled out (measured)

Benchmark: `sbatch/ppp/tools/bench_ppp_dataloader.py` + `sbatch/ppp/tools/bench_ppp_dataloader_sbatch.sh`
(CPU-only, `cs` partition; jobs 19714830, 19718559, 19724850).

Per-sample cost, single process, data staged to `$SLURM_TMPDIR`:

| Stage | Cost / sample |
|---|---|
| Sampling + zarr read, no augmentation | 0.073 s |
| simple (mirror/transpose) | ~0 |
| channel permute | 8 ms |
| intensity | 23 ms |
| overlay (when applied) | 95 ms |
| **RandomHue** (when applied) | **0.79 s** |
| **elastic** (spacing 20, ±45°, subsample 4) | **0.95 s** |
| Full chain, `basic_fmaps24_400k` | 0.116 s |
| Full chain, `basic-ppp-noenc-cp` | 0.133 s |

- **Augmentations weren't the bottleneck** for the cancelled configs (elastic,
  overlay and hue were all off). Hue and elastic matter only in configs that
  enable them (e.g. `default_l40s`, hue p=0.3).
- **The workers keep up easily:** with 32 workers, a batch of 18 needs ~0.07 s of
  worker time, yet the loop got one batch per **1.0–1.16 s**. That gap is the
  main-process handoff.
- **Capping threads** (`OMP/MKL/OPENBLAS/BLOSC_NTHREADS=1`) made it worse:
  12.95 vs 17.8 samples/s. Not applied.
- **TorchIO** was rejected. Its workers hand batches back the same pickled way,
  its patch queue loads whole volumes, its elastic runs on the CPU, and it can't
  express PPP's mask/overlap sampling or padded GT context. Not installed in `ppp`.

## 4. Fix: `ThreadedPrefetch`

File: `PatchPerPix/experiments/flylight/setups/setup01/threaded_prefetch.py`,
added to `train.py` right after `gp.Stack` in the training pipeline:

```python
+ gp.Stack(config["batch_size"]) +
# unpickle + stack the next batch while the GPU trains on this one
(ThreadedPrefetch(num_batches=config.get("prefetch_batches", 2))
 if config.get("prefetch_batches", 2) > 0 else gp.NoOp())
```

How it works:

- On its first request it starts a background thread that keeps requesting the
  same batch from upstream (fresh random seed each time) and holds up to
  `num_batches` stacked batches in a queue.
- Training requests are served from that queue immediately; the unpickle+stack
  work happens while the GPU computes, because the main thread releases the
  GIL while it waits on CUDA.
- A request that differs from the prefetched one (seed ignored) is served
  synchronously, as before.
- A lock serializes all upstream access, since gunpowder nodes aren't thread-safe.
- Errors in the thread are passed to the training loop and raised there, and a
  dead thread is detected, so failures don't hang the job.
- `teardown` stops the thread and resets its state.

```
background:  [unpickle+stack][unpickle+stack][unpickle+stack]...
main thread: [GPU step][GPU step][GPU step]...   ← takes a ready batch instantly
```

Config (TOML `[training]`):

| Key | Default | Meaning |
|---|---|---|
| `prefetch_batches` | 2 | Stacked batches held ready; `0` disables |

Cost: ~1.3 GB extra host RAM at batch_size 18 (2 × ~650 MB). Data,
augmentations and model are unchanged; only when batches are prepared changes.

## 5. Verification

CPU benchmark with a simulated 3.5 s GPU step (`time.sleep`), config
`basic_fmaps24_400k`, 32 workers, batch 18 (job 19724850):

| | GPU wait per step |
|---|---|
| No prefetch | 1.26 s (27% of step) |
| `prefetch_batches = 2` | 0.00 s (max 0.03 s) |

Re-run it:

```bash
PREFETCH_TEST=1 BENCH_DATASETS=completely sbatch --time=00:45:00 \
  sbatch/ppp/tools/bench_ppp_dataloader_sbatch.sh
```

Caveat: `sleep` frees the GIL completely; a real forward pass holds it for
part of the step, so the real gain is likely somewhat below 27%.

**Still to do:** a real GPU run. Compare `gpu_log/ppp/gpu_usage_log_<exp>_<job>.csv`
mean util and `Batch: iteration=…, time=` against §1.

## 6. Related fixes and remaining levers

- **Dataset staging bug (fixed):** `basic-ppp-noenc-cp` (job 19678616) never
  trained on `partly`. `ppp_submit.sh` defaulted `DATASETS=completely`, so only
  those tars were staged and `run_ppp.py` just warned. Now `DATASETS` defaults to
  the TOML `*_datasets` lists (`sbatch/ppp/ppp_datasets.py`), `--datasets` is an
  override, and `run_ppp.py` raises `FileNotFoundError` for a missing listed
  dataset dir.
- **Profiling:** `profiling = 2000/10000` never printed before jobs died. Use
  ~`200` in overlays so per-node timings land in the log.
- **Hue on GPU:** `RandomHue` (0.79 s per sample on CPU) is a few tensor ops on
  the GPU. Worth it only for configs with `probability_hue > 0`.
- **Elastic:** the only CPU augmentation that needs a redesign (padded crop + GPU
  `grid_sample`) if elastic-heavy configs become starved.
- **`RandomLocation(min_masked)` startup:** each masked source builds a
  full-volume integral image at pipeline build (minutes and tens of GB with
  `oversample_complete × many files`). This affects startup, not steady state.
