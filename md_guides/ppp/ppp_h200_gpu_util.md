# PPP on H200: fixing low GPU utilization (`our-noenc-cs`)

Written 2026-10-21. Companion to [`ppp_overview.md`](ppp_overview.md) §7 and the archived
[`historical/ppp_gpu_util_prefetch.md`](historical/ppp_gpu_util_prefetch.md).

## TL;DR

H200 partitions cap jobs at 16 CPUs, and the old H200 setup (bs 63, 16 workers) ran at ~33% GPU util.
The cause was **not CPU count**. It was bytes moved through `PreCache`'s `multiprocessing` pipes
into the single main process, plus 16 workers oversubscribing 16 CPUs. Shipping `RAW` as float16
and dropping to 12 workers / `cache_size = 128` brought a 45-minute smoke run to **85% GPU util**
(same as the L40S reference).

Launch:

```bash
./sbatch/ppp/ppp_submit_train.sh our-noenc-cs --base original_default.toml \
  --config our-noenc-cs-h200.toml --partition h200_public --cpus 16 --mem 500g \
  --id <exp_id> --time 24:00:00
```

## Jobs referenced

| Job | Partition | CPUs | Batch | What | Result |
|-----|-----------|------|-------|------|--------|
| 20330947 | l40s_public | 32 | 18 | L40S reference (late resume, ~iter 83k) | 85.2% util; step 2.59 s == batch 2.59 s |
| 20070698 | h200_public | 16 | 63 | Old H200 setup | 33.5% util; step 3.6 s, batch 6–19 s |
| 20488811 | h200_public | 16 | 63 | Bench, fp32, 16 workers | 6.6 samples/s |
| 20488812 | h200_public | 16 | 63 | Bench, fp16, 16 workers | 11.1 samples/s |
| 20488813 | h200_public | 16 | 63 | Bench, fp16, 12 workers | 29.2 samples/s |
| 20535976 | h200_public | 16 | 63 | 45-min smoke train with the fix | 85.3% util after startup |

## Problem

L40S (32 CPUs, bs 18) ran at 85% util. On H200 the GPU can consume ~17.5 samples/s
(3.6 s per bs-63 step), but the old setup only supplied a fraction of that and util fell to ~33%.
The puzzle: CPU efficiency on the L40S job was already very low, so "not enough CPUs" didn't fit.

## Diagnosis

1. **L40S was never CPU-bound.** 20330947 used 55 CPU-h over 24 h on 32 cores, about 2.3 cores busy.
2. **Workers are cheap.** Profiling per sample: ZarrSource 0.05 s, IntensityAugment 0.02 s, the rest ~0.
   At 17.5 samples/s that's ~3.5 cores of real work, so 16 CPUs are enough for the compute itself.
3. **The handoff is the limit.** One sample pickles to ~34.7 MiB: `RAW` float32 `(3,140,140,140)` is
   31.4 MiB of it. A bs-63 batch is ~2.2 GB per step through mp pipes into one process.
   H200 profiling: `Stack` mean 4.19 s (max 14 s) ≈ 63 × `PreCache` 0.07 s per sample.
4. **L40S hid the same cost.** 18 × 0.07 = 1.3 s per batch fits inside a 2.6 s step, and
   `ThreadedPrefetch` overlaps it.
5. **Oversubscription made it worse.** 16 workers + main process + prefetch thread + train thread
   on 16 CPUs: this is where the 6–19 s stalls came from.

The old TOML comment ("supply-limited, ~5 samples/s") was right about the symptom, but "more
workers" would not have fixed it.

## Changes

| File | Change |
|------|--------|
| `PatchPerPix/experiments/flylight/setups/setup01/train.py` | New `[training] raw_transport_fp16` (default false): `gp.Cast(raw, np.float16)` just before `gp.PreCache` |
| `.../setup01/torch_model.py` | `forward` starts with `raw = raw.float()` (no-op for float32) |
| `.../setup01/our-noenc-cs-h200.toml` (new) | H200 overlay: `batch_size = 63`, `num_workers = 12`, `cache_size = 128`, `raw_transport_fp16 = true` |
| `.../setup01/our-noenc-cs.toml` | H200 comment block rewritten with the measured numbers; active settings stay L40S |
| `sbatch/ppp/tools/bench_ppp_dataloader.py` | New `--raw-fp16` flag for the throughput test |
| `md_guides/ppp/ppp_overview.md` | One-line H200 note in §7 |

The L40S preset is unchanged, so job 20330947 stays reproducible. The H200 settings live in a
separate overlay passed with `--config`.

## Results

### Benchmark (no GPU work, bs 63, `synth_vh`, 16 CPUs)

| Workers | RAW | s/batch (max) | samples/s |
|---------|-----|---------------|-----------|
| 16 | fp32 | 9.58 (16.2) | 6.6 |
| 16 | fp16 | 5.67 (8.5) | 11.1 |
| 12 | fp16 | 2.16 (2.4) | 29.2 |

Target was ≥ 20 samples/s against the GPU's 17.5/s.

### Smoke train 20535976 (45 min, ended in expected `TIMEOUT`)

- GPU util: 77% over the whole log, **85.3%** over the last 60% (startup dominated the dip).
- Batch time mean 3.69 s vs train step ~3.33 s; only 2 of 684 batches above 6 s.
- Profiling per 500 iterations (old H200 run in brackets):
  `Stack` 0.44 s (4.19), `PreCache` 0.02 s (0.07), `ThreadedPrefetch` 0.01 s (4.85).
- Early loss in the same range as the fp32 L40S run 19859776: ~0.049–0.060 at iteration ~680
  vs 0.060–0.096. Batch sizes differ (63 vs 18), so this is only a rough sanity check.

## Problems hit along the way

- **Bench wrapper defaults:** `sbatch/ppp/tools/bench_ppp_dataloader_sbatch.sh` hardcodes partition `cs` and 32 CPUs.
  I overrode them on the `sbatch` command line. `h200_public` also needs `--gres=gpu:1`, so the
  CPU-only bench holds an idle GPU there (could trip the idle-util logger).
- **Bench step C is wrong for this use:** it re-runs throughput with defaults (bs 18, 32 workers on
  16 CPUs). Ignore it, or `scancel` after step B.
- **Bench "handoff" line is misleading with `--raw-fp16`:** it measures the uncast batch, so it
  never shows the fp16 savings.
- **Noisy nodes:** the fp32/16-worker baseline measured 6.6 samples/s here but 15.4 in an earlier bench;
  pickle round-trip was 23–27 ms vs 6 ms. Looks like memory-bandwidth contention. Trust the
  ordering of results more than the absolute numbers.
- **GPU log name:** the train job's CSV is `gpu_log/ppp/gpu_usage_log_<exp_id>_<jobid>.csv`
  (here `our-noenc-cs-h200-smoke_20535976`), not the preset name.
- **Overlay merge:** `cache_size` is in `[training]`. A duplicate key in the same TOML table is an
  error, so override it through the overlay rather than adding a second line to the preset.

## Insights

- Low CPU efficiency on one machine says nothing about whether the pipeline is CPU-bound on another.
  Look at **per-sample cost × required rate** and at **bytes/sec through the single consumer**.
- The binding constraint was the serial main-process consumer (pipe transfer + `np.stack`), so
  halving bytes helped, and *removing oversubscription* helped even more (11.1 → 29.2 samples/s).
- Rule of thumb: keep `num_workers` at roughly CPUs − 4 when the main process is also doing real
  work. Workers here are mostly idle waiting on a full queue.
- `cache_size = 400` held up to ~14 GB of 35 MiB samples in flight for no benefit; 128 is enough
  for two bs-63 batches' worth of lookahead.
- Bigger batch does not raise supply. Supply is samples/s, so a larger batch only helps the GPU if
  the pipeline can deliver samples faster than it consumes them.

## Caveats / not yet verified

- Only a 45-minute window. Check the GPU log again around validation (every 2000 iterations) and
  checkpointing in the first long run.
- fp16 transport slightly changes training inputs (~1e-3 error on values in [-1, 1]) relative to the
  fp32 L40S runs. Keep that in mind when comparing final metrics across the two.
- Snapshot writing with float16 `RAW` ran without error up to iteration 684; `snapshots = 2000`
  was never reached in the smoke run.
- If util drops later: try 10 workers, `prefetch_batches = 3`, or shared-memory transport
  instead of mp pipes (not implemented).
- `our-noenc-cs` is still commented out in `PRESET_TABLE` for `h200_public`
  (`sbatch/ppp/ppp_submit_lib.sh`); the preset default stays L40S/32 CPUs, so H200 launches need the
  explicit flags above.
