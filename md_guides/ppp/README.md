# PatchPerPix guides

Guides for running PatchPerPix (PPP) on FISBe on NYU Greene. Code architecture lives in
[`PatchPerPix/CODEBASE.md`](../../PatchPerPix/CODEBASE.md); the config catalog lives in
[`setup01/configs/README.md`](../../PatchPerPix/experiments/flylight/setups/setup01/configs/README.md).
Experiment plans and results live in [`md_guides/experiments/`](../experiments/).

## Which guide to read

| I want to… | Read |
|---|---|
| Train, run inference, evaluate, pick a preset, chain checkpoints, fine-tune | [`ppp_overview.md`](ppp_overview.md) |
| Prepare or refresh a dataset under `fisbe/` (PPP arrays, FG points, stage tars, sqf packing, synth builds) | [`ppp_prep.md`](ppp_prep.md) |
| Run a trained model on new data that has no GT (e.g. himanshu) | [`ppp_new_dataset_inference.md`](ppp_new_dataset_inference.md) |
| Choose GPU / batch / worker settings, or find out why training is slow | [`ppp_train_performance.md`](ppp_train_performance.md) |
| Understand or debug the GPU labgraph (patch-graph) stage | [`ppp_labgraph_speedup.md`](ppp_labgraph_speedup.md) |

## Current guides

| File | Kind | Last updated |
|---|---|---|
| [`ppp_overview.md`](ppp_overview.md) | Runbook: pipeline, launchers, presets, config stacking, eval protocol, train ops | 2026-11-06 |
| [`ppp_prep.md`](ppp_prep.md) | Reference: every `sbatch/ppp/prep` script with its variables | 2026-10-25 |
| [`ppp_new_dataset_inference.md`](ppp_new_dataset_inference.md) | Runbook: inference-only datasets | 2026-11-06 |
| [`ppp_train_performance.md`](ppp_train_performance.md) | Findings: training speed on L40S/H200, noenc and decoder; recommended settings; profiling tools | 2026-11-06 |
| [`ppp_labgraph_speedup.md`](ppp_labgraph_speedup.md) | Findings: labgraph GPU-util fix (device-memory consensus, kernel cache, vectorized pairs) | 2026-11-06 |

## Open work

- Rerun the L40S fp16-raw speed test and decide whether L40S presets get `raw_transport_fp16`
  ([`ppp_train_performance.md`](ppp_train_performance.md) §5, §7).
- Commit and benchmark the decoder `upsample_conv_first` change (same doc, §4).
- Make it easy to try different dataset combinations per experiment (data overlays under `configs/data/`,
  experiment overlays under `configs/exp/`). This was the goal of the removed `ppp_pipeline_update.md` stub.
- Cleanup: `.claude/worktrees/ppp-noenc-speed` (7 GB smoke-run dir) and
  `metrics/ppp/our-noenc-cs/test/_old_labgraph_20622991/`.

## Archived guides ([`historical/`](historical/))

Background only. Recipes, experiment ids and file paths in these may be out of date; the current guides above
win where they disagree.

| File | Was about | Superseded by |
|---|---|---|
| [`ppp_noenc_train_speedup.md`](historical/ppp_noenc_train_speedup.md) | noenc step profile, `fast_affinities`, `channels_last`, L40S vs H200 (2026-11-05) | `ppp_train_performance.md` |
| [`ppp_h200_gpu_util.md`](historical/ppp_h200_gpu_util.md) | H200 16-CPU cap, fp16 raw transport (2026-10-21) | `ppp_train_performance.md` |
| [`PPP_TRAIN_SPEEDUP_GUIDE.md`](historical/PPP_TRAIN_SPEEDUP_GUIDE.md) | Decoder affinity vectorization (2026-10-26) | `ppp_train_performance.md` |
| [`ppp_gpu_util_prefetch.md`](historical/ppp_gpu_util_prefetch.md) | Batch-handoff idle, `ThreadedPrefetch` root cause | `ppp_train_performance.md`, `ppp_overview.md` §6 |
| [`ppp_train_sampling.md`](historical/ppp_train_sampling.md) | Crop sampling, FG points, startup cost | `ppp_overview.md` §6 |
| [`ppp_config_defaults.md`](historical/ppp_config_defaults.md) | Per-key TOML dictionary (older defaults file) | live TOMLs + `configs/README.md` |
| [`ppp_evaluation.md`](historical/ppp_evaluation.md) | Deep evalinstseg metric semantics | `ppp_overview.md` §5 (summary) |
| [`ppp_list_mode_unify.md`](historical/ppp_list_mode_unify.md) | Val/test list mode + multi-dataset (implemented) | `ppp_overview.md` §3 |
| [`ppp_sbatch_consolidation.md`](historical/ppp_sbatch_consolidation.md) | Submitter consolidation plan (implemented) | `ppp_overview.md` §3 |
| [`ppp_recommendations.md`](historical/ppp_recommendations.md) | Post-mortem after early smoke / long runs | — |
| [`ppp_performance_plan.md`](historical/ppp_performance_plan.md) | Layer A/B/C quality-improvement plan | `ppp_overview.md` §7 |
| [`ppp_layer_a_prediction.md`](historical/ppp_layer_a_prediction.md) | Prediction-quality runbook (done) | — |
| [`ppp_layer_b_merge.md`](historical/ppp_layer_b_merge.md) | VI merge sweeps (closed; overlays in `configs/archive/vi_merge_*.toml`) | — |
