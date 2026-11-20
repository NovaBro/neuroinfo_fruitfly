# experiment_v2 Dataset Experiments status (hand-maintained)

Last update: 2026-11-10 21:37. Plan: [experiment_v2.md](experiment_v2.md). EXP 0 is run by a driver with its own,
auto-updated page: [dsexp_e0_status.md](dsexp_e0_status.md). EXP 1 / EXP 2 pretraining were submitted by hand, so this
page is updated manually.

## Setup (all pretraining runs)

| | value |
|---|---|
| Preset / base | `our-noenc-c1k_v1.1` + `original_default.toml` + `expv2_e1_common.toml` (FMAPs 15, torch.compile, fp16 raw, fast affinities) |
| Schedule | peak lr 4e-4, 10% linear warmup + cosine to 0; ckpt + val every 5k |
| Length | 35k its / 10 h job (`dsexp_10h.toml`; user 2026-11-09, plan said 75k / 100k). Measured 0.95–1.0 s/it on L40S, ~9.5–9.8 h |
| Validation | own val split, first 5 vols → `train/val_metrics.csv` (selects the checkpoint) + completely val, 5 vols → `train/val_metrics_completely.csv` (`[data] extra_val_datasets`) |
| Overlays | `configs/exp/dsexp/`: `dsexp_pre.toml` + `dsexp_1k_v1.1.toml` / `dsexp_swc2img.toml` / `dsexp_completely.toml` (+ `expv2_e1_n500.toml` for n=500) + `dsexp_10h.toml` |
| Account | `torch_pr_61_general` until 2026-11-10 19:54, then `torch_pr_61_tandon_priority` (fair-share 0.32 vs 0.11; general jobs sat ~420 deep in l40s_public) |
| Test (EXP x.5 / x.6, EXP 0.2) | best checkpoint, 7 completely test vols, sparse pred_affs pipeline, remove_small_components 1500 (OP-EXP 5 best) |

Splits (brain ids disjoint): 1k_v1.1 train 796 / val 109 / test 95; SWC2IMG uses each brain's 1k_v1.1 split, 788 / 105 / 93 of
986 renders; completely 18 / 5 / 7. n=1000 = all train volumes, n=500 = first 500 sorted.

## Runs

| Exp | Run | What | Jobs | Status | Best ckpt (val loss) | Test (rm1500) |
|---|---|---|---|---|---|---|
| 0.1 | `baseline_v1_completely` | from scratch on completely (18 vols) | 22462675 | done 18:05 (9 h 36 m) | 10k (0.1335) | — |
| 0.2 | `baseline_v1_completely` | test 0.1's best ckpt | 22610774–98 | done | 10k | **S 0.259**, avF1 0.169, C 0.348, nPred 14 / nGT 4.9 |
| 1.1 | `vince-data_n1000` | pretrain 1k_v1.1, 796 vols | 22451868 | done 14:35 (9 h 42 m) | 25k (0.0247) | — |
| 1.2 | `vince-data_n500` | pretrain 1k_v1.1, 500 vols | 22451872 | done 14:24 (9 h 31 m) | 25k (0.0284) | — |
| 1.3–1.6 | | finetune on completely + test | — | not started (configs ready) | | |
| 2.1 | `gaurav-data_n1000` | pretrain SWC2IMG, 788 vols | 22616361 (afterok smoke 22616358) | queued | | |
| 2.2 | `gaurav-data_n500` | pretrain SWC2IMG, 500 vols | 22616362 (afterok smoke 22616358) | queued | | |
| 2.3–2.6 | | finetune on completely + test | — | after 2.1 / 2.2 | | |

For reference: EXP 1.3 (expv2, pretrain 1k_v1.1 n1000 20k → finetune 6k) scored S 0.305 at rm1500 (OP-EXP 5.3), so synthetic
pretraining adds ~+0.05 S over the from-scratch baseline, mostly through coverage (C 0.418 vs 0.348).

## Validation curves (loss; lower is better)

| Run | val set | 5k | 10k | 15k | 20k | 25k | 30k | 35k |
|---|---|---|---|---|---|---|---|---|
| EXP 1.1 n1000 | 1k_v1.1 | 0.0390 | 0.0309 | 0.0278 | 0.0271 | **0.0247** | 0.0248 | 0.0251 |
| EXP 1.1 n1000 | completely | **0.2296** | 0.2672 | 0.2545 | 0.2567 | 0.2571 | 0.2792 | 0.2796 |
| EXP 1.2 n500 | 1k_v1.1 | 0.0541 | 0.0359 | 0.0303 | 0.0303 | **0.0284** | 0.0310 | 0.0287 |
| EXP 1.2 n500 | completely | **0.2164** | 0.2553 | 0.2818 | 0.3118 | 0.2908 | 0.2742 | 0.2827 |
| EXP 0.1 | completely | 0.1381 | **0.1335** | 0.1467 | 0.1899 | 0.2492 | 0.2917 | 0.3026 |

Synthetic val keeps improving to 25k, but completely val (real data) is best at 5k for both EXP 1 runs: longer synthetic
pretraining does not lower the real-data val loss. EXP 0 overfits its 18 training volumes after 10k.

## Finetune (EXP 1.3/1.4, 2.3/2.4): configs ready, not started

`dsexp_ft.toml` (completely train, 6k its, peak lr 5e-5, 10% warmup + cosine, ckpt + val every 1k) + `dsexp_ft_1k_v1.1.toml` /
`dsexp_ft_swc2img.toml`. Validates on completely val (selects the checkpoint, plan EXP x.5/x.6) and on the pretraining
dataset's val set (same first 5 vols as pretraining; user 2026-11-10). Seed the best pretrain ckpt with
`sbatch/ppp/expv2/seed_finetune_ckpt.py` (fresh Adam) and add an overlay with `max_iterations = N + 6000`, `start_iter = N`.
`sbatch/ppp/ppp_datasets.py --split train` now includes `extra_val_datasets`, so jobs mount the extra val set's images.

## SWC2IMG (EXP 2) data

- Source: the 4 SquashFS parts in `/scratch/gw2467/data/swc2img_1k_v1_sqf/` (read access granted 2026-11-09; the unpacked
  `/scratch/gw2467/data/swc2img_1k_v1/zarr` copy was deleted by its owner). They mount together at `/swc2img_1k_v1/{zarr,meta}`
  and are listed in `fisbe/swc2img_1k_v1/source_sqf.txt`, which `ppp_job.sh` and the pack script mount.
- Skeleton: `sbatch/ppp/prep/synth/build_swc2img_skeleton.py` (run inside the container with the 4 parts mounted).
- Prep: `PLAIN_SRC=1 SUBTRACT_MEDIAN=1 RAW_DTYPE=float16 SPLITS="train val" BATCH=20 DATASETS=swc2img_1k_v1 sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh`,
  then fg points (`SPLITS=swc2img_1k_v1/train,swc2img_1k_v1/val sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh`), then
  `GROUP=3 DATASETS=swc2img_1k_v1 sbatch sbatch/ppp/prep/merge_ppp_sqf_sbatch.sh` (45 → 15 images).
- Normalization (user 2026-11-09): subtract each channel's median (render pedestal ~2,500–2,680), clip 0..1500, / 1500.
  Background → 0, signal p99.9 ≈ 0.22. Plain clip-at-1500 would have saturated every voxel to 1.0.
- Stored float16 (float32 would be ~1.6 TB, over the /scratch quota; float16 ~0.6 GB/vol). `train.py` casts float16 raw back to
  float32 right after the source (gunpowder's IntensityAugment requires float32).

## Incidents

| When | What | Fix |
|---|---|---|
| 2026-11-09 16:06 | Unpacked SWC2IMG zarrs deleted by their owner mid-prep; `.sqf` parts not readable | cancelled prep; owner granted read on the `.sqf` parts |
| 2026-11-10 07:52 | EXP 2.1/2.2 (22453746/47) failed at container start: ~50 `--overlay` images (limit between 43 and 50) | merged 45 shards → 15 images (22549764, 22550680) |
| 2026-11-10 19:58 | EXP 2.1/2.2 (22610807/09) failed at the first batch: IntensityAugment rejects float16 raw | float16 → float32 cast in `train.py`; smoke test 22616358 before resubmitting |
| 2026-11-10 ~21:10 | gpu48 QOS per-user GPU limit briefly 0 (cluster-side; normally 16) | waited; back to 16 by 21:29 |
