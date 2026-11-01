> **Historical / archived.** Current PatchPerPix ops and canonical presets: [`ppp_overview.md`](../ppp_overview.md). Keep this file for background only; do not treat recipes here as the live path.
>
> ---

# PatchPerPix training sampling — slow startup, FG points, source mix

How `train.py` chooses training crops, why a run can sit at 0% GPU for ~2 h before
iteration 1, and what the sampling-related config keys actually do.

Code references are to
[`PatchPerPix/experiments/flylight/setups/setup01/train.py`](../../../PatchPerPix/experiments/flylight/setups/setup01/train.py)
(local, modified copy). `gp.CreateMask` lives in the gunpowder fork inside the `ppp` overlay
(`/ext3/miniforge3/envs/ppp/lib/python3.9/site-packages/gunpowder/nodes/create_mask.py`).

---

## 1. Symptom: "stuck" training job (job 19760845, `basic-ppp-noenc-cp`, 2026-10-14)

- `metrics/ppp/basic-ppp-noenc-cp/run.log` showed only repeated pairs of
  ```
  INFO:gunpowder.nodes.random_location:requesting complete mask...
  INFO:gunpowder.nodes.random_location:allocating mask integral array...
  ```
- `gpu_log/ppp/gpu_usage_log_basic-ppp-noenc-cp_19760845.csv`: 0% util, 1175 MiB used (model
  initialised, no training step yet).
- On the node: one `python3` at ~100% of a single core, RSS ~58 GB and growing; node otherwise idle.

**Not hung — still building the input pipeline.** 103 sources; ~67 masks done after ~80 min,
~70 s each, serial, in the main process. Expected time to first iteration ≈ 2 h.

**Risk:** public GPU partitions cancel persistently low-utilization jobs, so a 2 h
CPU-only startup can get the job killed before it trains.

### Why it is slow

With `sample_fg_points = false`, every foreground source gets
`gp.RandomLocation(mask=gt_sample_mask, min_masked=...)` (`train.py:835`). At setup each one:

1. requests the **entire** `volumes/gt_fg_rm_5` (431×687×684 ≈ 200M voxels, f4) — "requesting complete mask";
2. builds an **integral array** of it — "allocating mask integral array";
3. keeps that array in RAM for the whole run (~0.9 GB/source observed).

`oversample_complete = 3` duplicates every complete file in the source list, so the 18
complete volumes are each read and integrated **3×** (~36 redundant passes, ~40 min).

### Fix options

1. **Point sampling** (`sample_fg_points = true`, §3) — removes the per-source mask scan entirely. Preferred.
2. **De-duplicate oversampled sources** — one source per file, weight complete files via
   `gp.RandomProvider(probabilities=...)` instead of list repetition.
3. Parallelise or cache the mask setup (thread/process pool, or precomputed on disk).

---

## 2. What an integral array is

A summed-area table: same shape as the mask, each voxel holds the sum of the mask over the box
from the origin to that voxel:

```
I[z,y,x] = sum(mask[0..z, 0..y, 0..x])        # ≈ mask.cumsum(0).cumsum(1).cumsum(2)
```

The sum of the mask inside any box then costs **8 lookups** (3D inclusion–exclusion on the box
corners), independent of box size.

`RandomLocation(mask=..., min_masked=...)` uses it as follows:

- **setup (once per source):** load full mask, build `I` → expensive (the ~70 s).
- **per batch:** pick a random crop, compute masked fraction via `I`; reject and retry if below
  `min_masked` (0.0001) → cheap.
- **memory:** `I` stays resident, one value per voxel per source.

---

## 3. Point sampling (`sample_fg_points = true`)

Precompute foreground coordinates once, offline; at train time pick one and place the crop around it.

### Offline: `prepare_fg_points_csv.py`

[`PatchPerPix/experiments/flylight/prepare_fg_points_csv.py`](../../../PatchPerPix/experiments/flylight/prepare_fg_points_csv.py)
— **local script, not in upstream PatchPerPix** (untracked in the `PatchPerPix/` clone, created 2026-09-29).

- Reads `volumes/gt_fg_rm_5`, keeps FG voxels whose z, y, x are all multiples of `--stride`
  (default 4), writes `z y x` per line (space-separated, no header) to
  `fisbe/fg_points/<dataset>/<sample>.csv` (one subfolder per dataset, taken from the first component of each `--splits` entry, because datasets such as `synth_vh`/`synth_vh_u8` reuse `brain_XXXXX` names). Example: `R38F04-20181005_63_G3.csv` → 7,422 points.
- Default `--splits` is `completely/train,completely/val,completely/test` — **partly is not included.**

Status as of 2026-10-14:

| Split | CSVs present |
|---|---|
| `completely/train` | 18 / 18 |
| `partly/train` | 0 / 43 |

Because `add_partly = true`, every partly source also needs a CSV. Generate them first
(CPU-only, ~1 min/volume; run in the `ppp` container on a compute node):

```bash
python PatchPerPix/experiments/flylight/prepare_fg_points_csv.py \
  --splits partly/train,partly/val,partly/test
```

### Config

```toml
sample_fg_points = true
fg_csv_folder = "/scratch/wmz2007/neuroinfo_fruitfly/fisbe/fg_points"
```

In list mode (`train_datasets = [...]`) every file's CSV is looked up in `fg_csv_folder/<dataset>/` (dataset = the zarr's grandparent dir name; missing CSV raises at startup)
(`train.py:638-645`). This works because complete and partly sample names don't collide. The path
is absolute, so `$SLURM_TMPDIR` data staging doesn't affect it.

### At train time (`train.py:816-832`)

- A `gp.CsvPointsSource(fg_csvs[fn], gt_fg_points)` is merged with the zarr source.
- The foreground location node becomes
  `gp.RandomLocation(ensure_nonempty=gt_fg_points, point_balance_radius=point_balance_radius_overlap, copy_roi_from=gt_instances)`.
- gunpowder loads only the points into a KD-tree (milliseconds), then per batch picks a point and
  shifts the crop randomly so the point lies inside it. `point_balance_radius` down-weights points
  with many neighbours, so dense regions don't dominate.

### Equivalence to the mask branch

- Same practical guarantee: "crop contains ≥1 FG grid point" vs. "crop is ≥0.01% FG".
- Same source of truth: both use `gt_fg_rm_5`. In partly volumes that means **labelled neurons only**.
- Small difference: a crop that only grazes a neurite between stride-4 grid points can be missed;
  negligible at 140³ crops.

### Provenance of the original CSVs

Upstream `train.py` already reads FG-point CSVs, but **upstream never shipped a generator**. The
original configs (`original_default.toml`, `default.toml`) point to
`/nrs/saalfeld/maisl/data/flylight/flylight_complete/fg_points` — the authors' Janelia cluster
storage, produced by internal tooling. The overlap CSVs (`overlap_csv_folder`, for
`probability_overlap`) are in the same situation and don't exist here, hence the log warning
"no overlap CSVs found; using foreground sampling only". Whether the authors subsampled with stride 4
is unknown.

---

## 4. Source sets and how they're mixed

`train.py` builds up to three source sets; `train.py:903-930` mixes them with
`gp.RandomProvider(probabilities=...)`. gunpowder normalises the probabilities.

| Set | Weight key | Crop placement | Built for |
|---|---|---|---|
| Foreground | `probability_fg` (0.7) | must contain FG (mask or points) | every file |
| Overlap | `probability_overlap` (0.25) | near neuron-overlap points from CSV | only files with an overlap CSV — **none here, set dropped** |
| Random | `probability_random` (0.05) | plain `gp.RandomLocation()`, anywhere in padded volume | complete files; partly only if `mask_bg_weight != 0` **and** `use_gt_extra = true` (`train.py:781-785`) |

Effective mix in the current runs: [0.7, 0.05] → **≈93% foreground, ≈7% random**.

**Random crops** are mostly background; they teach the net to predict empty space and suppress
false positives. Partly volumes are excluded by default because their "background" can contain
unannotated neurons, so random crops there would teach wrong negatives.

**Complete + partly with random only on complete** is therefore the *current default* behaviour
(`use_gt_extra = false`). This was confirmed in the stdout of earlier job 19587924
(`basic-ppp-noenc`, same sampling settings): all `random source:` lines are `completely/...`.

Within the foreground set, a source is drawn uniformly from the source list (18 complete ×3 +
partly files), so roughly half of foreground crops come from partly volumes.

Random sources never use the mask or the points, so switching to point sampling doesn't affect them.

---

## 5. Key reference

| Key | What it does | Notes |
|---|---|---|
| `add_partly` | Adds the partly dataset to training; also enables `CreateMask` (loss mask) in every source pipeline | This is the "use partly data" switch |
| `oversample_complete` | Repeats each complete file N× in the source list | Repeats the mask setup N× in the mask branch |
| `sample_fg_points` | `true`: FG crops placed from precomputed CSV points. `false`: from `gt_fg_rm_5` mask + integral array | Only affects foreground sources |
| `fg_csv_folder` | Folder of FG-point CSVs (only read when `sample_fg_points = true`) | Unrelated to `gt_fg_extra` |
| `sampling.min_masked` | Minimum FG fraction for a mask-branch crop | **Mask branch only** |
| `sampling.point_balance_radius_overlap` | Neighbour-balancing radius | **Points branch** and overlap sources only |
| `sampling.probability_fg` / `probability_random` / `probability_overlap` | Weights for mixing the source sets | Used in both branches; normalised |
| `mask_bg_weight` | Loss weight on background voxels in **partly** crops (0.01) | Complete files get `mask_source=None` → all-ones mask, so it has no effect there. Also one of the two gates for partly random sources |
| `use_gt_extra` | For partly files, also load `volumes/gt_fg_extra` and build a 3-channel loss mask from it | See §6. **Not** "use the partly dataset" |
| `reject_min_masked` | Optional `gp.Reject` after sampling, in both branches | Unset here → `NoOp` |

### `CreateMask` behaviour (`use_gt_extra = false`)

- **Complete file** (`mask_source = None`): loss mask = all ones.
- **Partly file** (`mask_source = gt_numinst`): 1 where `gt_numinst > 0`, `mask_bg_weight` on background.

---

## 6. `gt_fg_extra` — not available

- **Not in FISBe.** A partly zarr (e.g. `fisbe/partly/train/R14B11-20180905_65_D2.zarr/volumes/`)
  has `raw`, `gt_instances` (FISBe) plus `raw_normalized`, `gt_instances_rm_5`, `gt_numinst`,
  `gt_fg_rm_5` (added by `prepare_fisbe_for_ppp.py`). In FISBe, unlabelled neurons in partly volumes
  are simply background.
- **Upstream references it once** (`datasets[gt_fgbg_extra] = 'volumes/gt_fg_extra'`), with no code
  that creates it and no documentation. Most likely it is from the authors' internal FlyLight data,
  like the CSVs.
- **Behaviour, inferred from `CreateMask`:** where `gt_fg_extra > gt_numinst`, channel 0 (FG/BG)
  gets weight 1 and channels 1–2 (instance-related) get weight 0. That trains "this is a neuron"
  without assigning an instance, which only makes sense if `gt_fg_extra` marks FG beyond the
  labelled instances. How it was produced (thresholded raw, coarse annotation, …) is unknown.
- **Consequence:** `use_gt_extra = true` would fail on a missing dataset unless such an array is
  generated. Keep it `false`; partly random sampling stays off.
