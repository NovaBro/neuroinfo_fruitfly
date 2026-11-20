# `simple_v2` data generation guide

How the second simple synthetic tube dataset (`simple_v2`) was generated, stored and checked, and
how to run or rebuild it. `simple_v2` is `simple_v1` (rev 3) with a wider range of **brightness,
thickness, length and branching**. The pipeline, noise, storage layout and training hook-up are the
same as for `simple_v1`.

This guide describes the code in the working tree on 2026-11-20. The `simple_v2` additions sit on
top of commit `b20b1bf` and are **not committed yet**. Where nothing changed, it points to the
[`simple_v1` guide](simple_v1_generation.md) (cited below as "v1 §n") instead of repeating it.
The design record is §9 of [`simple_dataset_plan.md`](simple_dataset_plan.md). When the docs
disagree, the code wins, then this guide.

| | |
|---|---|
| Name / mount point | `simple_v2`, parts mount at `/simple_v2` |
| Volume | 400 × 600 × 600 (Z, Y, X) voxels; attrs say 0.44 µm isotropic (nominal) |
| Raw | 3 channels, `uint8` with values 0–15 ("4-bit"), `clipmax = 15` attr |
| Neurons | U{10..30} per volume (mean 20.0 in the build), one connected piece each |
| Brightness | base level U{3..15} per neuron, ±2 per branch, drift along the cable, clipped to 1–15 |
| Thickness | root radius log-uniform 1.0–5.0 voxels, taper 0–50 %, floor 1.0 |
| Splits | 50,000 train / 100 val / 100 test (`tube_00000` … `tube_50199`) |
| Storage | 20 SquashFS parts in `fisbe/sqf/simple_v2_src/`, 141 GB, ~2.8 MB per sample |
| Code | `synthetic/simple_tubes/` with `--preset simple_v2` (`generate.py PRESETS`) |
| Build job | `sbatch/ppp/prep/simple/generate_simple_sbatch.sh` with `DATASET=simple_v2` (CPU only) |
| Training config | `setup01/configs/exp/dsexp/dsexp_simple_v2.toml` |

---

## 1. What changed from `simple_v1`

| | `simple_v1` (rev 3) | `simple_v2` |
|---|---|---|
| Neuron brightness | one level per neuron, U{10..15}, the same on every voxel | base U{3..15}; each branch adds U{−2..2}; a sinusoidal drift along the cable (amplitude 0–1.5 levels, period 50–200 voxels); clipped to 1–15; a voxel takes the max over the branches covering it (§3.5) |
| Root radius | U(1.5, 3.0) | log-uniform 1.0–5.0 |
| Radius floor | 1.5 | 1.0 |
| Taper along a branch | 10–30 % | 0–50 % |
| Root length | 10–30 control points, 12–20 voxels apart | 4–60 control points, 8–24 voxels apart |
| Branching | Poisson λ = 2.5 / 1.5 / 0.8 / 0.3 by order (max order 4) | per-neuron factor, log-uniform 0.1–3×, on λ = 2.5 / 1.5 / 0.8 / 0.4 / 0.2 (max order 5) |
| Branch angle | 30–75° | 20–90° |

**Unchanged:** shape, neuron count U{10..30}, MCFO colours, random-walk turning, spline,
truncation at the first exit, acceptance rules, all noise (haze, blobs, confetti), the ground-truth
format, splits, storage layout, the build job, and GPU confetti in training.

**Same seed, same background.** Sample seeds are `base_seed · 1,000,000 + index` (v1 §3.1). The
neuron count is the first draw from the geometry stream, and all noise comes from a separate stream
with unchanged parameters. So `simple_v2/tube_k` has the **same neuron count and the same haze,
blobs and confetti** as `simple_v1/tube_k`; only the neurons differ. The noise summaries and
neuron counts match for all 200 val/test samples. A v1-vs-v2 comparison at the same index therefore
isolates the neuron changes.

**`simple_v1` is still reproducible.** The new parameters default to the v1 values and draw no extra
random numbers when they are off. `verify --regen` still reproduces `simple_v1` byte for byte
(checked on `tube_00007` and `tube_50003`, plan §9).

---

## 2. Pipeline overview

Identical to v1 §1, except that every render call passes `--preset simple_v2`:

```
generate_simple_sbatch.sh  (DATASET=simple_v2; PRESET defaults to DATASET)
 ├─ generate skeleton  → fisbe/simple_v2/   (train/ & meta/ links, loose val/test zarrs,
 │                       sample_list_per_split.txt, source_sqf.txt)
 │                       fisbe/fg_points/simple_v2 -> /simple_v2/fg_points
 └─ sbatch --array=0-19  (one task per part, 2,510 samples each)
      ├─ generate render --preset simple_v2 --verify   on $SLURM_TMPDIR, 16 workers
      │    per sample:  seed → neuron trees (v2 geometry) → rasterise, keeping each voxel's
      │                 nearest centreline point → per-branch brightness → background noise
      │                 → neurons on top (per-voxel level) → (confetti, val/test only)
      │                 → zarr with every PPP array + meta + fg-point CSV → verify_zarr()
      ├─ mksquashfs  → part_<i>.tmp.sqf     (image root simple_v2/)
      ├─ verify again through the mounted image
      └─ mv → fisbe/sqf/simple_v2_src/part_<i>.sqf

training (ppp_job.sh mounts the parts from fisbe/simple_v2/source_sqf.txt)
 └─ train.py: uint8 raw → gp.Normalize(1/15) → GPU confetti each step (allowlist simple_v2)
```

As in v1, there is no `prepare_fisbe_for_ppp.py`, pack or fg-points step.

---

## 3. How one sample is generated

`render.make_sample(index, base_seed, GeometryParams, RenderParams, NoiseParams)` with the
`simple_v2` preset (§7). All lengths are in voxels. Coordinates are continuous `(z, y, x)`, and voxel
centres sit at integers.

### 3.1 Seeds and determinism

Same as v1 §3.1: geometry, colours and levels use `default_rng(seed)`, and all noise uses
`default_rng([seed, 1])`. All per-branch brightness draws come from the geometry stream, right after
each neuron's colour and base level. Every parameter is stored in the zarr attrs, and
`verify --regen` rebuilds the sample from those stored parameters, so it needs no `--preset`.
Re-rendering `tube_50003` and `tube_50150` reproduced raw and every GT channel exactly
(2026-11-20).

### 3.2 Neuron geometry (`geometry.py`)

Each branch is built as in v1 §3.2. The v2 values are:

1. **Root start.** With probability 0.5 the root starts on a random face, pointing straight inward;
   otherwise it starts uniformly inside with a random direction (unchanged).
2. **Control points.** A persistent random walk. Each step turns by `|N(0, 20°)|` (capped at 45°) and
   moves **`U(8, 24)`** voxels. Roots get **`K ~ U{4..60}`** points, so root length varies about
   15× (v1: 3×).
3. **Spline.** Centripetal Catmull-Rom through every control point, resampled by arc length every
   0.25 voxels (unchanged).
4. **Radius.** Root radius is **log-uniform in [1.0, 5.0]** (`root_radius_log = True`), so thin and
   thick tubes are about equally common in log terms (median 2.3). It tapers linearly by
   **`U(0 %, 50 %)`** along the branch, with a floor of **1.0** voxel.
5. **Truncation at the first exit**, with a straight `r + 1` extension (unchanged). So every neuron is
   still one 26-connected piece.

**Branching** (breadth-first, as in v1, with two changes):

- Each neuron draws a **branchiness factor `b`, log-uniform in [0.1, 3.0]**, right after its root is
  accepted. A branch of order `o` gets `Poisson(b · λ_o)` children, with
  **λ = 2.5 / 1.5 / 0.8 / 0.4 / 0.2** for orders 0–4. So the maximum order is **5**. A neuron with
  `b ≈ 0.1` is usually a single unbranched cable; one with `b ≈ 3` can have hundreds of branches
  (466 at most in val/test).
- Child direction: the parent tangent rotated by **`U(20°, 90°)`**.
- Unchanged: attach point at arc-length fraction 0.15–0.85, ≥ 1 voxel inside, ≥ 10 voxels from
  other attach points (20 tries); child `K = max(2, round(U(0.3, 0.7) · K_parent))`; child radius
  `U(0.70, 0.85)` × the parent's radius at the attach point, now with floor 1.0.

The neuron's `branchiness` goes into the meta JSON next to its branch tree.

### 3.3 Rasterisation (`render.py`)

Same as v1 §3.3 (keep voxels within `r[i]` of the nearest centreline point `i`, per branch, union
over branches, hard edges). With varied brightness, `rasterize_branch(..., return_index=True)` also
returns, for every kept voxel, the index of its nearest centreline point. §3.5 uses it to look up the
voxel's level.

### 3.4 Acceptance rules

Unchanged (v1 §3.4): target `U{10..30}` neurons; redraw (up to 20 tries) if the root never enters
or the in-volume mask has fewer than 500 voxels; exactly one 26-connected component; crossings are
allowed. Short, thin v2 neurons fail the 500-voxel rule more often: 63,486 redraws in the build
(v1: 26,634), still 0 skipped.

### 3.5 Brightness (new in v2)

Each neuron still draws one of the 7 MCFO colours and a base level, now **`U{3..15}`**. Then
`branch_levels()` draws, **per branch**:

- an offset `U{−2..2}` (`branch_level_offset = 2`), independent of the parent's offset;
- a drift amplitude `A ~ U(0, 1.5)` levels, period `P ~ U(50, 200)` voxels and phase `φ ~ U(0, 2π)`.

The level at centreline point `j` (arc length `s = 0.25 · j` from the branch's first stored point,
exit extension included) is

```
level_j = clip(rint(base + offset + A · sin(2π s / P + φ)), 1, 15)
```

`voxel_levels()` gives every voxel of the neuron the level of its nearest centreline point on each
branch that covers it, and takes the **max over those branches** (so at a junction the brighter
branch wins). That per-voxel level is written on the neuron's "on" channels with a max, as in v1.

Result, over the 4,072 val/test neurons: the per-neuron level range (max − min) has median 3 and
maximum 6. Minimum levels spread evenly over 1–12 and thin out above that. 28.7 % of neurons have some
voxels at level ≤ 4, and 4.4 % are at ≤ 4 everywhere.

**Neurons can now be as dim as the background.** Haze reaches level 3, blobs 4, and layers combine
with a max. Where a level-1 to level-4 neuron runs through haze or a blob, raw can show the
background, not the neuron, while the GT keeps the neuron. This is intended ("dim neurons reach the
haze", plan §9). Outside haze and blobs the background is 0, so even a level-1 neuron is visible
there.

Layer table (compare v1 §3.5; only row 3 changed):

| # | Layer | Levels | Channels | Combine | Train samples | Val/test |
|---|---|---|---|---|---|---|
| 1 | Haze | 0..hmax, hmax ∈ {1,2,3} | all 3 (grey) | max | baked | baked |
| 2 | Distractor blobs (1–4) | 1..lmax, lmax ∈ {2,3,4} | one random colour each | max | baked | baked |
| 3 | Neurons | **per voxel, 1..15** (§3.5) | the neuron's colour | max | baked | baked |
| 4 | Bright confetti | 8..15 per spot | random colour per spot | max | GPU, per step | baked |
| 5 | Dark confetti | 0..1 per spot | all 3 | **min** | GPU, per step | baked |

### 3.6 Noise layers

Unchanged; see v1 §3.6. The draws are also identical to `simple_v1` at the same index (§1).

### 3.7 Ground truth

Same format as v1 §3.7: `gt_instances` channel `n` holds `n + 1`, `gt_numinst` counts overlaps,
and `gt_fg_rm_5` is `gt_numinst > 0` opened with `ball(1)`.

**Thin neurons largely vanish from `gt_fg_rm_5`.** The radius-1 opening removes most of a tube
thinner than ~1.5 voxels. The table shows the fraction of a neuron's GT voxels kept in `gt_fg_rm_5`,
measured on val samples 50000–50009 (202 neurons):

| Root radius | Neurons | Kept in `gt_fg_rm_5` (median, range) |
|---|---|---|
| 1.00–1.25 | 24 | 0.09 (0.02–0.31) |
| 1.25–1.50 | 25 | 0.24 (0.07–0.69) |
| 1.50–2.00 | 40 | 0.65 (0.13–0.89) |
| 2.00–3.00 | 56 | 0.89 (0.41–0.98) |
| 3.00–5.00 | 57 | 0.96 (0.61–0.99) |

`gt_instances` and `gt_numinst` keep these neurons, so the training targets include them. Crop
sampling, however, uses `gt_fg_rm_5` (`sample_mask_key`, and the fg-point CSVs are built from it), so
crops are centred on thin neurons less often. About 23 % of v2 roots are thinner than 1.5.

---

## 4. Output format

Arrays, chunking, compression, the `raw_normalized` / `gt_instances_rm_5` links and the sidecars are
the same as in v1 §4. The differences:

- **Root `.zattrs` → `simple_tubes.instances[*]`** gain `level_min` and `level_max`, the per-voxel
  level range actually written for that neuron. `level` is still there, but it is now the **base
  draw**, which can lie outside `[level_min, level_max]` (offsets and clipping). Use
  `level_min` / `level_max` for what is in raw. `simple_tubes.params` holds the v2 geometry and
  render parameters (§7), and `simple_tubes.dataset = "simple_v2"`.
- **`meta/tube_XXXXX.json`**: each instance also has `branchiness` and `branch_levels`, one
  `{offset, drift_amp, drift_period}` per branch, in branch-id order. The drift phase is not stored.
  To get exact per-voxel levels, re-render from the seed (`verify --regen` or `make_sample`).
- On disk, as v1 §4.3 with `simple_v2` substituted:

```
fisbe/sqf/simple_v2_src/part_{0..19}.sqf     image root simple_v2/ → mounts at /simple_v2
fisbe/simple_v2/                             train -> /simple_v2/train, meta -> /simple_v2/meta,
                                             loose val/ & test/ skeletons, sample_list_per_split.txt,
                                             source_sqf.txt (the 20 parts, read by ppp_job.sh)
fisbe/fg_points/simple_v2 -> /simple_v2/fg_points
```

Section `(10)` of `evaluate-instance-segmentation/assets/sample_list_per_split.txt` lists the first
1,000 train samples plus val/test, as section `(9)` does for `simple_v1`.

---

## 5. Running it

All commands run from the repo root inside the `ppp` overlay, on a compute node (v1 §2):

```bash
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --overlay env/ppp.ext3:ro $SIF /bin/bash -c \
  'source /ext3/env.sh; conda activate ppp; cd /scratch/wmz2007/neuroinfo_fruitfly; <command>'
```

### 5.1 Quick look

```bash
python -m synthetic.simple_tubes.generate render --preset simple_v2 --dataset simple_v2 \
    --out <scratch>/v2_look --meta-out <scratch>/v2_look/meta --splits 2,0,0 --count 2 \
    --shape 100,160,160 --neurons 4,8 --verify
```

`--root-radius`, `--radius-floor` and `--neurons` override the preset. Every other v2 setting can
only be changed by editing `PRESETS` in `generate.py`.

### 5.2 Dry run

```bash
DATASET=simple_v2_dry PRESET=simple_v2 TOTAL=60 SPLITS=40,10,10 PARTS=1 \
  sbatch sbatch/ppp/prep/simple/generate_simple_sbatch.sh
```

`PRESET` defaults to `DATASET`, so **a dry-run name needs `PRESET=simple_v2`**. Without it, every
array task fails at argument parsing, because `simple_v2_dry` is not a preset. Delete
`fisbe/simple_v2_dry`, `fisbe/sqf/simple_v2_dry_src` and `fisbe/fg_points/simple_v2_dry` afterwards.

### 5.3 Full build (as run)

```bash
DATASET=simple_v2 SBATCH_ACCOUNT=torch_pr_61_tandon_advanced \
  sbatch sbatch/ppp/prep/simple/generate_simple_sbatch.sh
```

All other variables keep their defaults (v1 §2.3): `TOTAL=50200`, `SPLITS=50000,100,100`,
`PARTS=20`, `SEED=0`, `CONFETTI_SPLITS=val,test`, `MIP_SPLITS=val,test`, 16 workers. The only new
variable is `PRESET` (default `${DATASET}`); the submitter exports it to the array. Job resources:
partition `cs`, 16 CPUs, 96 GB, 8 h.

### 5.4 Inspecting, verifying and regenerating

Mount the parts listed in `fisbe/simple_v2/source_sqf.txt` (all 20 for train; `part_19` alone holds
val/test), as in v1 §2.5:

```bash
OVL=$(grep -v '^#' fisbe/simple_v2/source_sqf.txt | sed 's/^/--overlay /; s/$/:ro/' | tr '\n' ' ')
singularity exec $OVL --overlay env/ppp.ext3:ro $SIF /bin/bash -c \
  'source /ext3/env.sh; conda activate ppp; cd /scratch/wmz2007/neuroinfo_fruitfly; <command>'
```

```bash
# every check on a split
python -m synthetic.simple_tubes.verify /simple_v2/val --meta-dir /simple_v2/meta \
    --fg-dir /simple_v2/fg_points --workers 8

# re-render from the stored seed + params and compare byte for byte (~40 s per sample)
python -m synthetic.simple_tubes.verify /simple_v2/val/tube_50003.zarr --regen

# regenerate one sample outside the parts
python -m synthetic.simple_tubes.generate render --preset simple_v2 --dataset simple_v2 \
    --out <scratch>/one --meta-out <scratch>/one/meta --splits 50000,100,100 \
    --confetti-splits val,test --mip-splits val,test --start 42 --count 1 --verify
```

When regenerating with `generate render`, `--preset simple_v2` is required: without it you get the
`simple_v1` neuron for that index. `--dataset simple_v2` only sets the name in the attrs (the CLI
default is `simple_v1`). As in v1, the split counts and `--confetti-splits` must match the build.

---

## 6. Training hook-up

- **Data overlay:** `configs/exp/dsexp/dsexp_simple_v2.toml` sets train and val to `simple_v2`, and
  `[training.augmentation.gpu_confetti] datasets = ["simple_v2"]`. Pair it with `dsexp_pre.toml` to
  also validate on `completely`. It is `dsexp_simple_v1.toml` with the name swapped.
- **GPU confetti:** unchanged (v1 §5). Train samples are stored without confetti; val/test have it
  baked in.
- **Registered in:** `run_ppp.py KNOWN_DATASETS`, eval sample list section `(10)`, and the web viewer
  (`ppp_loader.py`, `sample_list.py`, `zarr_reader.py`, `client.ts`). `sbatch/web/web.sh` loops over
  `simple_v1` and `simple_v2` and mounts 40 SquashFS images.
- **No experiment uses `simple_v2` yet.** experiment_v4 trains on `simple_v1`.

---

## 7. Verification changes (`verify.py`)

All v1 checks (v1 §6) still run. One check changed. When any instance has `level_min`, the raw vs GT
check becomes "on every voxel of neuron n, each 'on' channel is ≥ `level_min`". The val/test
dark-confetti exception (≤ 1) still applies. The exact `raw == max(level × mask)` comparison with
noise off is skipped for varied brightness, and `--regen` does the exact comparison instead.

---

## 8. Parameter reference (`generate.py PRESETS`)

`simple_v1` is the dataclass defaults. `simple_v2` overrides the fields below; all others keep the
v1 values listed in v1 §7.

**`GeometryParams`**

| Field | `simple_v1` | `simple_v2` | CLI |
|---|---|---|---|
| `step_len` | `(12, 20)` | `(8, 24)` | |
| `root_points` | `(10, 30)` | `(4, 60)` | |
| `child_lambda` | `(2.5, 1.5, 0.8, 0.3)` | `(2.5, 1.5, 0.8, 0.4, 0.2)` | |
| `branch_angle_deg` | `(30, 75)` | `(20, 90)` | |
| `root_radius` | `(1.5, 3.0)` | `(1.0, 5.0)` | `--root-radius lo,hi` |
| `root_radius_log` *(new)* | `False` (uniform) | `True` (log-uniform) | |
| `taper` | `(0.1, 0.3)` | `(0.0, 0.5)` | |
| `radius_floor` | `1.5` | `1.0` | `--radius-floor` |
| `branchiness` *(new)* | `None` (factor 1) | `(0.1, 3.0)`, log-uniform per neuron | |

**`RenderParams`**

| Field | `simple_v1` | `simple_v2` | CLI |
|---|---|---|---|
| `level` | `(10, 15)` | `(3, 15)` (base level) | |
| `branch_level_offset` *(new)* | `0` | `2` (U{−2..2} per branch) | |
| `drift_levels` *(new)* | `(0, 0)` | `(0, 1.5)` | |
| `drift_period` *(new)* | `(50, 200)` (unused) | `(50, 200)` | |
| `level_floor` *(new)* | `1` (unused) | `1` | |

Brightness variation is on (`RenderParams.varied`) when `branch_level_offset > 0` or
`drift_levels[1] > 0`. `NoiseParams` is identical in both presets.

---

## 9. Build record (2026-11-19/20)

| | |
|---|---|
| Full-size test | job 23791375: 32 volumes, 16 workers, 32/32 verified, 30–37 s per volume |
| Jobs | submitter 23791579, array 23791709 (20 tasks, `torch_pr_61_tandon_advanced`), submitted 2026-11-19 21:41 |
| Samples | 50,200 rendered, 0 failed verify (at generation and through the mount) |
| Neurons | 1,004,362 total (the same as `simple_v1`, §1), 10–30 per volume, mean 20.0; 63,486 redraws, 0 skipped |
| Time | 33.2 s mean per sample per worker (18.9–81.7 s); render step 79–102 min per part; 3:03–4:08 h per task; all done 2026-11-20 01:51 |
| Size | 141 GB in 20 parts (7.0–7.1 GB each, `part_19` 7.8 GB with val/test); v1 was 123 GB |
| Container view | 50,000 / 100 / 100 zarrs, 50,200 fg CSVs, 100,600 meta files |
| Spot checks | `verify --regen` exact on `tube_50003` and `tube_50150` (2026-11-20) |

`sacct` reports MaxRSS at the 96 GB request for every task, in both the v1 and the v2 build. No task
was killed, but don't lower `--mem`.

**v1 vs v2 on the same 200 val/test volumes** (4,072 neurons each):

| | `simple_v1` | `simple_v2` |
|---|---|---|
| Neuron voxels, % of volume | median 0.10 (0.04–0.19) | median 0.16 (0.03–0.33) |
| Voxels per neuron | median 6.1 k, p90 13.5 k, max 33 k | median 5.5 k, p90 28 k, max 215 k |
| Branches per neuron | median 9, p90 19, max 47; 8 % unbranched | median 4, p90 53, max 466; 30 % unbranched |
| Deepest branch order | 43 % reach order 4 | 30 % order 0, 23 % reach order 5 |
| Cable length per neuron (voxels) | median 635, p90 1.45 k, max 3.5 k | median 528, p90 3.4 k, max 27.5 k |
| Root radius | 1.50–3.00, median 2.26 | 1.00–5.00, median 2.31; 23 % < 1.5 |
| Branchiness factor | 1 | 0.10–3.0, median 0.55 |
| Base level | 10–15 | 3–15, median 9 |
| Level range on one neuron | 0 | median 3, max 6 |
| Neurons with any voxel at level ≤ 4 | 0 % | 28.7 % |
| Redraws per volume | mean 0.47 | mean 1.35 |
| Face-start roots | 51 % | 51 % |

---

## 10. Gotchas

Every v1 gotcha (v1 §9) still applies, including never mounting under `/synth_MCFO` and `PARTS ≤ 20`.
In addition:

- **`PRESET` follows `DATASET`.** Any build whose name is not a preset name (dry runs, variants)
  must set `PRESET=simple_v2` explicitly.
- **`generate render` defaults to the v1 preset.** Pass `--preset simple_v2 --dataset simple_v2`
  when regenerating. `verify --regen` reads the stored params and needs neither.
- **Overlay budget.** `simple_v2` alone is 20 overlays. Training on `simple_v1` + `simple_v2`
  together means 40 parts plus the env and any real-data images, close to the ~50-overlay apptainer
  limit (43 is known to work). Merge parts before mixing more datasets.
- **`level` ≠ brightness in raw.** In the attrs, `level` is the base draw. Use `level_min` /
  `level_max`, or re-render for exact per-voxel levels (the drift phase is not stored).
- **Dim neurons can be hidden by haze or blobs** (§3.5), and **thin neurons are under-sampled** by
  `gt_fg_rm_5`-based crop sampling (§3.7). Both are expected properties of the data, not bugs, but
  they matter when reading per-neuron metrics.
- **Any further change gets a new name** (`simple_v3`, …) and a new `PRESETS` entry, with new
  parameters defaulting to "off". That keeps `simple_v1` and `simple_v2` regenerating exactly.
