# `simple_v1` data generation guide

How the simple synthetic tube dataset (`simple_v1`) is generated, stored and checked, and how
to run or rebuild it. This guide describes the code as it is now (rev 3, commit `b20b1bf`).
The design decisions, their reasons and the history of revisions 1–3 are in
[`simple_dataset_plan.md`](simple_dataset_plan.md). The two docs overlap on purpose in a few
places; when they disagree, the code wins, then this guide.

`simple_v1` is meant as the first rung of a PatchPerPix (PPP) training curriculum
(`simple_v1` → `1k_v1.1` → `completely`). Each sample has smooth branching tubes with MCFO-style
colours, stored as 4-bit intensities, plus simple background noise. There is no optics model
(no PSF, no blur, no shot noise).

| | |
|---|---|
| Name / mount point | `simple_v1`, parts mount at `/simple_v1` |
| Volume | 400 × 600 × 600 (Z, Y, X) voxels; attrs say 0.44 µm isotropic (nominal) |
| Raw | 3 channels, `uint8` with values 0–15 ("4-bit"), `clipmax = 15` attr |
| Neurons | U{10..30} per volume (mean 20.0 in the build), one connected piece each |
| Splits | 50,000 train / 100 val / 100 test (`tube_00000` … `tube_50199`) |
| Storage | 20 SquashFS parts in `fisbe/sqf/simple_v1_src/`, 123 GB, ~2.45 MB per sample |
| Code | `synthetic/simple_tubes/` (generator, verifier), `setup01/gpu_confetti.py` (training) |
| Build job | `sbatch/ppp/prep/simple/generate_simple_sbatch.sh` (CPU only) |
| Training config | `setup01/configs/exp/dsexp/dsexp_simple_v1.toml` |

---

## 1. Pipeline overview

```
generate_simple_sbatch.sh  (submitter, no array id)
 ├─ generate skeleton  → fisbe/simple_v1/  (train/ & meta/ links, loose val/test zarrs,
 │                       sample_list_per_split.txt, source_sqf.txt)
 │                       fisbe/fg_points/simple_v1 -> /simple_v1/fg_points
 └─ sbatch --array=0-19  (one task per part, 2,510 samples each)
      ├─ generate render --verify      on $SLURM_TMPDIR, 16 workers
      │    per sample:  seed → neuron trees → rasterise → background noise
      │                 → neurons on top → (confetti, val/test only)
      │                 → zarr with every PPP array + meta + fg-point CSV
      │                 → verify_zarr()
      ├─ mksquashfs  → part_<i>.tmp.sqf     (image root simple_v1/)
      ├─ verify again through the mounted image
      └─ mv → fisbe/sqf/simple_v1_src/part_<i>.sqf

training (ppp_job.sh mounts the parts from source_sqf.txt)
 └─ train.py: uint8 raw → gp.Normalize(1/15)
              → TorchTrainExt hook adds bright/dark confetti on the GPU each step
```

There is **no** `prepare_fisbe_for_ppp.py`, `pack_ppp_sqf_sbatch.sh` or fg-points step. The
generator writes every array PPP reads, plus the fg-point CSVs, directly into the parts.

---

## 2. Running it

All commands run from the repo root (`/scratch/wmz2007/neuroinfo_fruitfly`) inside the `ppp`
overlay. The generator is CPU only, but anything bigger than a toy volume should run on a
compute node (`srun` or `sbatch`), not on a login node. Dependencies (numpy, scipy, zarr,
numcodecs, scikit-image, pillow; torch for the GPU confetti check) are all in the `ppp` env.

```bash
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --overlay env/ppp.ext3:ro $SIF /bin/bash -c \
  'source /ext3/env.sh; conda activate ppp; cd /scratch/wmz2007/neuroinfo_fruitfly; <command>'
```

### 2.1 Quick look (small volume, a few seconds)

```bash
python -m synthetic.simple_tubes.generate render --out <scratch>/simple_look \
    --meta-out <scratch>/simple_look/meta --splits 2,0,0 --count 2 \
    --shape 100,160,160 --neurons 4,8 --verify
```

This writes `train/tube_00000.zarr`, `train/tube_00001.zarr` and, in `meta/`, a JSON, a skeleton
`.npz` and a MIP PNG per sample. Open the `_mip.png` files first: raw MIP on the left, GT
instances on the right. Add `--no-noise` for tubes only, or `--spline bspline` for the smoother
B-spline variant.

### 2.2 Dry run of the full job (small, separate name)

```bash
DATASET=simple_v1_dry TOTAL=60 SPLITS=40,10,10 PARTS=1 \
  sbatch sbatch/ppp/prep/simple/generate_simple_sbatch.sh
```

Delete `fisbe/simple_v1_dry`, `fisbe/sqf/simple_v1_dry_src` and `fisbe/fg_points/simple_v1_dry`
afterwards.

### 2.3 Full build

```bash
SBATCH_ACCOUNT=torch_pr_61_tandon_priority \
  sbatch sbatch/ppp/prep/simple/generate_simple_sbatch.sh
```

The same script runs in two modes:

- **Submitter** (no `SLURM_ARRAY_TASK_ID`): runs `generate skeleton`, writes `source_sqf.txt`,
  then submits itself as array `0..PARTS-1` with `--export=ALL`. `SBATCH_ACCOUNT` in the
  environment overrides the script's `#SBATCH --account` for both jobs.
- **Array task `i`**: renders samples `[i·per, (i+1)·per)` with `per = ceil(TOTAL / PARTS)` into
  `$SLURM_TMPDIR/pack/<DATASET>/`, with `--verify` on. It then packs them with `mksquashfs
  -keep-as-directory` into `part_<i>.tmp.sqf`, mounts that image and verifies every sample
  again, and only then renames it to `part_<i>.sqf`.

| Env var | Default | Meaning |
|---|---|---|
| `DATASET` | `simple_v1` | folder and mount name. Must not start with `synth_MCFO` (the script refuses). |
| `TOTAL` | `50200` | number of samples |
| `SPLITS` | `50000,100,100` | train, val, test counts; must sum to `TOTAL` |
| `PARTS` | `20` | SquashFS parts = array tasks; capped at 20 (each part is one `--overlay`) |
| `SEED` | `0` | base seed (§3.1) |
| `CONFETTI_SPLITS` | `val,test` | splits with confetti baked in; train gets it on the GPU (§5) |
| `MIP_SPLITS` | `val,test` | splits that get a MIP PNG in `meta/` |
| `OVERWRITE` | `0` | `0` skips parts that already exist, so resubmitting resumes a failed build |
| `WORKERS` | `cpus-per-task` (16) | render processes per task |

Job resources: partition `cs`, 16 CPUs, 96 GB, 8 h. With 16 workers, the render step takes
70–92 min per part and the whole task 2:45–3:31 h. Logs go to
`sbatch/ppp/logs/simple_tubes_gen-<jobid>.{out,err}`, one line per sample:
`[k/n] tube_XXXXX OK N=<kept>/<target> redraws=<r> skipped=<s> <seconds>s`.

With the default split and part counts, part `i` holds samples `2510·i … 2510·i + 2509`.
All of val and test (`50,000–50,199`) sit in `part_19`, which is why it is the largest part
(6.9 GB, others 6.1–6.2 GB).

### 2.4 After the build

Nothing else is needed. `ppp_job.sh` reads `fisbe/simple_v1/source_sqf.txt` and adds each part
as an `--overlay`. Its `need_ppp` stays 0 because no link points into `/synth_MCFO`. Training
uses the data overlay `configs/exp/dsexp/dsexp_simple_v1.toml` (train and val on `simple_v1`,
GPU confetti on). Pair it with `dsexp_pre.toml` to also validate on `completely`.

### 2.5 Inspecting, verifying and regenerating samples

The train zarrs, `meta/` and the fg points exist only inside the parts. On the host,
`fisbe/simple_v1/train` and `fisbe/simple_v1/meta` are dangling links by design. To read them,
mount the parts (all 20 for train; `part_19` alone is enough for val/test):

```bash
OVL=$(grep -v '^#' fisbe/simple_v1/source_sqf.txt | sed 's/^/--overlay /; s/$/:ro/' | tr '\n' ' ')
singularity exec $OVL --overlay env/ppp.ext3:ro $SIF /bin/bash -c \
  'source /ext3/env.sh; conda activate ppp; cd /scratch/wmz2007/neuroinfo_fruitfly;
   ls /simple_v1/train | wc -l   # 50000'
```

Inside that container:

```bash
# re-run every check on a split (or a single .zarr, or the dataset dir)
python -m synthetic.simple_tubes.verify /simple_v1/val --meta-dir /simple_v1/meta \
    --fg-dir /simple_v1/fg_points --workers 8

# also re-render from the seed and compare byte for byte (slow: renders the sample again)
python -m synthetic.simple_tubes.verify /simple_v1/train/tube_00042.zarr --regen

# regenerate one sample outside the parts (same seed, same split → same arrays)
python -m synthetic.simple_tubes.generate render --out <scratch>/one \
    --meta-out <scratch>/one/meta --splits 50000,100,100 --confetti-splits val,test \
    --start 42 --count 1 --verify
```

When regenerating, pass `--splits 50000,100,100` and `--confetti-splits val,test`. The CLI
defaults (`800,100,100`, confetti in every split) are not the build settings, and the split
decides both the output path and whether confetti is baked.

`synthetic/simple_tubes/check_gpu_confetti.py` (GPU node) compares the GPU confetti with the
numpy version and times it; `--sample <zarr> --png <file>` also saves a crop as stored and with
GPU confetti.

---

## 3. How one sample is generated

`render.make_sample(index, base_seed, GeometryParams, RenderParams, NoiseParams)`.
All lengths are in voxels. Coordinates are continuous `(z, y, x)`. Voxel centres sit at
integers, so the volume spans `[-0.5, n - 0.5]` on each axis.

### 3.1 Seeds and determinism

- Sample seed = `base_seed · 1,000,000 + index`. Geometry, colours and levels use
  `default_rng(seed)`; all noise uses a separate stream, `default_rng([seed, 1])`.
- So any sample can be regenerated alone, and the neurons do not change when only noise
  settings change. Baking confetti or not does not change haze and blobs either, because
  confetti is drawn last from the noise stream.
- Every parameter used is stored in the zarr attrs. `verify --regen` rebuilds from those stored
  parameters, not from the current defaults. (Byte-identical results assume the same numpy/scipy
  as the `ppp` env.)

### 3.2 Neuron geometry (`geometry.py`)

A neuron is a tree of branches. Each branch goes through five steps.

1. **Root start.** With probability 0.5 the root starts on a random face of the volume, pointing
   straight inward (it looks cut by the crop). Otherwise it starts uniformly inside with a random
   direction.
2. **Control points: persistent random walk.** `K` points; each step turns the direction by
   `|N(0, 20°)|` (capped at 45°) about a random perpendicular axis, then moves `U(12, 20)` voxels.
   Roots get `K ~ U{10..30}`.
3. **Spline.** A centripetal Catmull-Rom spline (α = 0.5, Barry-Goldman form) passes through every
   control point, so a child's attach point lies exactly on the curve. End tangents come from
   reflected phantom points (`2·P0 − P1`). The curve is evaluated every 0.05 voxels, then
   **resampled by arc length every 0.25 voxels**.
4. **Radius.** Root radius `U(1.5, 3.0)`. It tapers linearly by `U(10 %, 30 %)` along the branch,
   with a floor of 1.5 voxels. (PPP's `gt_fg_rm_5` uses an opening of radius 1, which would erase
   thinner tubes, and crop sampling depends on `gt_fg_rm_5`.)
5. **Truncation at the first exit.** The resampled centreline is cut at the first point that
   leaves the volume after it has been at least 1 voxel deep inside. A straight extension of
   `r + 1` voxels then continues along the exit direction, so the tube runs off the edge instead
   of ending in a rounded cap. A branch that left the volume can never come back in, so **every
   neuron is one 26-connected piece**. A face-started branch keeps its entry from the last
   outside point. A root that never gets 1 voxel deep returns `None`, and the neuron is redrawn.

**Branching** (breadth-first over the tree):

- Each branch of order 0–3 gets `Poisson(λ)` children, with λ = 2.5 / 1.5 / 0.8 / 0.3 for orders
  0 / 1 / 2 / 3. So the maximum order is 4.
- Attach point: a random resampled point of the parent's core (before the exit extension), at
  arc-length fraction 0.15–0.85, at least 1 voxel deep inside the volume, and at least 10 voxels
  from the parent's other attach points. After 20 failed tries the child is dropped.
- Child direction: the parent tangent there, rotated by `U(30°, 75°)`. The child is a new random
  walk from that point (its first step goes straight along this direction), with
  `K = max(2, round(U(0.3, 0.7) · K_parent))`.
- Child radius: `U(0.70, 0.85)` × the parent's radius at the attach point, floor 1.5.

The tree (branch id, parent id, order, `t_parent`, control points, radius, truncation) goes into
the meta JSON.

### 3.3 Rasterisation (`render.py`)

Per branch: take the voxels near the resampled centreline (rounded centreline points plus a
ball of offsets with radius `ceil(r_max) + 0.9`), query a `cKDTree` of the centreline points
for each voxel centre, and keep the voxel if its distance `d` to the nearest centreline point
`i` satisfies `d ≤ r[i]`. This gives the sub-voxel distance to a 0.25-spaced centreline with a
varying radius. The neuron mask is the union of its branches. (It is done per branch so that at a
junction the thin child does not notch the thicker parent.) Edges are hard.

Masks are kept as flat voxel indices, never as N dense volumes.

### 3.4 Acceptance rules

- Neurons per volume: target `U{10..30}`.
- A neuron is redrawn (up to 20 tries) if the root never enters the volume or its in-volume
  mask has fewer than 500 voxels. After 20 failures it is skipped (this never happened in the
  build: 26,634 redraws, 0 skipped over 1,004,362 neurons).
- Each accepted neuron must have exactly one 26-connected component, otherwise generation stops
  with a `RuntimeError` (a generator bug, since truncation makes it impossible).
- Crossings between neurons are not rejected. They are what the model has to learn.

### 3.5 Intensity composition

Each neuron draws one of the 7 non-empty on/off combinations of the 3 channels
(`COLOURS`, uniform, repeats allowed) and one brightness level `U{10..15}`. The level is
written on its "on" channels; "off" channels stay 0. Same-colour crossings are common on purpose:
the model must separate those by geometry and continuity, as in real MCFO.

Layers, in the order they are applied to the 3 × Z × Y × X `uint8` raw:

| # | Layer | Levels | Channels | Combine | Train samples | Val/test |
|---|---|---|---|---|---|---|
| 1 | Haze | 0..hmax, hmax ∈ {1,2,3} | all 3 (grey) | max | baked | baked |
| 2 | Distractor blobs (1–4) | 1..lmax, lmax ∈ {2,3,4} | one random colour each | max | baked | baked |
| 3 | Neurons | one level in 10..15 | the neuron's colour | max | baked | baked |
| 4 | Bright confetti | 8..15 per spot | random colour per spot | max | GPU, per step | baked |
| 5 | Dark confetti | 0..1 per spot | all 3 | **min** | GPU, per step | baked |

So neurons are always brighter than haze and blobs (≥ 10 vs ≤ 4); only bright confetti reaches
the same levels. Dark confetti comes last and also punches holes in neurons. `raw.max() ≤ 15`
always. With `--no-noise`, raw is exactly the per-channel max of level × mask.

### 3.6 Noise layers (`noise.py`)

All Perlin fields are computed on a coarse grid (1/4 resolution; blob texture at 1/2) and
trilinearly upsampled. `fbm` sums octaves with halving amplitude and period and normalises to
[0, 1] by the 1st/99th percentiles.

- **Haze.** 3-octave fbm with largest period `U(100, 250)` voxels. Below a threshold
  `t0 ~ U(0.3, 0.7)` it is 0; above, it is quantised to `1..hmax`, `hmax ~ U{1..3}`. The same
  value goes into all channels.
- **Distractor blobs.** `U{1..4}` per volume. Each is an ellipsoid at a uniform random centre with
  radius `R ~ U(30, 100)` and per-axis stretch `U(0.6, 1.4)`. Its surface is wobbled ±35 % by a
  Perlin field of period `R/2`. It is filled with a fine 2-octave texture (period `U(6, 12)`)
  mapped to levels `1..lmax`, `lmax ~ U{2..4}`, in one random colour. Blobs may be cut by the
  volume edge.
- **Confetti.** One density field is shared by bright and dark confetti: a 2-octave fbm with period
  `U(60, 200)`, raised to a power `γ ~ U(1, 3)` (higher γ = patchier). For each kind, the
  density is log-uniform in `[1e-5, 2e-3]` spots per voxel, drawn separately per volume. The spot
  count is `Poisson(density · 144 M voxels)`, i.e. about 1.4 k to 290 k spots per kind. Centres are
  drawn in proportion to the field, jittered inside their coarse cell. A spot covers the voxels
  within `U(0.5, 1.5)` of its centre (up to ~3 voxels across).
  Bright spots take a level `U{8..15}` and a random colour; dark spots take `U{0, 1}` on all channels.

None of the noise enters the ground truth.

### 3.7 Ground truth

Ground truth follows FISBe `completely`. `gt_instances` has one channel per neuron, and
channel `n` holds `n + 1` inside neuron `n` and 0 elsewhere. Neurons may overlap (a voxel can be
non-zero in several channels). `gt_numinst` counts instances per voxel, and `gt_fg_rm_5` is
`gt_numinst > 0` opened with `ball(1)`, computed by the same `fg_rm_mask` function as
`prepare_fisbe_for_ppp.py`.

---

## 4. Output format

### 4.1 One sample: `<split>/tube_XXXXX.zarr` (zarr v2)

| Array (`volumes/…`) | dtype, shape | Notes |
|---|---|---|
| `raw` | `uint8 (3, 400, 600, 600)` | values 0–15, attr `clipmax = 15` |
| `gt_instances` | `uint8 (N, 400, 600, 600)` | channel n holds n + 1 |
| `gt_numinst` | `uint8 (400, 600, 600)` | instances per voxel |
| `gt_fg_rm_5` | `float32 (400, 600, 600)` | 0/1 foreground opened with `ball(1)` |
| `raw_normalized` | relative link → `raw` | so the default config keys work |
| `gt_instances_rm_5` | relative link → `gt_instances` | likewise |

Chunks are 64³ (raw `3×64³`, GT `1×64³`), compressed with Blosc zstd level 3 and bitshuffle.
Empty chunks are not stored. Training and prediction see `raw_normalized` (= `raw`) as
integer data and divide it by `clipmax` (`gp.Normalize(1/15)` in `train.py`, the same scaling in
`predict_no_gp.py`), so no float copy is stored. `raw_normalized` therefore tops out at the
brightest level / 15, which is not always 1.0.

Root `.zattrs`: `voxel_size_um = [0.44, 0.44, 0.44]`, `axes = "CZYX"`,
`channels = ["ch0", "ch1", "ch2"]`, plus a `simple_tubes` block: dataset, index, base seed,
seed, split counts, shape, all geometry / render / noise parameters (including
`render.confetti`, i.e. whether confetti is baked), a summary of the noise actually drawn
(haze, blobs, confetti densities and counts), and per instance its label, colour, level and voxel
count.

### 4.2 Sidecars (inside the parts)

| Path | Content |
|---|---|
| `meta/tube_XXXXX.json` | index, seed, split, stats (`n_target`, `n_instances`, `redraws`, `skipped`), all params, noise summary, and per instance the full branch tree |
| `meta/tube_XXXXX_skeletons.npz` | `instance_001`, … : centreline voxels inside the volume, `int16 (k, 3)` zyx |
| `meta/tube_XXXXX_mip.png` | val/test only: colour MIP over Z of raw (×17) next to the GT instances |
| `fg_points/tube_XXXXX.csv` | `gt_fg_rm_5` voxels on a stride-4 grid (`prepare_fg_points_csv.write_sample_csv`) |

### 4.3 On disk

```
fisbe/sqf/simple_v1_src/part_{0..19}.sqf     image root simple_v1/ → mounts at /simple_v1
  /simple_v1/{train,val,test}/tube_XXXXX.zarr   (overlays merge the 20 parts)
  /simple_v1/meta/   /simple_v1/fg_points/

fisbe/simple_v1/                              on scratch
  train -> /simple_v1/train                   no host inodes for 50k samples
  meta  -> /simple_v1/meta
  val/, test/                                 loose zarr skeletons: .zgroup/.zattrs and
                                              volumes/<array> -> /simple_v1/<split>/... links
  sample_list_per_split.txt                   all 50,200 names
  source_sqf.txt                              the 20 parts, read by ppp_job.sh
fisbe/fg_points/simple_v1 -> /simple_v1/fg_points
```

The loose val/test skeletons exist so host-side tools such as `ppp_submit_test.sh` can list
them. Section `(9)` of `evaluate-instance-segmentation/assets/sample_list_per_split.txt` lists
only the first 1,000 train samples plus val/test (`layout.sample_list_text(max_train=1000)`),
which keeps the web viewer's sidebar usable.

---

## 5. Confetti at training time (GPU)

Train samples are stored without confetti. `train.py` wraps `gp.TorchTrainExt` in
`_TorchTrainWithGpuConfetti`, which calls `setup01/gpu_confetti.py:GpuConfetti` on the raw
batch after it is on the GPU, before the forward pass. It is configured by
`[training.augmentation.gpu_confetti]` (set in `dsexp_simple_v1.toml`):

```toml
[training.augmentation.gpu_confetti]
datasets = ["simple_v1"]     # allowlist; training refuses to start if any train dataset is outside it
# optional overrides, defaults = generator values:
# levels = 15, coarse = 4, density_period = [60, 200], density_gamma = [1, 3],
# bright_density = [1e-5, 2e-3], dark_density = [1e-5, 2e-3], spot_radius = [0.5, 1.5],
# bright_level = [8, 15], dark_level = [0, 1], seed = 0
```

- It is a torch port of `noise.confetti` with the same parameters. Values are written as
  `level / 15` (mapped to [-1, 1] if `shift_intensity` is on), in place, in the batch dtype.
- Bright is a max and dark is the last layer in the generator too, so confetti on a stored
  train sample has the same distribution as baking it in.
- Two differences from the baked version: densities, γ and period are drawn **per crop** rather
  than per volume, and the random stream is seeded per step (`seed · 1,000,003 + step`), so each
  step sees new confetti and a run is reproducible.
- It is applied in the training step, not as a gunpowder node, so validation crops (and the
  `completely` val crop checksum) are unchanged.
- Cost: 4.7 ms per batch of 18 on an L40S (8.5 ms at maximum density), batched over crops.
  Parity with numpy is within 2 % (voxel fractions and level histogram).

Doing confetti on the GPU keeps 50,000 train samples small and gives fresh noise every step.
Haze and blobs stay baked because they are slow on the CPU (~120 ms and ~545 ms per crop).

---

## 6. Verification (`verify.py`)

`verify_zarr` runs on every sample twice: right after it is written (`--verify`) and again
through the mounted part. Checks:

- `raw` is `uint8 (3, Z, Y, X)`, chunks `(3, 64, 64, 64)`, `max ≤ 15`, `clipmax = 15`, and its
  shape matches the attrs.
- `gt_instances` is `uint8 (N ≥ 1, Z, Y, X)`, chunks `(1, 64, 64, 64)`, and N matches the attrs.
- Each channel n contains only 0 and n + 1, has ≥ 500 voxels, the voxel count from the attrs,
  and exactly one 26-connected component.
- Raw vs GT: on every voxel of neuron n, each "on" channel is ≥ its level, or ≤ 1 where dark
  confetti is baked (val/test only). Train samples allow no such exception. With noise off, raw
  must equal max(level × mask) exactly.
- `gt_numinst` equals the per-voxel count, and `gt_fg_rm_5` equals `fg_rm_mask(gt_numinst, 1)`.
- `raw_normalized` and `gt_instances_rm_5` are links to `raw` and `gt_instances`.
- With `--meta-dir`: every skeleton voxel lies inside its instance mask.
- With `--fg-dir`: the CSV exists, is non-empty and every point is inside `gt_fg_rm_5`.
- With `--regen`: re-rendering from the stored seed and params reproduces raw and every GT
  channel exactly.

The rev 3 build passed all 50,200 samples both times.

---

## 7. Parameter reference

Defaults are the values used for `simple_v1`. Only the ones with a CLI flag can be changed
without editing the dataclasses. Any change that alters the data should get a **new dataset
name** (e.g. `simple_v2`), not a rebuild of `simple_v1`.

**`GeometryParams`** (`geometry.py`)

| Field | Default | CLI |
|---|---|---|
| `shape` | `(400, 600, 600)` | `--shape Z,Y,X` |
| `step_len` | `(12, 20)` | |
| `turn_sigma_deg` / `turn_max_deg` | `20` / `45` | |
| `root_points` | `(10, 30)` | |
| `face_start_prob` | `0.5` | |
| `spline` | `catmull-rom` | `--spline {catmull-rom,bspline}` |
| `bspline_smoothing` | `4.0` per control point | |
| `dense_step` / `resample_step` | `0.05` / `0.25` | |
| `child_lambda` | `(2.5, 1.5, 0.8, 0.3)` (length = max order) | |
| `branch_t` / `branch_min_sep` / `branch_tries` | `(0.15, 0.85)` / `10` / `20` | |
| `branch_angle_deg` | `(30, 75)` | |
| `child_points_frac` / `child_radius_frac` | `(0.3, 0.7)` / `(0.7, 0.85)` | |
| `root_radius` | `(1.5, 3.0)` | `--root-radius lo,hi` |
| `taper` | `(0.1, 0.3)` | |
| `radius_floor` | `1.5` | `--radius-floor` |
| `deep_margin` | `1.0` | |

**`RenderParams`** (`render.py`)

| Field | Default | CLI |
|---|---|---|
| `neurons` | `(10, 30)` | `--neurons lo,hi` |
| `level` | `(10, 15)` | |
| `min_voxels` / `max_tries` | `500` / `20` | |
| `noise` | `True` | `--no-noise` |
| `confetti` | per split | `--confetti-splits` (build: `val,test`) |

**`NoiseParams`** (`noise.py`)

| Field | Default |
|---|---|
| `coarse` | `4` |
| `haze_period` / `haze_octaves` | `(100, 250)` / `3` |
| `haze_max` / `haze_threshold` | `(1, 3)` / `(0.3, 0.7)` |
| `blobs` / `blob_radius` / `blob_axis_scale` | `(1, 4)` / `(30, 100)` / `(0.6, 1.4)` |
| `blob_wobble` / `blob_level_max` / `blob_texture_period` | `0.35` / `(2, 4)` / `(6, 12)` |
| `density_period` / `density_gamma` | `(60, 200)` / `(1, 3)` |
| `bright_density` / `dark_density` | `(1e-5, 2e-3)` each, log-uniform |
| `spot_radius` | `(0.5, 1.5)` |
| `bright_level` / `dark_level` | `(8, 15)` / `(0, 1)` |

Other `render` flags: `--seed` (base seed), `--splits`, `--start`, `--count`, `--workers`,
`--verify`, `--mip-splits`, `--meta-out`, `--fg-out`, `--dataset` (name stored in attrs).
`skeleton` flags: `--dataset-dir`, `--splits`, `--dataset`, `--mount` (default `/<dataset>`),
`--fg-dir`, `--sources` (parts for `source_sqf.txt`), `--overwrite`.

---

## 8. Build record (rev 3, 2026-11-17/18)

| | |
|---|---|
| Jobs | submitter 23545912, array 23546010 (20 tasks, `torch_pr_61_tandon_priority`) |
| Samples | 50,200 rendered, 0 failed verify (at generation and through the mount) |
| Neurons | 1,004,362 total, 10–30 per volume, mean 20.0; 26,634 redraws, 0 skipped |
| Time | ~31 s per sample per worker (render + write + verify); 70–92 min render per part; 2:45–3:31 h per task |
| Size | 123 GB in 20 parts (6.1–6.2 GB each, `part_19` 6.9 GB with val/test) |
| Container view | 50,000 / 100 / 100 zarrs, 50,200 fg CSVs, 100,600 meta files |

---

## 9. Gotchas

- **Never mount under `/synth_MCFO`.** `ppp_job.sh` maps that prefix to another user's images;
  the build script refuses such a `DATASET`.
- **Keep `PARTS ≤ 20`.** Each part is one `--overlay`, and around 50 overlays break apptainer
  startup (training also mounts other datasets).
- **`sbatch --export=VAR=a,b` splits on commas.** The script exports the variables and passes
  `--export=ALL` instead; keep it that way if you edit it.
- **CLI defaults differ from the build.** Use `--splits 50000,100,100 --confetti-splits val,test
  --mip-splits val,test` to reproduce build samples.
- **Host paths dangle.** Train zarrs, `meta/` and fg points resolve only inside a container that
  mounts the parts.
- **Don't add a gunpowder node for new noise.** Any extra node shifts the fixed validation crops.
  Put cheap noise in the GPU hook (§5) or bake it in. Benchmark it first.
- **Real data must not get confetti.** `gpu_confetti` checks `train_datasets` against its
  `datasets` allowlist; keep the allowlist when mixing `simple_v1` with other datasets.
- **The web viewer** shows `simple_v1` with a `[0, max]` contrast stretch (a percentile clip made
  level-10 neurons look black). `sbatch/web/web.sh` mounts the parts.
