# Simple synthetic tube dataset (`simple_v1`)

A simple synthetic dataset: smooth branching tubes in 4-bit intensities, no optics. It
is meant as the **first rung of a PPP training curriculum**: `simple_v1` → `1k_v1.1` →
`completely`. The model first learns "follow a tube and keep its instance identity",
then moves to realistic synthetic MCFO, then to real FISBe.

Revisions, same name and paths:
- **Rev 1** (2026-11-17): 256³, 3–12 neurons, clean background. Built, then deleted.
- **Rev 2** (2026-11-17): 400×600×600, 10–30 bushier neurons, dim Perlin haze, dim
  distractor blobs, bright and dark confetti (§3.8), 1,000 samples through PPP prep/pack
  (~130 MB/sample). Replaced by rev 3.
- **Rev 3** (2026-11-17, `simple_v1` today): same images, **50,000 / 100 / 100** samples in a compact
  self-contained layout (~3.4 MB/sample, §2/§5). Train samples are stored without
  confetti; training adds it on the GPU each step. Val/test keep it baked. Plan file:
  `~/.claude/plans/make-plan-for-implementation-cheerful-eagle.md`.

Status: implemented in `synthetic/simple_tubes/` and `sbatch/ppp/prep/simple/`; build
record in §8. Training on it is not planned yet.

Related: [`ppp_prep.md`](../ppp/ppp_prep.md) (sqf packing, fg points),
[`experiment_v2.md`](../experiments/experiment_v2.md) (Dataset Experiments, which this
rung would extend), `lm2lineage/mcfo_sim/README.md` (the realistic simulator; this is
its opposite end).

---

## 0. Decisions so far (2026-11-17)

| Question | Decision |
|---|---|
| Purpose | PPP curriculum rung. Output must drop into the existing PPP prep, configs and eval unchanged. |
| "4-bit" | Raw values are 16 levels, 0–15: background 0, neurons at varying brightness. Stored as `uint8`, since zarr has no 4-bit dtype. Not bit-packed. |
| Channels | 3 channels with MCFO-style random colour per neuron (7 on/off combinations, repeats allowed). Matches `num_channels = 3`. |
| Scale | Rev 2: 400 × 600 × 600 (Z, Y, X) per volume, 1,000 volumes (800 / 100 / 100). Rev 1 was 256³. |
| Code location | `synthetic/simple_tubes/` |
| Neurons / radius | Rev 2: U{10..30} neurons per volume, children λ = 2.5 / 1.5 / 0.8 / 0.3, max order 4, 10–30 control points per root. Root radius U(1.5, 3.0) voxels, floor 1.5. (Rev 1: 3–12 neurons, λ = 1.5 / 0.8 / 0.3, 6–20 points.) |
| Somata | Excluded from v1. Tubes only. |
| Leaving the volume | Each branch stops at its first exit from the volume, so every neuron is one piece (§3.7). |
| Noise (rev 2) | Dim distractor blobs (1–4 per volume, levels 1–4); Perlin haze (levels 0–3) plus a second Perlin field that sets confetti density; bright confetti (levels 8–15, random colour) and dark confetti (clamps to 0–1, punches holes in neurons). Nothing of it enters the GT. |
| Rev 2 vs rev 1 | Overwrite `simple_v1` in place (same paths, config, viewer tab). |
| Rev 3 (storage) | 50,000 train + 100 val + 100 test, rebuilt in place. No PPP prep/pack: the generator writes every array PatchPerPix reads; raw stays uint8 and training/prediction scale it by its `clipmax` attr. Also fix `prepare_fisbe_for_ppp.py` for all datasets. |
| Rev 3 (speed) | Keep training fast: blobs and haze stay baked (expensive on CPU); confetti moves to the GPU in the training step (~ms per batch). |
| Training plan | Out of scope here; planned separately later. |

---

## 1. Task checklist

- [ ] Super basic synthetic dataset. Use only 4-bit data format (foreground and background).
  - [x] Generate a sequence of 3D control points (either via jittered grids or constrained
        random walks) and interpolate a smooth Catmull-Rom or cubic B-spline through them. → §3.1, §3.2
  - [x] Evaluate the spline at high sub-voxel resolution (\(t \in [0, 1]\)) and compute the
        Euclidean distance from the curve centerline to surrounding grid coordinates,
        thresholding by the strand radius. → §3.5
  - [x] Spawn child splines directly from sampled points along a parent spline to create
        tree-like bifurcations without disconnected gaps. → §3.3
  - [x] Colour, 4-bit intensity and per-volume composition. → §3.6, §3.7
  - [x] Writer in FISBe layout, sidecars, `verify.py`. → §2, §6
  - [x] Generate 1k volumes, pack, PPP prep, fg points, sample list. → §5
  - [ ] Smoke train (40 its) on `simple_v1` with completely as extra val. → §6

---

## 2. Output format (rev 3)

Each zarr is self-contained for PatchPerPix and lives inside a SquashFS part
(`fisbe/sqf/simple_v1_src/part_<i>.sqf`, image root `simple_v1/`, mounted at `/simple_v1`):

```
/simple_v1/                                   (inside the mounted parts; overlays merge them)
  {train,val,test}/tube_XXXXX.zarr/volumes/
    raw                uint8 (3, 400, 600, 600), 0..15, attrs {"clipmax": 15}
    gt_instances       uint8 (N, 400, 600, 600), channel n holds n + 1, overlaps allowed
    gt_numinst         uint8 (400, 600, 600), instances per voxel
    gt_fg_rm_5         float32 0/1, foreground opened with ball(1) (prepare_fisbe_for_ppp.fg_rm_mask)
    raw_normalized   -> raw            (relative links, so the default config keys work)
    gt_instances_rm_5 -> gt_instances
  meta/tube_XXXXX.json, _skeletons.npz (all), _mip.png (val/test only)
  fg_points/tube_XXXXX.csv           (stride-4 grid of gt_fg_rm_5, prepare_fg_points_csv.write_sample_csv)

fisbe/simple_v1/                              (on scratch)
  train -> /simple_v1/train            (no per-sample inodes for 50k samples)
  meta  -> /simple_v1/meta
  val/, test/                          loose skeletons: volumes/<array> -> /simple_v1/<split>/...
  sample_list_per_split.txt, source_sqf.txt (the parts)
fisbe/fg_points/simple_v1 -> /simple_v1/fg_points
```

- Paths resolve only inside a container that mounts the parts; `ppp_job.sh` does that from
  `source_sqf.txt`. Host-side listing of train samples sees a dangling link (by design).
- Chunks 64³ (raw 3×64³, GT 1×64³), blosc zstd clevel 3 with bitshuffle, empty chunks not stored.
- GT values follow FISBe completely (channel n holds n + 1). Root `.zattrs` carry
  `voxel_size_um`, `axes`, `channels` and a `simple_tubes` block (index, seeds, split counts,
  shape, all parameters incl. whether confetti is baked, noise summary, per-instance label /
  colour / level / voxels).
- Names `tube_00000` … `tube_50199`; split fixed by index (0–49,999 train, 50,000–50,099 val,
  50,100–50,199 test). Per-sample seed = `base_seed * 1_000_000 + index`; noise uses its own
  stream (`default_rng([seed, 1])`), so any sample can be regenerated alone.

---

## 3. Generator

All lengths are in voxels. Neuron count and radius are confirmed (§0); other defaults are proposals.

### 3.1 Control points: constrained random walk

Persistent random walk, one per root branch:

1. Pick a start point `p0`. Half the neurons start on a random face of the volume and
   head inward, so they look like neurons cut by the crop. The other half start
   uniformly inside. No soma in v1.
2. Initial direction `d0`: inward normal for face starts, uniform on the sphere otherwise.
3. Each step: rotate `d` by an angle drawn from `|N(0, σ_turn)|` (capped at `θ_max`)
   about a random perpendicular axis, then `p_{k+1} = p_k + L · d`.
4. Stop after `K` points. Cutting the branch at the volume boundary happens later, on
   the resampled curve (§3.7).

| Parameter | Default |
|---|---|
| step length `L` | U(12, 20) |
| turn σ_turn / cap θ_max | 20° / 45° |
| points per root `K` | U(10, 30) (rev 1: U(6, 20)) |

The jittered-grid variant (control points on a coarse lattice plus jitter, connected
by a random path through the lattice) is a later option. The random walk gives more
neuron-like curves for less code.

### 3.2 Spline

- **Centripetal Catmull-Rom** (α = 0.5) through the control points. It passes through
  every point (so branch attachment points are exact), and the centripetal form avoids
  the cusps and self-loops that uniform Catmull-Rom makes on uneven spacing.
  A cubic B-spline (`scipy.interpolate.splprep`, k = 3) is smoother but does not pass
  through the points. Keep it as a flag.
- Phantom end points are reflections (`2·P0 − P1`, `2·Pn − Pn−1`). Duplicated end
  points, as first planned, give a zero knot interval in the centripetal form and
  divide by zero.
- Sample densely, then **resample by arc length at 0.25 voxel spacing**. The
  rasteriser then works on an evenly spaced polyline, and the error from using
  samples instead of the true curve stays below 1/8 voxel.

### 3.3 Branching

Children start **exactly at a sample point of the parent**, so the union of parent and
child tubes is connected by construction.

1. Number of children per branch: Poisson(λ_order), with λ = 2.5, 1.5, 0.8, 0.3 for
   orders 0–3. Max order 4. (Rev 1: λ = 1.5, 0.8, 0.3, max order 3.)
2. Branch point: arc-length fraction `t ∈ [0.15, 0.85]` of the parent, at least 10 voxels
   from other branch points, and on the part of the parent that is inside the volume
   (after truncation, §3.7).
3. Child direction: the parent tangent at `t`, rotated by U(30°, 75°) about a random
   perpendicular axis.
4. The child is a new random walk (§3.1) from that point, with `K` scaled by 0.3–0.7
   of the parent's length.
5. The child's radius starts at 0.7–0.85 × the parent radius at `t` (§3.4).

The tree is stored in the JSON sidecar as `{branch_id, parent_id, t_parent, order}`, with
the control points and radius profile.

### 3.4 Radius profile

- Root radius U(1.5, 3.0) (≈ 0.7–1.3 µm), tapering linearly by 10–30 % along each branch.
- **Floor at 1.5 voxels.** PPP prep builds `gt_fg_rm_5` with an opening of radius 1, which
  erodes thinner tubes, and `gt_fg_rm_5` drives crop sampling. Thinner tubes can come in
  a later rung.

### 3.5 Rasterisation (distance to the centreline)

Per neuron, over all of its branches:

1. Concatenate every branch's resampled points `P` (M × 3) with per-point radius `r_P`.
2. Candidate voxels: round `P` to the grid, then add every offset within
   `ceil(max r) + 0.9`. This band is a small fraction of the volume.
3. Build a `scipy.spatial.cKDTree` on `P` and query each candidate voxel centre for the
   nearest point distance `d` and index `i`.
4. Mask: `d ≤ r_P[i]`. This is the exact sub-voxel distance to the 0.25-spaced
   centreline, with a varying radius.

This replaces a full-grid `distance_transform_edt`, which would snap the centreline to
voxel centres and does not handle a per-point radius. Hard edges in v1. A soft edge
(coverage `clip(r − d + 0.5, 0, 1)`) is a knob for later rungs.

### 3.6 Colour and 4-bit intensity

- Colour: one of the 7 non-empty subsets of {ch0, ch1, ch2}, uniform. With 3–12 neurons
  per volume, colours repeat often. That is intended: same-colour crossings force the
  model to separate neurons by geometry and continuity, as in real MCFO.
- Brightness: one level per neuron, U{10, …, 15}, used on its "on" channels. "Off"
  channels are 0.
- Overlaps: per-channel **max** across neurons.
- Background: haze, blobs and confetti (§3.8); 0 with `--no-noise`.
- Invariant (checked in §6): on every GT voxel of neuron n, each "on" channel is
  ≥ its level, or ≤ 1 where dark confetti punched a hole. With `--no-noise`, raw equals
  the per-channel max of level × GT exactly. Always `raw.max() ≤ 15`.

### 3.7 Per-volume composition

- Neurons per volume: U{10, …, 30} (rev 1: U{3, …, 12}).
- **Truncate at the first exit.** Walk along each branch's resampled centreline and cut
  it at the first point outside the volume. Then extend it straight along the exit
  direction for `r + 1` voxels, so the tube runs cleanly off the edge instead of ending
  in a rounded cap just inside. A straight extension cannot curve back in. A branch
  that starts on a face (§3.1) counts its first exit only after it has been inside.
  Children only start from in-volume parent points (§3.3), so every neuron is one
  26-connected piece by construction.
- **Why truncate:** a branch that left and came back in would appear as two
  disconnected pieces with one ID. PPP links voxels only through local 7³ patches, so
  it can never join them, and `simple_v1` val/test would score a correct prediction as a
  false split. Real FISBe splits come from dim stretches in imaging, not from crop
  edges: 79 of 139 completely GT neurons have more than one piece, but the median
  largest piece holds 99.75 % of the voxels and only 17 have it below 90 %
  (`imaging_helpers_hpc/output/gt_instance_connectivity.csv`). Gaps of that kind belong
  in a later rung (the dim-gap knob in §3.8).
- **Reject and resample** a neuron (up to 20 tries) only if its in-volume mask has
  fewer than 500 voxels.
- No rejection of crossings between different neurons. Crossings are the point.
- GT: `gt_instances[n] = (n + 1) · mask_n` (uint8), overlaps allowed, same as FISBe.

### 3.8 Noise and distractors (rev 2, `noise.py`)

None of this enters the GT. Layers in order, per channel on the 4-bit raw:

1. **Haze**: fractal Perlin field (3 octaves, largest period U(100, 250) voxels),
   normalised to [0, 1], everything below a threshold U(0.3, 0.7) is 0 and the rest is
   quantised to 1..hmax with hmax ∈ {1, 2, 3}. Same value in all channels (grey).
2. **Distractor blobs**: U{1..4} per volume. Ellipsoid of radius U(30, 100) with per-axis
   stretch U(0.6, 1.4), its edge wobbled ±35 % by a mid-frequency Perlin field. Filled
   with a fine Perlin texture (period 6–12) at levels 1..U{2..4}, in one random colour.
3. **Neurons** (§3.6), per-channel max, levels 10–15.
4. **Bright confetti**: spots of radius U(0.5, 1.5) (1–3 voxels across), levels 8–15,
   random colour, per-channel max.
5. **Dark confetti**: same spot shape, clamps all channels to 0–1. Applied last, so it
   punches holes in neurons, blobs and haze.

Rev 3: layers 1–3 are baked into every sample; layers 4–5 are baked only into val/test.
Train samples get them on the GPU each step instead (`setup01/gpu_confetti.py`, §5).

Confetti centres follow a second Perlin field raised to a power U(1, 3), so density is
patchy inside a volume. Overall density is log-uniform per volume in
[1e-5, 2e-3] spots per voxel, drawn separately for bright and dark. Perlin noise is
computed at 1/4 resolution (1/2 for blob texture) and trilinearly upsampled.

Still possible later (CLI flag + new name, not a change to `simple_v1`): somata, soft
edges, blur, dim gaps along the cable, per-branch brightness drift, single-colour
neurons, thinner tubes.

---

## 4. Code layout

| File | Purpose |
|---|---|
| `synthetic/simple_tubes/layout.py` | names, fixed splits, sample list, `fisbe/<dataset>` tree (stdlib only) |
| `synthetic/simple_tubes/geometry.py` | random walk, Catmull-Rom / B-spline, arc-length resample, truncation, branching tree |
| `synthetic/simple_tubes/render.py` | per-branch cKDTree rasteriser, colour and 4-bit compose, redraw rule, connectivity guard; instances kept as flat voxel indices |
| `synthetic/simple_tubes/noise.py` | Perlin / fractal noise, haze, distractor blobs, confetti |
| `synthetic/simple_tubes/render_util.py` | colours, offsets, sparse 26-connectivity count, dense GT channel |
| `synthetic/simple_tubes/generate.py` | CLI `render` (all PPP arrays + links, meta, fg CSVs; `--confetti-splits`, `--mip-splits`) and `skeleton` |
| `synthetic/simple_tubes/verify.py` | invariants in §6; `--regen` re-renders from the seed; `--fg-dir` checks fg CSVs |
| `synthetic/simple_tubes/check_gpu_confetti.py` | numpy-vs-GPU confetti parity, GPU speed, picture of a GPU-noised crop |
| `sbatch/ppp/prep/simple/generate_simple_sbatch.sh` | submitter writes the tree, then an array renders each part on `$SLURM_TMPDIR`, packs it and re-verifies it through the mount |
| `PatchPerPix/.../setup01/gpu_confetti.py` | torch port of the confetti layers, applied to training batches on the GPU |
| `PatchPerPix/.../setup01/train.py` | integer raw → `gp.Normalize(1/clipmax)` per source (float sources unchanged); `_TorchTrainWithGpuConfetti` hook; `[training.augmentation.gpu_confetti]` |
| `PatchPerPix/.../setup01/predict_no_gp.py` | integer raw scaled by its `clipmax` attr |
| `PatchPerPix/experiments/run_ppp.py` | `simple_v1` in `KNOWN_DATASETS` |
| `PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py` | (all datasets) empty chunks really skipped, zstd + bitshuffle; `fg_rm_mask` / `write_array` helpers |
| `.../configs/exp/dsexp/dsexp_simple_v1.toml` | data overlay; enables `gpu_confetti` for `simple_v1` |

Run from the repo root as a module, e.g.
`python -m synthetic.simple_tubes.generate render --out /tmp/x --count 2 --verify`
(`--shape 100,160,160 --neurons 4,8` for a quick look, `--no-noise` for tubes only).
Dependencies are numpy, scipy, zarr, scikit-image and pillow, all in the `ppp` overlay.

---

## 5. Pipeline into PPP (rev 3)

1. **Generate** (`sbatch sbatch/ppp/prep/simple/generate_simple_sbatch.sh`; defaults
   `TOTAL=50200 SPLITS=50000,100,100 PARTS=20 CONFETTI_SPLITS=val,test MIP_SPLITS=val,test`).
   The submitter writes the `fisbe/simple_v1` tree, `source_sqf.txt` and the
   `fisbe/fg_points/simple_v1` link, then submits the array. Use
   `SBATCH_ACCOUNT=<account>` to pick the account for both. Dry run:
   `DATASET=simple_v1_dry TOTAL=60 SPLITS=40,10,10 PARTS=1`.
   Do **not** mount under `/synth_MCFO` (`ppp_job.sh` maps that to vh2308's images).
2. **Nothing else.** No `prepare_fisbe_for_ppp.py`, no `pack_ppp_sqf_sbatch.sh`, no fg-points
   job: the parts already hold every array and the fg CSVs. `ppp_job.sh` mounts the parts
   from `source_sqf.txt`; its `need_ppp` stays 0 because no link points at `/synth_MCFO`.
3. **Training** reads uint8 raw (`gp.Normalize` by `clipmax`) and, with
   `dsexp_simple_v1.toml`, adds confetti on the GPU each step. `gpu_confetti` refuses to start if
   any train dataset is outside its `datasets` allowlist.
4. **Sample lists**: `fisbe/simple_v1/sample_list_per_split.txt` lists all 50,200;
   section `(9)` of `evaluate-instance-segmentation/assets/sample_list_per_split.txt` lists
   the first 1,000 train plus val/test (`layout.sample_list_text(max_train=1000)`), so the
   viewer sidebar stays usable.

Measured (dry run, 60 samples): 206 MB per 60 samples, ~3.4 MB/sample (train without
confetti ~3 MB, val/test with ~5 MB) → ~170 GB for 50,200. Generation + in-worker verify
~41 s/sample/worker.

---

## 6. Verification

`verify.py`, run on every volume at generation time and again after packing:

- shapes, dtypes and chunks as in §2; `raw.max() ≤ 15`
- the raw-vs-GT invariant from §3.6 (noise-aware; exact equality with `--no-noise`)
- each `gt_instances[n]` is non-empty, ≥ 500 voxels, and one 26-connected component.
  The last holds by construction (§3.7), so a failure means a generator bug.
- skeleton points lie inside their instance mask
- determinism: regenerating a sample from its seed gives byte-identical arrays
- rev 3: raw `clipmax` = 15; `gt_numinst` = per-voxel count; `gt_fg_rm_5` =
  `fg_rm_mask(gt_numinst, 1)`; the two alias links; fg CSV points inside `gt_fg_rm_5`;
  no dark-confetti exception on train samples (no confetti baked)

Then:

1. Look at about 20 MIP PNGs across splits: branch shapes, crossings, colour repeats.
2. One-batch dry run through steps 1–5 of §5. Inside the container,
   `raw_normalized` and `gt_instances_rm_5` should open through the symlinks, and
   `raw_normalized` should equal `raw / 15`. (Its max is the brightest level / 15, so
   not always 1.0.)
3. 40-iteration smoke train like `dsexp-smoke-extraval`. Check that both
   `val_metrics.csv` and `val_metrics_completely.csv` are written.

---

## 7. Open questions

None. Resolved 2026-11-17 (see §0). Training on `simple_v1` (curriculum stages, lengths,
baselines) will be planned in a separate doc.

---

## 8. Build record

### Rev 1 (2026-11-17, 256³, clean; data deleted when rev 2 overwrote it)

**Dry run** (`simple_v1_dry`, 200 volumes, 2 parts; deleted afterwards):
- generation verify and verify through the mounted part: 200/200 OK
- `pack_ppp_sqf_sbatch.sh` with `PLAIN_SRC=1 CLIPMAX=15 RAW_DTYPE=float16` worked unchanged.
  In the container, `raw_normalized == raw / 15`, `gt_instances_rm_5 == gt_instances`,
  and `gt_numinst` matches.
- fg points: every volume has points (min 63 on the dry run)
- `verify --regen` in the `ppp` env reproduced 4 samples exactly
- `verify` flagged a planted raw voxel, a cut GT mask and skeleton voxels outside it

**`simple_v1` rev 1** (seed 0):
- Generation job 23538474 (array 23538512–16): 1,000 volumes, 0 verify failures,
  7,626 neurons (mean 7.6 per volume), 595 redraws for the 500-voxel rule, 0 skipped,
  3.7 s per volume
- Pack job 23538690 (`BATCH=100`): 10 shards in `fisbe/sqf/simple_v1/` (83 MB)
- Source parts: `fisbe/sqf/simple_v1_src/part_{0..3}.sqf` (27 MB)
- Sidecars: `fisbe/simple_v1/meta/` (3,000 files, 27 MB)
- fg points job 23539114: `fisbe/fg_points/simple_v1/` (1,000 CSVs, min 28 points)
- `ppp_job.sh` overlay discovery for `simple_v1,completely` mounts 14 images
- Section `(9) samples for simple_v1` added to
  `evaluate-instance-segmentation/assets/sample_list_per_split.txt`

### Rev 2 (2026-11-17, 400×600×600, noise)

- Full-size single-volume test (job 23541184): generate + verify, `verify --regen` and
  PPP prep all passed; numbers in §5.
- Rev 1 data deleted (`fisbe/simple_v1`, `fisbe/sqf/simple_v1_src`, `fisbe/sqf/simple_v1`,
  `fisbe/fg_points/simple_v1`), then regenerated: generation job 23541501.
- Generation job 23541501 (array 23541540): 1,000 volumes, 0 verify failures, sources 5.2 GB.
- PPP pack 23542835 cancelled after ~1 h and all rev 2 data deleted for rev 3 (user OK).

### Rev 3 (2026-11-17, compact layout, 50,000 / 100 / 100)

- Prep fix (all datasets): on a test sample, fixed `prepare_fisbe_for_ppp.py` gives
  `gt_instances_rm_5` with 59 chunk files (= `gt_instances`) instead of every chunk.
- Generator arrays match the fixed prep exactly (`gt_numinst`, `gt_fg_rm_5`,
  `gt_instances_rm_5`; `raw_normalized == raw / 15`).
- Completely-val crop checksum with the new `train.py`: 1.416670e+07 (= crop set A, unchanged).
- Dry run `simple_v1_dry` (job 23545253, 60 samples): 60/60 verified twice; 206 MB.
- GPU confetti check (job 23545775, L40S): numpy-vs-GPU parity within 2 % (voxel fractions,
  level histogram) at 1e-4 and 2e-3 spots/voxel. **Speed 86–88 ms per batch of 18** (target was
  < 5 ms): per-crop Python loop and small kernels dominate. TODO: vectorise across the batch
  (one density field tensor for all crops, batched spot indices, drop `torch.quantile`).
- Smoke train `simple-v1-dry-smoke` (job 23545776, tandon_priority, configs our-noenc-c1k_v1.1 +
  expv2_e1_common + dsexp_pre + dsexp_simple_v1_dry + dsexp_smoke): COMPLETED in 5.5 min.
  `gpu_confetti` active; `val_metrics.csv` and `val_metrics_completely.csv` written; completely
  val crop checksum 3.516400e+06 (same as the 1k_v1.1 smoke 22448122, so no crop drift);
  ~1.26 s per step with the per-crop confetti.
- Full build (submitter 23545912, array 23546010, tandon_priority): 20 parts × 2,510 samples,
  2:45–3:31 h per task. 50,200 / 50,200 verified at generation and again through the mounted
  parts; `fisbe/sqf/simple_v1_src` = 123 GB (~2.45 MB/sample). Inside the container the merged
  parts list 50,000 train / 100 val / 100 test, 50,200 fg CSVs, 100,600 meta files.
- GPU confetti rewritten batched over the crops (one density-field pass, one multinomial and one
  `scatter_reduce_` per kind, in place in fp16, 57 candidate offsets instead of 123). Same results
  as the per-crop version at batch size 1.
- Eval sample list section (9) already rewritten for 50,000 / 100 / 100 (first 1,000 train).
- After the build: delete `fisbe/simple_v1_dry`, `fisbe/sqf/simple_v1_dry_src`,
  `fisbe/fg_points/simple_v1_dry`, `configs/exp/dsexp/dsexp_simple_v1_dry.toml`,
  `metrics/ppp/simple-v1-dry-smoke`; relaunch the web viewer.


---

## 9. simple_v2 (2026-11-19)

User request: a greater range of brightness, thickness, length and branching. Built as a **new dataset
`simple_v2`** next to `simple_v1` (experiment_v4 keeps using simple_v1). Everything else as rev 3:
400×600×600, 50,000 / 100 / 100, 4-bit raw, MCFO colours, haze + blobs baked, confetti baked only in val/test
(GPU confetti in training), same self-contained SquashFS layout.

`python -m synthetic.simple_tubes.generate render --preset simple_v2 ...` (`generate.py PRESETS`). New
parameters default to the simple_v1 values and draw no extra random numbers when off, so simple_v1 still
regenerates byte for byte (checked: `verify --regen` on tube_00007 and tube_50003).

| | simple_v1 (rev 3) | simple_v2 |
|---|---|---|
| Neuron brightness | one level per neuron, U{10..15} | base U{3..15}; each branch + U{−2..2}; sinusoidal drift along the cable (amplitude 0–1.5 levels, period 50–200 voxels); clipped to 1–15; per voxel = max over the branches covering it |
| Root radius | U(1.5, 3.0), floor 1.5 | log-uniform 1.0–5.0, floor 1.0 |
| Taper along a branch | 10–30 % | 0–50 % |
| Root length | 10–30 control points, 12–20 voxels apart | 4–60 points, 8–24 voxels apart |
| Branching | λ = 2.5 / 1.5 / 0.8 / 0.3 (max depth 4) | per-neuron factor log-uniform 0.1–3× on λ = 2.5 / 1.5 / 0.8 / 0.4 / 0.2 (max depth 5) |
| Branch angle | 30–75° | 20–90° |

- `verify.py`: with varied brightness only the neuron's minimum level is guaranteed on its voxels
  (`level_min` / `level_max` per instance in the zarr attrs); exact equality is left to `--regen`.
- Radius-1 tubes are partly removed from `gt_fg_rm_5` by its radius-1 opening, so the thinnest neurons get
  fewer fg points (sampling only; the GT keeps them).
- Neurons can end up near level 1 everywhere, as dim as the haze: intended ("dim neurons reach the haze").

**Full-size test** (job 23791375, 32 volumes, 16 workers): 32/32 verified, 30–37 s per volume (as v1); neuron
voxels 0.05–0.33 % of the volume (median 0.17 %); branches per neuron median 4, p90 56, max 409; root radius
1.00–4.99; levels 1–15; ~10 MB per train sample before packing (block-rounded `du`).

**Build:** `DATASET=simple_v2 SBATCH_ACCOUNT=torch_pr_61_tandon_advanced sbatch
sbatch/ppp/prep/simple/generate_simple_sbatch.sh` (2026-11-19 21:41): submitter 23791579, array 23791709
(20 parts). Wired up: `configs/exp/dsexp/dsexp_simple_v2.toml` (GPU confetti allowlist simple_v2),
`run_ppp.py KNOWN_DATASETS`, section (10) of the eval sample list, the web viewer (tab + mounts; `web.sh` now
loops over simple_v1 / simple_v2, 40 SquashFS images).
