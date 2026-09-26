# Synthetic FISBe-style MCFO brain stacks from the male CNS connectome

A reproducible method to generate and validate ≥5,000 synthetic multicolour
FlpOut (MCFO) light-microscopy stacks of the *Drosophila* central brain, for
training and testing neuron instance segmentation alongside the FISBe
benchmark. Neuron counts and overlap vary by design, and the chosen neuron ids and
neuron-count statistics are logged for every image.

Nothing here needs to be run in a particular place. `pool` and `manifest` are cheap.
`render` is the expensive step: it is resumable and shardable across machines.

---

## 1. Why these design choices (research summary)

| Decision | Choice | Reason |
|---|---|---|
| EM source | male CNS v1.0, brain neurons | Public bucket, no login. Every neuron has an SWC skeleton **already registered to the JRC2018 unisex brain template** (`skeletons-unisex-template`, µm), the same template family FlyLight imagery is registered to. That makes the skeletons LM-scale without a separate transform. |
| Output format | FISBe zarr layout | FISBe stores `<sample>.zarr/volumes/raw` (uint16, 3×Z×Y×X, 0.44 µm isotropic, blosc-zstd) and `volumes/gt_instances` (uint8, one channel per neuron) with a `dim_neurons` attribute. Our output uses the same layout, dtype and compressor, plus an extra `gt_labels` map, so FISBe loaders and the official evaluation code run unchanged. |
| Image size | FISBe-like crops, 308×293×173 µm ≈ 700×666×393 voxels | FISBe images are native 40× central-brain tiles averaging ~400×700×700 voxels (e.g. 385×685×686). |
| Neuron counts | 1–60 per image, six bins | FISBe images range from 1–2 to >50 expressed neurons. The completely labelled FISBe set averages 4.6 labelled neurons per image, the partly labelled set 6.3. |
| Overlap | three modes (isolated / natural / crowded), plus forced same-colour overlaps | FISBe's partly labelled images left "overlapping neurons with the same or similar colour" unlabelled. That is exactly where synthetic data can supply ground truth. |
| MCFO model | 3 tags (HA, V5, FLAG), independent excision probability q per tag, conditioned on ≥1 tag | This follows MCFO-1 (Nern et al. 2015). q is calibrated from FISBe instance colours (below). |
| Imaging model | membrane (myr) label ∝ membrane area, boutons, depth-dependent Gaussian PSF, attenuation, tissue haze with texture, puncta, crosstalk, Poisson + PMT-gamma + read noise, 12-bit | Calibrated earlier against a raw Gen1 40× tile. `validate calibrate` re-fits the detector ranges to FISBe itself. |
| Splits | by **cell type** (80/10/10) | A test image never contains a cell type seen in training. |
| Validation | integrity → calibration → realism statistics → classifier two-sample test → downstream utility on FISBe test | These are standard checks of whether synthetic data transfers to real data (Lopez-Paz & Oquab 2017; Tobin et al. 2017). The decisive check is the utility test. |

---

## 2. Pipeline

```
pip install -r requirements.txt
export PYTHONPATH=/path/to/mcfo_sim

# (1) neuron pool: downloads ~50k JRC2018U skeletons (~5 GB), footprints, tissue grid
python -m mcfo_sim.brain.cli pool --cache cache --out pool --workers 32

# (2) optional but recommended: calibrate imaging to FISBe *train* images
#     (unzip fisbe_v1.0_completely.zip / partly.zip from Zenodo 10.5281/zenodo.10875063)
python -m mcfo_sim.brain.cli validate calibrate --fisbe fisbe/completely/train --out validation

# (3) design 5000 images: neuron ids, counts, overlap, tags, imaging, seeds (no pixels)
python -m mcfo_sim.brain.cli manifest --pool pool --out manifest --n-images 5000 \
       --calibration validation/calibration.json

# (4) statistics over the design
python -m mcfo_sim.brain.cli stats --manifest manifest

# (5) render (resumable; shard k/n across n jobs or machines)
python -m mcfo_sim.brain.cli render --manifest manifest --pool pool --cache cache --out rendered --shard 0/32
...
python -m mcfo_sim.brain.cli render --manifest manifest --pool pool --cache cache --out rendered --shard 31/32

# (6) validation (section 4)
python -m mcfo_sim.brain.cli validate integrity --rendered rendered --manifest manifest
python -m mcfo_sim.brain.cli stats --manifest manifest --rendered rendered
python -m mcfo_sim.brain.cli validate compare --fisbe fisbe/completely/train --rendered rendered --out validation --limit 200
python -m mcfo_sim.brain.cli validate c2st    --fisbe fisbe/completely/train --rendered rendered --out validation --limit 100
```

All choices live in `mcfo_sim/brain/config.py` (`BrainConfig`). Override any of them
with a JSON file passed as `--config` to `pool` or `manifest`. The config is copied into
`pool/config.json` and `manifest/config.json`.

### Reproducibility guarantees
- **Fixed inputs.** The male CNS release is versioned (v1.0), and the bucket paths are in the config.
- **Deterministic manifest.** It depends only on (config, pool, seed). Rebuilding it gives identical tables (tested).
- **Deterministic render.** Every image has its own `render_seed`, and all imaging parameters are stored per row. Rendering a row twice gives bit-identical `raw` and `gt_instances` (tested).
- **Pinned libraries.** `requirements.txt` pins the versions the code was tested with.

### Compute and storage (measured at 0.88 µm, scaled ×8 in voxels)
- **`pool`:** ~30 ms per neuron with 24 download threads, so ~25–30 min for the full pool (~50,000 neurons, 8,300 cell types).
- **`manifest`:** ~35 ms per image on the 3k-neuron test pool. Expect roughly 5–15 min for 5,000 images on the full pool.
- **`render`, at 0.88 µm:** a 33-neuron crowded image took 38 s, peaked at 1.7 GB RAM and used 37 MB on disk.
- **`render`, at 0.44 µm (estimated):** ~3–5 min per image, ~10–12 GB RAM, ~230–300 MB on disk (FISBe's own images are ~231 MB compressed).
- **All 5,000 images:** ≈ 250–400 CPU-hours and ≈ 1.2–1.5 TB. Run as many shards as RAM allows (e.g. 10 per 128 GB node).
- **Cheaper variant:** set `crop_um` to a training-patch size, e.g. `[90, 90, 56]`, for ~2 % of the cost.

---

## 3. What is logged

| File | Content |
|---|---|
| `manifest/manifest_images.csv` / `.parquet` | One row per image with the design and its outcome.<br>Design: `image_id`, `split`, `overlap_mode`, `count_bin`, `n_target`, `render_seed`, crop, placement and imaging parameters.<br>Outcome: `n_neurons` (achieved), `n_candidates`, `n_types`, **`body_ids`**, `overlapping_pairs`, `same_colour_overlapping_pairs`, `frac_neurons_overlapping`, `max_overlap_degree`, `expression_set_size` / `expressed_types` (natural mode), `forced_same_colour_pair`. |
| `manifest/manifest_neurons.csv` / `.parquet` | One row per neuron instance per image: `image_id`, `instance`, **`bodyId`**, `type`, `superclass`, `itoleeHl`, `trumanHl`, `somaSide`, `split`, `cable_in_crop_um`, MCFO `tags` and per-tag levels, `overlap_degree`. |
| `manifest/stats.json`, `stats.png` | Dataset-level statistics:<br>- neurons per image (mean, sd, quantiles, full histogram), overall and per overlap mode;<br>- images per bin, mode and split; shortfalls;<br>- images with no overlap or with same-colour overlap;<br>- overlap distributions;<br>- how often each bodyId is reused; unique ids and types;<br>- tag-combination and superclass frequencies; top brain hemilineages;<br>- a split-leakage check. |
| `rendered/<id>.json` | Rendered instances in `gt_instances` channel order, each with `bodyId`, type, hemilineage, tags, levels, visible/total voxels, median SNR, `frac_voxels_shared_3d` and `frac_pixels_shared_mip`.<br>Also: neurons planned but not rendered (and why), the realised image-level 3D and projection overlap, all imaging parameters, and the per-channel detector gain. |
| `manifest/rendered_counts.csv` + `stats.json["rendered"]` | Planned vs visible neurons per image and the realised overlap statistics. Written by `stats --rendered`. |

"Overlap" appears at three levels. The first is the design proxy: sharing a 4 µm footprint
cell, i.e. neurites within about 4–7 µm. The second is realised 3D contact of the
ground-truth masks. The third is overlap in the Z-projection (MIP), which is what a
2D viewer or a projection-based method sees.

---

## 4. Validation protocol

**4.1 Integrity (every image; must be 100 %).** `validate integrity` checks four things for each image:

- one `gt_instances` channel per rendered neuron;
- label ids no larger than the instance count;
- the rendered and not-rendered body ids together equal the manifest's ids for that image;
- each neuron's colour matches its tags on voxels it does not share with another neuron (the mean on its "on" channels exceeds the mean on its "off" channels).

**4.2 Calibration (on FISBe train only, never test).** `validate calibrate` measures, per
FISBe image:

- dark level and dark noise;
- effective bit depth;
- the neurons' 99th-percentile brightness relative to full scale;
- background-to-foreground ratio;
- instance colour multiplicity.

It writes the 5–95 % ranges to `calibration.json` (offset, read noise, gain target,
max ADU, and per-tag excision probability q). The q range is found by inverting
E[#tags | ≥1] = 3q / (1 − (1 − q)³). The manifest samples its imaging parameters from these ranges.

**4.3 Realism statistics.** `validate compare` computes the same features on FISBe and on
synthetic images and reports medians, the two-sample KS statistic and a scale-normalised
Wasserstein distance. Any KS above 0.3 is flagged. It also suggests a background scale
(`suggestions.json`); put it into `calibration.json` as `bg_scale: [lo, hi]` and rebuild the manifest.

- **Image level:** `fg_p99_rel`, `contrast`, `bg_to_fg`, `texture_rel`, `dark_sd`; also `n_instances` and `fg_fraction`, which differ from FISBe on purpose (report them, don't tune to them).
- **Instance level:** neurite diameter at the skeleton, skeleton length, voxels, number of colour channels on, dominant-channel fraction.
- **Target:** KS < 0.3 on the image-level intensity metrics and on diameter and colour.

**4.4 Classifier two-sample test.** `validate c2st` trains a gradient-boosted classifier on
hand-made patch features: per-channel intensity and gradient quantiles, Laplacian
energy and inter-channel correlation. Folds are grouped by image. The AUC measures how
separable synthetic and real patches are: 0.5 means indistinguishable by these features.
Track it while tuning; a reasonable target is AUC < 0.75. For a stronger test, repeat with a small 3D CNN.

**4.5 Downstream utility (the decisive test).** Train PatchPerPix (Hirsch et al. 2020;
the best FISBe baseline) or your own model with three seeds per regime:

| Regime | Training data |
|---|---|
| A | FISBe train only (baseline) |
| B | synthetic only |
| C | synthetic pre-train, then FISBe fine-tune |
| D | mixed batches (e.g. 50 % synthetic) |
| Scaling | C with 500 / 1,000 / 2,500 / 5,000 synthetic images |
| Ablations | no calibration; isolated-only; no same-colour overlaps; no boutons or texture |

- **Evaluation:** score the FISBe test split with the official code (`github.com/Kainmueller-Lab/evaluate-instance-segmentation`). Report S = ½(avF1 + C), avF1, C, clDice_TP, false splits and false merges, overall and on the dim-neuron subset (`dim_neurons_val_and_test_sets.json`).
- **Success criterion:** C or D beats A with non-overlapping 95 % bootstrap CIs over test images.
- **Synthetic test split:** it holds only unseen cell types, so it gives difficulty curves (score vs. neuron count and vs. realised 3D overlap) that FISBe is too small to provide.

---

## 5. Known limitations
- **One male brain as the morphology source.** FISBe includes female flies. FlyWire/BANC neurons could be added through the flybrains FLYWIRE → JRC2018 transforms; this is not implemented.
- **Native tiles vs template space.** Crops are cut from template space. FISBe images are native tiles (stitched and distortion-corrected, not registered), so orientation and scale jitter are small by default (±3 %, ±3°, 2 µm elastic).
- **Tissue haze comes from pool occupancy**, not from a brain mesh: the JRC2018U mesh shipped with flybrains is not watertight.
- **Visibility is gated by SNR ≥ 2.** Neurons below that are logged in `not_rendered`, so a rendered image can hold fewer neurons than planned; statistics report both counts.
- **Crowded images are mainly overlap-driven.** Picking neurons that share footprint cells tends to pick co-fasciculating or same-neuropil neurons; "natural" images instead mimic GAL4 co-expression by cell type.
- **`dim_neurons` is defined differently from FISBe.** Here it lists instances with median SNR < 5; FISBe's list was curated by hand.
- **Chunking differs.** Output chunks are (1, 64, 64, 64) rather than FISBe's (3, 64, 64, 64). Readers are unaffected.

## References
- Mais, Hirsch et al. 2024, FISBe (CVPR); dataset at doi:10.5281/zenodo.10875063.
- Male CNS connectome v1.0 (Janelia FlyEM et al.), male-cns.janelia.org; bioRxiv 10.1101/2025.10.09.680999.
- Nern, Pfeiffer & Rubin 2015, PNAS: MCFO. Meissner et al. 2023, eLife: Gen1 MCFO imaging.
- Bogovic et al. 2020, PLoS One: JRC2018 templates.
- Hirsch, Mais & Kainmueller 2020: PatchPerPix. Lopez-Paz & Oquab 2017 (ICLR): classifier two-sample tests. Tobin et al. 2017 (IROS): domain randomisation.
