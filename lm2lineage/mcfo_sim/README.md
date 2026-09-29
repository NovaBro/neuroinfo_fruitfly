# mcfo_sim — synthetic MCFO VNC stacks from the male CNS connectome

Generates FlyLight-like MultiColor FlpOut (MCFO) confocal stacks of the ventral
nerve cord from EM-reconstructed neurons, with perfect instance ground truth
and a hemilineage label for every neuron. Output format matches the FISBe
benchmark (zarr, `raw` + `gt_instances`, CZYX), so PatchPerPix and the FISBe
evaluation code can consume it directly.

## Quick start

```bash
pip install numpy scipy scikit-image pandas pyarrow zarr pillow
python -m mcfo_sim.simulate --out synthetic --cache data --n 10 --seed 0 \
       --neuromeres T2 --types 3,20 --visible 4,25
python -m mcfo_sim.verify synthetic/synth_00000.zarr
```

First run downloads the male CNS annotation table (14 MB) and each neuron's
SWC skeleton on demand from the public bucket `gs://flyem-male-cns/v1.0`
(no token needed). One 0.44 µm T2 sample (~400×420×420 voxels) takes 1–3 min
and ~5 GB RAM on 2 CPUs.

## Outputs per sample

| file | content |
|---|---|
| `synth_XXXXX.zarr/raw` | uint16 (3, Z, Y, X), channels HA / V5 / FLAG, 12-bit |
| `synth_XXXXX.zarr/gt_instances` | uint8 (N, Z, Y, X), one binary mask per neuron (overlaps allowed) |
| `synth_XXXXX.zarr/gt_labels` | uint16 (Z, Y, X), single label map (brightest wins overlaps) |
| `synth_XXXXX.json` | every sampled parameter; per instance: bodyId, type, superclass, **hemilineage** (trumanHl / itoleeHl), side, neuromere, tags, levels |
| `synth_XXXXX_skeletons.npz` | centreline voxel coordinates per instance (for clDice / skeleton metrics) |
| `synth_XXXXX_mip.png` | colour MIP (left) and GT MIP (right) |

Axes: X = left–right, Y = anterior–posterior, Z = dorso-ventral (imaging depth).

## What is simulated

1. **Virtual GAL4 line** — 3–20 random VNC cell types (intrinsic, ascending,
   motor, plus descending with p = 0.25); every neuron of those types expresses.
2. **MCFO labelling** — each of three reporters is excised independently per
   cell with probability q, chosen so the expected number of visible cells
   hits the target (capped at q = 0.45 so colours stay mixed). This gives seven
   colour combinations, repeated colours on different cells, per-type /
   per-cell / per-tag expression variation (lognormal).
3. **Morphology → fluorescence** — EM skeletons with radii, scaled 0.70 ± 5 %
   (the EM→DPX-LM shrinkage fitted from the release's template-space skeletons),
   small rotations and a smooth elastic warp. Membrane label is
   proportional to membrane area (myr-smFP): thin neurites are weighted 2πr per
   µm; thick processes and somata are hollow shells (rim-bright somata); random
   boutons on thin neurites; smooth brightness variation and dim gaps along the cable.
4. **Optics** — Gaussian PSF (σxy 0.22–0.40 µm, σz 0.6–1.3 µm) that widens with depth,
   exponential depth attenuation, smooth illumination field, channel crosstalk.
5. **Tissue background** — tissue outline = per-slab convex hull of the ~15,000
   VNC soma positions in the connectome; neuropil haze with fine texture, a
   cell-body rind term, sparse speckle, and soma-sized non-specific blobs.
6. **Detector** — Poisson shot noise, gamma PMT excess noise, ~1–3 ADU read
   noise, a 0–6 ADU dark offset, 12-bit clipping, and a per-channel auto-gain
   that puts the brightest labelled structures at 12–60 % of full range.
   The noise model was calibrated against a real Gen1 40× VNC tile
   (R10A07, signal channel: dark level ~3 ADU, dark s.d. ~1.3 ADU, foreground
   median ~46 ADU, foreground p99.9 ~1000 ADU).
7. **Ground truth** — a filled tube (radius ≥ ½ voxel) per neuron, gated to
   voxels where that neuron's own signal-to-noise ratio is ≥ 2, so masks cover
   what is visible, not invisible cable.

All imaging parameters are resampled per sample (domain randomisation);
`--no-randomize` uses the defaults in `ImagingParams`.

## Known limitations / next steps

- **Frame is EM space re-oriented and scaled, not JRC2018 VNC.** That's fine for
  segmentation training. For template-space work, chain the flybrains
  landmark transforms maleCNS → BANC → JRCVNC2018F. The release's
  `skeletons-unisex-template` VNC coordinates fall outside the JRCVNC2018U bounding
  box, so check before using them for the VNC.
- Sensory afferents (no VNC soma) are not yet included; they make up much of the
  dense tract labelling in real VNC MCFO.
- Boutons are placed at random; placing them at the neuron's presynaptic sites
  (the `syn-points` table or neuPrint) would be more faithful.
- No Brp/nc82 reference channel (it could be rendered from synapse density).
- Close the remaining domain gap with (a) this randomisation, (b) pre-training on
  synthetic data and then fine-tuning on FISBe plus a small set of hand-labelled VNC
  stacks, and/or (c) unpaired image translation (CycleGAN / CUT) or a
  segmentation-conditioned diffusion model trained on unlabelled FlyLight VNC
  stacks. Always evaluate on real annotated VNC data. FISBe is brain-only.
