# swc2img_py — run SWC2IMG from Python

Renders SWC neuron skeletons with the [SWC2IMG](https://imagescience.org/meijering/software/swc2img/) Fiji
plugin (Radojević & Meijering), so the results can be compared with our `mcfo_sim` synthetic images
(synth_MCFO 1k_v1).

## Setup (once)
1. Get Fiji (the Windows build bundles Java): https://downloads.imagej.net/fiji/latest/fiji-latest-win64-jdk.zip
2. Put `SWC2IMG_-1.1.0.jar` into `Fiji/plugins/` and `ImageScience-3.1.0.jar` into `Fiji/jars/`. Both are at
   https://github.com/ImageScience/Maven/tree/public/org/imagescience
3. Set `FIJI_DIR` to the Fiji folder (default `D:/tools/Fiji`). Needs numpy, tifffile, Pillow, matplotlib and zarr.

## Use
Run from this folder (`lm2lineage/swc2img_py`):

```bash
# any SWC file(s), field of view = neuron extent + border
python -m swc2img_py swc path/to/neuron.swc --out renders/

# a 1k_v1 image: all its neurons, cropped to the v1 field of view at 0.44 µm voxels
python -m swc2img_py v1 brain_00049 brain_00091 --bundle D:/1k_v1_neuron_sets/1k_v1_neuron_sets --out renders/
```

MCFO colour version (whiteboard step 3): add `--rgb`. It runs one SWC2IMG render per tag channel (HA, V5, FLAG
→ R, G, B); a neuron with two tags lands in two channels and gets a mixed colour. `--tags v1` (default) uses the
tags in `neurons.csv`, the same colours as the v1 image; `--tags random --seed N` draws new MCFO tags. It writes
`<name>.swc2img_rgb.tif` (3-channel composite, ~2.2 GB), `<name>_swc2img_rgb_mip.png` and `<name>_rgb_compare.png`.
About 3 minutes per image.

```bash
python -m swc2img_py v1 brain_00049 --rgb --bundle D:/1k_v1_neuron_sets/1k_v1_neuron_sets --out renders/
```

Outputs per item: `<name>.swc2img.tif` (float32 ZYX), `<name>_swc2img_mip.png`, and for v1 images
`<name>.swc2img.json` (settings) plus `<name>_compare.png` (v1 vs SWC2IMG side by side) when the bundle has that
example.

SWC2IMG options: `--voxel` (µm, default 0.44), `--background` (10), `--correlation` (noise blur σ in µm, 0.5),
`--snr` (10). `--radius-scale 0.75` matches the radius scaling v1 used.

### FISBe zarr output (for PatchPerPix, evalinstseg and mcfo_sim's validator)
Add `--zarr` to a `v1` render, or convert renders that already exist:

```bash
python -m swc2img_py v1 brain_00049 --rgb --zarr --bundle D:/1k_v1_neuron_sets/1k_v1_neuron_sets --out renders/
python -m swc2img_py zarr brain_00049 --rgb --bundle D:/1k_v1_neuron_sets/1k_v1_neuron_sets --out renders/   # existing
```

Writes `<name>[_rgb].zarr` in the v1 layout:
- `volumes/raw`: uint16 12-bit, CZYX.
- `volumes/gt_instances`: one mask per neuron, rasterised from the SWCs with mcfo_sim's rule.

Also writes `<name>_gt_channels.csv`, which maps each mask channel to its bodyId and tags.

### Visibility (how much of each neuron stands out from background)
```bash
python -m swc2img_py visibility brain_00091 brain_00049 brain_00381 \
    --bundle D:/1k_v1_neuron_sets/1k_v1_neuron_sets --out renders/ --csv visibility.csv
```

For each neuron, it gives the fraction of its voxels with SNR ≥ 2 (local background) in the greyscale SWC2IMG render
and in the v1 example render. Needs the greyscale renders and the bundle's `examples/`. Method and results:
[NOTES.md §3.1](NOTES.md).

## How it works / gotchas
- SWC2IMG's dialog breaks Fiji `--headless` (`HeadlessException`). We run `ij.ImageJ -batch` with batch mode on
  instead. That still needs a display, so on Greene wrap the call in `xvfb-run -a`.
- SWC2IMG sizes the image to the neuron extent plus a border. To match the v1 field of view we clip nodes to the v1
  box (+5 µm), add two near-invisible anchor nodes at the box corners, then crop using the anchors' positions in
  SWC2IMG's corrected SWC.
- One greyscale channel per render. SWC2IMG has no PSF model (only correlated noise) and no MCFO colour.
- One v1 image at 0.44 µm takes about 70 s and ~0.75 GB per TIFF.

Theory, findings and comparison images: [NOTES.md](NOTES.md).
