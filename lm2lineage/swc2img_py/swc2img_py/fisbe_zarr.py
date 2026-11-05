"""Write SWC2IMG renders in the FISBe / synth_MCFO v1 zarr layout.

Same layout as v1: zarr format 2, ``volumes/raw`` uint16 (C, Z, Y, X) with 12-bit values, chunks (1, 64, 64, 64),
Blosc zstd level 3 with bit-shuffle, and ``volumes/gt_instances`` uint8 (N, Z, Y, X), one binary mask per neuron.
So PatchPerPix, evalinstseg and mcfo_sim's ``validate`` read it unchanged.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import tifffile

from .masks import rasterize
from .swc import clip_to_box, read_swc

CHUNKS = (1, 64, 64, 64)
MAX_ADU = 4095


def _compressor():
    from numcodecs import Blosc
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _open_tif_czyx(tif: Path):
    """Memory-map a render TIFF as (C, Z, Y, X) without loading it (greyscale ZYX -> C=1)."""
    a = tifffile.memmap(tif, mode="r")
    if a.ndim == 3:
        return a[None]
    return np.moveaxis(a, 1, 0)  # ImageJ hyperstack is stored Z, C, Y, X


def _scale_to_12bit(vol_czyx) -> float:
    """One scale for all channels (keeps their relative brightness): brightest voxel -> 4095.

    A percentile doesn't work here: neurons are < 1 % of voxels, so even the 99.99th percentile is background.
    """
    hi = max(float(np.asarray(vol_czyx[c]).max()) for c in range(vol_czyx.shape[0]))
    return MAX_ADU / max(hi, 1e-6)


def neuron_masks(bundle: Path, meta: dict) -> list[tuple[str, np.ndarray]]:
    """(bodyId, flat voxel indices) per neuron, in the cropped render's grid."""
    lo = np.array(meta["crop_lo_um_xyz"])
    hi = np.array(meta["crop_hi_um_xyz"])
    shape = meta.get("shape_zyx") or meta["shape_czyx"][1:]
    out = []
    for b in meta["body_ids"]:
        n = read_swc(bundle / "swc" / f"{b}.swc")
        n["r"] *= meta.get("radius_scale", 1.0)
        out.append((b, rasterize(clip_to_box(n, lo, hi), lo, shape, meta["voxel_um"])))
    return out


def write_fisbe_zarr(tif, meta: dict, bundle_dir, out_zarr) -> Path:
    """Convert one cropped SWC2IMG render (TIFF + its JSON meta) to a FISBe-layout zarr with per-neuron masks."""
    import zarr

    bundle, out_zarr = Path(bundle_dir), Path(out_zarr)
    vol = _open_tif_czyx(Path(tif))
    C, Z, Y, X = vol.shape
    scale = _scale_to_12bit(vol)
    root = zarr.open_group(str(out_zarr), mode="w", zarr_format=2)
    tags = meta.get("channels") or ["grey"]
    root.attrs.update({"voxel_size_um": [meta["voxel_um"]] * 3, "axes": "CZYX", "channels": tags,
                       "source": "SWC2IMG via swc2img_py", "image_id": meta["image_id"]})
    g = root.require_group("volumes")

    raw = g.create_array("raw", shape=(C, Z, Y, X), dtype="uint16", chunks=CHUNKS, compressors=_compressor(),
                         fill_value=0)
    for c in range(C):
        raw[c] = np.clip(np.asarray(vol[c], dtype=np.float32) * scale, 0, MAX_ADU).round().astype(np.uint16)
    raw.attrs.update({"swc2img_float_to_adu_scale": scale, "max_adu": MAX_ADU})

    masks = neuron_masks(bundle, meta)
    gt = g.create_array("gt_instances", shape=(len(masks), Z, Y, X), dtype="uint8", chunks=CHUNKS,
                        compressors=_compressor(), fill_value=0)
    flat = np.zeros(Z * Y * X, np.uint8)
    for i, (_, idx) in enumerate(masks):
        flat[:] = 0
        flat[idx] = 1
        gt[i] = flat.reshape(Z, Y, X)
    gt.attrs["body_ids"] = [b for b, _ in masks]

    tag_of = meta.get("tags", {})
    with open(out_zarr.parent / f"{meta['image_id']}_gt_channels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_channel", "bodyId", "tags", "n_voxels"])
        for i, (b, idx) in enumerate(masks):
            w.writerow([i, b, tag_of.get(b, ""), len(idx)])
    return out_zarr


def convert_render(out_dir, image_id: str, bundle_dir, rgb: bool = False) -> Path:
    """Convert a render already in ``out_dir`` (made by ``render_v1_image`` or ``render_v1_image_rgb``)."""
    out_dir = Path(out_dir)
    stem = f"{image_id}.swc2img_rgb" if rgb else f"{image_id}.swc2img"
    meta = json.load(open(out_dir / f"{stem}.json"))
    if not rgb and "body_ids" not in meta:
        raise ValueError(f"{stem}.json has no body_ids")
    if rgb and "body_ids" not in meta:
        meta["body_ids"] = list(meta["tags"].keys())
    if not meta.get("crop", True):
        raise ValueError("only cropped renders can be converted (the mask grid needs the crop box)")
    return write_fisbe_zarr(out_dir / f"{stem}.tif", meta, bundle_dir, out_dir / f"{image_id}{'_rgb' if rgb else ''}.zarr")
