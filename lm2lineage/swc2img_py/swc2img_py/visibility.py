"""How much of each neuron is visible: SWC2IMG render vs the v1 render, measured the same way.

A neuron voxel counts as visible when it is >= ``k`` noise levels above the local background (default k = 2, the
SNR >= 2 rule mcfo_sim uses for ``gt_instances_visible``) in any channel the neuron is expressed in. Background is the
median and robust sd (1.4826 x MAD) of non-neuron voxels in 32^3 blocks, so uneven haze in v1 is handled. Each render
is scored against its own masks (v1's ``gt_instances``; our SWC-rasterised masks for SWC2IMG), because v1 jittered
neuron placement slightly and the two renders don't align at the voxel level.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .fisbe_zarr import _open_tif_czyx, neuron_masks

BLOCK = 32


def background_maps(vol_zyx: np.ndarray, exclude: np.ndarray, block: int = BLOCK):
    """Per-block background median and robust sd (block grid), ignoring excluded (neuron) voxels."""
    Z, Y, X = vol_zyx.shape
    nb = [-(-s // block) for s in (Z, Y, X)]
    med = np.zeros(nb, np.float32)
    sd = np.zeros(nb, np.float32)
    keep = ~exclude
    glob_vals = vol_zyx[::4, ::4, ::4][keep[::4, ::4, ::4]]
    g_med = float(np.median(glob_vals))
    g_sd = float(1.4826 * np.median(np.abs(glob_vals - g_med))) or float(np.std(glob_vals))
    for i in range(nb[0]):
        for j in range(nb[1]):
            for l in range(nb[2]):
                s = (slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block),
                     slice(l * block, (l + 1) * block))
                v = vol_zyx[s][keep[s]]
                if v.size < 200:
                    med[i, j, l], sd[i, j, l] = g_med, g_sd
                    continue
                m = float(np.median(v))
                mad = 1.4826 * float(np.median(np.abs(v - m)))
                med[i, j, l], sd[i, j, l] = m, (mad if mad > 0 else max(float(v.std()), 1e-3))
    return med, sd


def snr_at(vol_zyx, idx, med, sd, block: int = BLOCK) -> np.ndarray:
    """(I - local background) / local noise at flat voxel indices."""
    Z, Y, X = vol_zyx.shape
    z, rem = np.divmod(idx, Y * X)
    y, x = np.divmod(rem, X)
    bz, by, bx = z // block, y // block, x // block
    return (vol_zyx.reshape(-1)[idx] - med[bz, by, bx]) / sd[bz, by, bx]


def _dilate_union(index_lists, shape, iterations=2) -> np.ndarray:
    from scipy import ndimage as ndi
    u = np.zeros(int(np.prod(shape)), bool)
    for idx in index_lists:
        u[idx] = True
    return ndi.binary_dilation(u.reshape(shape), iterations=iterations)


def score_render(channels_zyx, masks, neuron_channels, k: float = 2.0):
    """For each neuron: (n voxels, visible fraction, median SNR) in a render.

    channels_zyx: list of ZYX float arrays (one per channel); masks: list of flat-index arrays;
    neuron_channels: per neuron, which channel indices it is expressed in.
    """
    shape = channels_zyx[0].shape
    excl = _dilate_union(masks, shape)
    bg = [background_maps(c, excl) for c in channels_zyx]
    out = []
    for idx, chans in zip(masks, neuron_channels):
        if len(idx) == 0:
            out.append((0, np.nan, np.nan))
            continue
        snr = np.max([snr_at(channels_zyx[c], idx, *bg[c]) for c in chans], axis=0)
        out.append((len(idx), float((snr >= k).mean()), float(np.median(snr))))
    return out


TAG_CH = {"HA": 0, "V5": 1, "FLAG": 2}


def compare_image(image_id: str, bundle_dir, render_dir, k: float = 2.0) -> list[dict]:
    """Per-neuron visibility in the greyscale SWC2IMG render and in the v1 example render."""
    import zarr

    bundle, render_dir = Path(bundle_dir), Path(render_dir)
    ex = bundle / "examples" / image_id
    meta = json.load(open(render_dir / f"{image_id}.swc2img.json"))

    # SWC2IMG: greyscale render, our own masks
    ours = np.asarray(_open_tif_czyx(render_dir / f"{image_id}.swc2img.tif")[0], dtype=np.float32)
    our_masks = neuron_masks(bundle, meta)
    s_ours = score_render([ours], [m for _, m in our_masks], [[0]] * len(our_masks), k)
    del ours

    # v1: 3-channel render, its own masks, neuron scored on the channels of its tags
    rows = list(csv.DictReader(open(ex / "gt_channels.csv")))
    g = zarr.open_group(str(ex / f"{image_id}.zarr"), mode="r")["volumes"]
    v1_masks, v1_vis_builtin = [], []
    for r in rows:
        full = np.flatnonzero(np.asarray(g["gt_instances"][int(r["gt_channel"])]))
        vis = int(np.asarray(g["gt_instances_visible"][int(r["gt_channel"])]).sum())
        v1_masks.append(full)
        v1_vis_builtin.append(vis / max(len(full), 1))
    raw = [np.asarray(g["raw"][c], dtype=np.float32) for c in range(g["raw"].shape[0])]
    chans = [[TAG_CH[t] for t in r["tags"].split("+")] for r in rows]
    s_v1 = score_render(raw, v1_masks, chans, k)
    del raw

    v1_by_body = {r["bodyId"]: (s, b, r) for r, s, b in zip(rows, s_v1, v1_vis_builtin)}
    types = {r["bodyId"]: r["type"] for r in csv.DictReader(open(bundle / "neurons.csv")) if r["image_id"] == image_id}
    out = []
    for (body, _), (n_o, f_o, snr_o) in zip(our_masks, s_ours):
        (n_v, f_v, snr_v), builtin, r = v1_by_body.get(body, ((0, np.nan, np.nan), np.nan, {"tags": ""}))
        out.append(dict(image_id=image_id, overlap_mode=meta["overlap_mode"], bodyId=body, type=types.get(body, ""),
                        tags=r["tags"], swc2img_voxels=n_o, swc2img_visible_frac=round(f_o, 3),
                        swc2img_median_snr=round(snr_o, 2), v1_voxels=n_v, v1_visible_frac=round(f_v, 3),
                        v1_median_snr=round(snr_v, 2), v1_builtin_visible_frac=round(builtin, 3)))
    return out


def write_csv(rows: list[dict], path) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return Path(path)
