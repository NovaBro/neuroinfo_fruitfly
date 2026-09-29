"""Scene renderer shared by the VNC (`simulate`) and brain (`brain.render`) front ends.

Input: neurons already placed in image microns (X, Y, Z; origin at the stack
corner), a Tissue background model, ImagingParams, and an RNG. Output: a
FISBe-compatible zarr (raw CZYX uint16, gt_instances NZYX uint8, gt_labels),
a JSON sidecar, centreline voxel coordinates, and optionally a MIP preview.
"""
from __future__ import annotations

import json
import os

import numcodecs
import numpy as np
import zarr
from PIL import Image
from scipy.ndimage import gaussian_filter

from .imaging import detector, psf_blur, puncta_points, texture_field
from .labeling import TAGS, along_cable_modulation
from .render import (bbox_vox, centerline_samples, membrane_points,
                     rasterize_mask, splat)


COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)  # as FISBe


def open_sample(zpath):
    """(raw, gt_instances, gt_labels|None) for FISBe-layout or flat zarr samples."""
    root = zarr.open_group(zpath, mode="r")
    g = root["volumes"] if "volumes" in root else root
    return g["raw"], g["gt_instances"], (g["gt_labels"] if "gt_labels" in g else None)


def _peak_norm(sig_vox):
    return float((2 * np.pi) ** 1.5 * np.prod(sig_vox))


def render_scene(neurons, tissue, p, shape, rng, out_dir, name, extra_meta=None,
                 vis_snr=2.0, bead_prob=0.6, bead_per_um=0.25, save_mip=True, log=None, dim_snr=5.0):
    """neurons: iterable of dicts with keys
         xyz (N,3 image um), radius (N,), edges (E,2), soma_idx (int|None),
         levels (3,) per-tag expression (0 = tag off), meta (dict, logged).
    """
    log = log or (lambda *a: None)
    voxel = p.voxel_um
    Z = shape[0]
    shape_um_xyz = np.array(shape[::-1]) * voxel
    att_z = np.exp(-(np.arange(Z) + 0.5) * voxel / p.atten_length_um).astype(np.float32)
    r2 = (p.read_noise_adu / p.adu_per_photon) ** 2
    acc = np.zeros((3,) + tuple(shape), np.float32)
    sig = np.array([p.psf_sigma_z_um, p.psf_sigma_xy_um, p.psf_sigma_xy_um]) * p.deep_blur_factor / voxel
    pad = int(np.ceil(3 * sig.max())) + 2
    thin_r = 0.6 * voxel
    instances, skipped = [], []

    for nd in neurons:
        xyz, rad, levels = nd["xyz"], nd["radius"], np.asarray(nd["levels"], float)
        pts, crad, ds, dirs = centerline_samples(xyz, rad, nd["edges"], step=voxel / 2)
        inside = np.all((pts >= 0) & (pts < shape_um_xyz), axis=1)
        if inside.sum() < 20:
            skipped.append(dict(nd["meta"], reason="outside_volume"))
            continue
        pts, crad, ds, dirs = pts[inside], crad[inside], ds[inside], dirs[inside]
        soma_xyz = soma_r = None
        si = nd.get("soma_idx")
        if si is not None and np.all((xyz[si] >= 0) & (xyz[si] < shape_um_xyz)):
            soma_xyz, soma_r = xyz[si], float(rad[si])
        mod = along_cable_modulation(pts, rng)
        mp, mw = membrane_points(pts, crad, ds * mod, dirs, soma_xyz, soma_r,
                                 thin_r=thin_r, ring_spacing=voxel / 2, rng=rng)
        thin = np.flatnonzero(crad < thin_r)
        rate = rng.lognormal(np.log(bead_per_um), 0.8) if rng.random() < bead_prob else 0.0
        nb = rng.poisson(rate * ds[thin].sum()) if thin.size else 0
        if nb:
            bi = rng.choice(thin, nb, p=ds[thin] / ds[thin].sum())
            br = rng.uniform(0.25, 0.6, nb)
            mp = np.vstack([mp, pts[bi]])
            mw = np.concatenate([mw, 4 * np.pi * br ** 2 * mod[bi]])
            gt_pts, gt_rad = np.vstack([pts, pts[bi]]), np.concatenate([crad, br])
        else:
            gt_pts, gt_rad = pts, crad
        lo, hi = bbox_vox(np.vstack([mp, pts]), voxel, shape, pad=pad)
        local = psf_blur(splat(mp, mw, voxel, lo, hi), lo[0], p, Z)
        cv = np.floor(pts[:, ::-1] / voxel).astype(int) - lo
        ok = np.all((cv >= 0) & (cv < local.shape), axis=1)
        sel = ok & (crad < thin_r) if (ok & (crad < thin_r)).sum() > 20 else ok
        m = np.median(local[tuple(cv[sel].T)])
        local *= p.neuron_photons / max(m, 1e-12)
        atten_il = att_z[lo[0]:hi[0], None, None] * tissue.local_grid("illum", lo, hi, voxel)
        local *= atten_il
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        snr = np.zeros(local.shape, np.float32)
        haze = tissue.local_grid("haze", lo, hi, voxel)
        cortex = tissue.local_grid("cortex", lo, hi, voxel)
        for c in np.flatnonzero(levels > 0):
            s_c = local * levels[c]
            acc[c][sl] += s_c
            bg_c = (p.haze_photons[c] * haze + p.cortex_photons[c] * cortex) * atten_il
            snr = np.maximum(snr, s_c / np.sqrt(s_c + bg_c + r2))
        full = rasterize_mask(gt_pts, gt_rad, voxel, lo, hi, soma_xyz, soma_r)
        vis = full & (snr >= vis_snr)
        if vis.sum() == 0:
            skipped.append(dict(nd["meta"], reason="below_visibility"))
            continue
        instances.append(dict(
            lo=lo, hi=hi, mask=vis, brightness=float(np.median(snr[vis])),
            centerline_zyx=np.unique(np.floor(pts[:, ::-1] / voxel).astype(np.int32), axis=0),
            meta=dict(nd["meta"], soma_in_volume=soma_xyz is not None,
                      tags=[t for t, l in zip(TAGS, levels) if l > 0],
                      levels=[round(float(l), 3) for l in levels],
                      visible_voxels=int(vis.sum()), total_voxels=int(full.sum()),
                      median_snr=round(float(np.median(snr[vis])), 2))))
        log(f"  {nd['meta'].get('bodyId')} tags={instances[-1]['meta']['tags']} "
            f"visible={vis.sum()}/{full.sum()} vox")

    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, f"{name}.zarr")
    root = zarr.open_group(zpath, mode="w", zarr_format=2)
    vol = root.require_group("volumes")              # FISBe layout: <sample>.zarr/volumes/{raw,gt_instances}
    chunks = (1, 64, 64, 64)
    raw = vol.create_array("raw", shape=(3,) + tuple(shape), dtype="uint16", chunks=chunks,
                           compressors=COMPRESSOR)
    total = acc.sum(0)
    haze_f = tissue.upsample("haze", shape, voxel)
    haze_f *= texture_field(shape, p, rng)
    cortex_f = tissue.upsample("cortex", shape, voxel)
    atten_il_f = tissue.upsample("illum", shape, voxel) * att_z[:, None, None]
    psf_vox = np.array([p.psf_sigma_z_um, p.psf_sigma_xy_um, p.psf_sigma_xy_um]) / voxel
    blob_sig = 1.2 / voxel
    mip, gains = [], []
    for c in range(3):
        bg = p.haze_photons[c] * haze_f + p.cortex_photons[c] * cortex_f
        sp = puncta_points(tissue, p.speckle_per_1e4um3[c], rng)
        if len(sp):
            w = rng.lognormal(np.log(p.speckle_photons), 0.5, len(sp)) * _peak_norm(psf_vox)
            bg += psf_blur(splat(sp, w, voxel, np.zeros(3, int), np.array(shape)), 0, p, Z)
        bl = puncta_points(tissue, p.blob_per_1e4um3[c], rng, weight_field="cortex")
        if len(bl):
            w = rng.lognormal(np.log(p.blob_photons), 0.4, len(bl)) * _peak_norm([blob_sig] * 3)
            bg += gaussian_filter(splat(bl, w, voxel, np.zeros(3, int), np.array(shape)), blob_sig)
        expected = bg * atten_il_f
        del bg
        expected += acc[c] * (1 - 1.5 * p.crosstalk) + (total - acc[c]) * (p.crosstalk / 2)
        s = acc[c][acc[c] > 1.0]
        gain = p.adu_per_photon
        if s.size > 100:
            gain = float(np.clip(p.gain_target * (p.max_adu - p.offset_adu) / np.percentile(s, 99.9),
                                 0.5, 3 * p.adu_per_photon))
        gains.append(gain)
        img = detector(expected, p, rng, adu_per_photon=gain)
        raw[c] = img
        mip.append(img.max(0))
        del expected, img
    del acc, total, haze_f, cortex_f, atten_il_f

    n = len(instances)
    gt = vol.create_array("gt_instances", shape=(max(n, 1),) + tuple(shape), dtype="uint8", chunks=chunks,
                          compressors=COMPRESSOR)
    labels = np.zeros(shape, np.uint16)
    for i, ins in enumerate(instances):
        lo, hi = ins["lo"], ins["hi"]
        gt[i, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = ins["mask"].astype(np.uint8)
    for i in np.argsort([ins["brightness"] for ins in instances]):
        lo, hi = instances[i]["lo"], instances[i]["hi"]
        labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][instances[i]["mask"]] = i + 1
    vol.create_array("gt_labels", shape=tuple(shape), dtype="uint16", chunks=chunks[1:],
                     compressors=COMPRESSOR)[:] = labels
    overlap = _overlap_summary(instances, shape)
    # FISBe marks low-intensity neurons in gt_instances/.zattrs "dim_neurons" (1-based ids)
    dim = [i + 1 for i, ins in enumerate(instances) if ins["meta"]["median_snr"] < dim_snr]
    gt.attrs.update(dict(dim_neurons=dim))
    root.attrs.update(dict(voxel_size_um=[voxel] * 3, axes="CZYX", channels=list(TAGS)))
    meta = dict(name=name, shape_zyx=list(map(int, shape)), voxel_um=voxel,
                imaging={k: (list(v) if isinstance(v, tuple) else v) for k, v in p.__dict__.items()},
                adu_per_photon_per_channel=gains,
                instances=[dict(index=i + 1, **ins["meta"]) for i, ins in enumerate(instances)],
                overlap=overlap,
                not_rendered=skipped, **(extra_meta or {}))
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=1, default=_json_default)
    np.savez_compressed(os.path.join(out_dir, f"{name}_skeletons.npz"),
                        **{f"inst_{i + 1}": ins["centerline_zyx"] for i, ins in enumerate(instances)})
    if save_mip:
        save_mip_png(np.stack(mip), labels, os.path.join(out_dir, f"{name}_mip.png"), p.offset_adu)
    return meta


def _overlap_summary(instances, shape):
    """Realised overlap of the ground-truth masks, in 3D and in the Z-projection."""
    n = len(instances)
    cnt = np.zeros(shape, np.uint8)
    cnt2 = np.zeros(shape[1:], np.uint8)
    for ins in instances:
        lo, hi = ins["lo"], ins["hi"]
        cnt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += ins["mask"]
        cnt2[lo[1]:hi[1], lo[2]:hi[2]] += ins["mask"].any(0)
    pairs3d, pairs2d, same3d = 0, 0, 0
    for a in range(n):
        A = instances[a]
        for b in range(a + 1, n):
            B = instances[b]
            lo = np.maximum(A["lo"], B["lo"])
            hi = np.minimum(A["hi"], B["hi"])
            if np.any(hi <= lo):
                continue
            ma = A["mask"][tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, A["lo"]))]
            mb = B["mask"][tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, B["lo"]))]
            if (ma & mb).any():
                pairs3d += 1
                same3d += A["meta"]["tags"] == B["meta"]["tags"]
            if (ma.any(0) & mb.any(0)).any():
                pairs2d += 1
    for ins in instances:
        lo, hi = ins["lo"], ins["hi"]
        m = ins["mask"]
        ins["meta"]["frac_voxels_shared_3d"] = round(float((cnt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][m] > 1).mean()), 4)
        p = m.any(0)
        ins["meta"]["frac_pixels_shared_mip"] = round(float((cnt2[lo[1]:hi[1], lo[2]:hi[2]][p] > 1).mean()), 4)
    fg = cnt > 0
    return dict(pairs_touching_3d=int(pairs3d), pairs_overlapping_mip=int(pairs2d),
                same_colour_pairs_touching_3d=int(same3d),
                frac_fg_voxels_shared_3d=round(float((cnt > 1).sum() / max(fg.sum(), 1)), 4),
                frac_fg_pixels_shared_mip=round(float((cnt2 > 1).sum() / max((cnt2 > 0).sum(), 1)), 4))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_mip_png(mip, labels, path, offset):
    rgb = []
    for c in range(3):
        m = np.clip(mip[c].astype(np.float32) - offset, 0, None)
        m /= np.percentile(m, 99.9) + 1e-9
        rgb.append(np.clip(m, 0, 1) ** 0.8)
    img = (np.stack(rgb, -1) * 255).astype(np.uint8)
    lab_mip = np.zeros(labels.shape[1:], np.uint16)
    for z in range(labels.shape[0]):
        sl = labels[labels.shape[0] - 1 - z]
        lab_mip[sl > 0] = sl[sl > 0]
    pal = np.random.default_rng(0).integers(60, 255, (int(labels.max()) + 1, 3)).astype(np.uint8)
    pal[0] = 0
    Image.fromarray(np.concatenate([img, pal[lab_mip]], axis=1)).save(path)
