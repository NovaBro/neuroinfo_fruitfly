"""Render manifest rows to FISBe-format zarr stacks."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from ..imaging import ImagingParams, Tissue
from ..scene import render_scene
from .config import BrainConfig
from .pool import _get, parse_swc


def tissue_for_crop(pool_dir, cfg: BrainConfig, lo_um, shape, rng, p, coarse_um=4.0):
    """Neuropil haze (closed, filled pool occupancy) + cortex (soma density),
    resampled from the template grid onto the crop's coarse grid."""
    t = np.load(os.path.join(pool_dir, "tissue_grid.npz"))
    occ, soma, cell = t["occupancy"], t["soma"], float(t["cell_um"])   # x, y, z grids
    neuropil = ndi.binary_closing(occ >= 2, iterations=2)
    for z in range(neuropil.shape[2]):                                   # fill each frontal section
        neuropil[:, :, z] = ndi.binary_fill_holes(neuropil[:, :, z])
    tissue = ndi.binary_dilation(neuropil, iterations=2) | (soma > 0)
    for z in range(tissue.shape[2]):
        tissue[:, :, z] = ndi.binary_fill_holes(ndi.binary_closing(tissue[:, :, z], iterations=3))
    haze_t = ndi.gaussian_filter(tissue.astype(np.float32), 0.75)
    cortex_t = ndi.gaussian_filter(soma.astype(np.float32), 1.0)
    cortex_t /= np.percentile(cortex_t[cortex_t > 0], 99) + 1e-9 if (cortex_t > 0).any() else 1.0
    cshape = np.maximum(np.ceil(np.array(shape) * p.voxel_um / coarse_um).astype(int), 4)  # z, y, x
    zc, yc, xc = [(np.arange(n) + 0.5) * coarse_um for n in cshape]
    Zi, Yi, Xi = np.meshgrid(zc, yc, xc, indexing="ij")
    idx = [(Xi + lo_um[0]) / cell - 0.5, (Yi + lo_um[1]) / cell - 0.5, (Zi + lo_um[2]) / cell - 0.5]
    haze = ndi.map_coordinates(haze_t, idx, order=1, mode="constant")
    cortex = np.clip(ndi.map_coordinates(cortex_t, idx, order=1, mode="constant"), 0, 1.5)
    n = ndi.gaussian_filter(rng.normal(size=cshape), 2.5)
    haze *= np.exp(0.25 * n / (n.std() + 1e-9))
    il = ndi.gaussian_filter(rng.normal(size=cshape), 10.0)
    illum = np.exp(p.illum_sigma * il / (il.std() + 1e-9))
    return Tissue(haze.astype(np.float32), cortex.astype(np.float32), illum.astype(np.float32),
                  coarse_um, np.zeros((0, 3)))


def _elastic(rng, extent_um, amp_um, spacing=40.0):
    g = np.maximum(2, np.ceil(extent_um / spacing).astype(int) + 3)
    d = np.stack([ndi.gaussian_filter(rng.normal(0, 1, g), 1.0) for _ in range(3)])
    return d * (amp_um / (d.std() + 1e-9)), spacing


def render_row(img, neu, pool_dir, cache, out_dir, cfg: BrainConfig, save_mip=True, log=None):
    rng = np.random.default_rng(int(img["render_seed"]))
    p = ImagingParams(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in json.loads(img["imaging"]).items()})
    lo_um = np.array(json.loads(img["crop_lo_um"]))
    hi_um = np.array(json.loads(img["crop_hi_um"]))
    extent = hi_um - lo_um
    shape = tuple(np.ceil(extent / p.voxel_um).astype(int)[::-1])    # Z, Y, X
    center = (lo_um + hi_um) / 2
    R = Rotation.from_euler("xyz", json.loads(img["rot_deg"]), degrees=True).as_matrix()
    S = np.array(json.loads(img["scale"]))
    disp, sp = _elastic(rng, extent, cfg.elastic_um)

    def place(xyz):
        q = ((xyz - center) @ R.T) * S + center - lo_um
        gi = (q / sp + 1.0).T
        return q + np.stack([ndi.map_coordinates(c, gi, order=1, mode="nearest") for c in disp], 1)

    tissue = tissue_for_crop(pool_dir, cfg, lo_um, shape, rng, p)

    def neurons():
        for r in neu.itertuples(index=False):
            fp = _get(f"{cfg.bucket}/{cfg.swc_dir}/{r.bodyId}.swc",
                      os.path.join(cache, "swc_jrc2018u", f"{r.bodyId}.swc"))
            xyz, rad, parent = parse_swc(open(fp).read())
            c = np.flatnonzero(parent >= 0)
            k = int(np.argmax(rad))
            yield dict(xyz=place(xyz), radius=rad * cfg.radius_scale,
                       edges=np.stack([c, parent[c]], 1),
                       soma_idx=k if rad[k] >= 1.5 else None,
                       levels=[r.level_HA, r.level_V5, r.level_FLAG],
                       meta=dict(bodyId=int(r.bodyId), manifest_instance=int(r.instance), type=r.type,
                                 superclass=r.superclass, itoleeHl=_s(r.itoleeHl),
                                 trumanHl=_s(r.trumanHl), somaSide=_s(r.somaSide),
                                 overlap_degree=int(r.overlap_degree)))

    extra = dict(image_id=img["image_id"], split=img["split"], overlap_mode=img["overlap_mode"],
                 count_bin=img["count_bin"], n_neurons_planned=int(img["n_neurons"]),
                 source="male-cns v1.0 brain, skeletons in JRC2018U",
                 crop_lo_um=lo_um.tolist(), crop_hi_um=hi_um.tolist(), render_seed=int(img["render_seed"]))
    return render_scene(neurons(), tissue, p, shape, rng, out_dir, img["image_id"], extra_meta=extra,
                        save_mip=save_mip, log=log)


def _s(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def render_many(manifest_dir, pool_dir, cache, out_dir, rows=None, shard=None, save_mip=True, log=print):
    """rows: list of image ids or slice; shard: (k, n) renders every n-th row starting at k."""
    cfg = BrainConfig.from_json(open(os.path.join(manifest_dir, "config.json")).read())
    imgs = pd.read_parquet(os.path.join(manifest_dir, "manifest_images.parquet"))
    neus = pd.read_parquet(os.path.join(manifest_dir, "manifest_neurons.parquet"))
    if rows:
        imgs = imgs[imgs.image_id.isin(rows)]
    if shard:
        k, n = shard
        imgs = imgs.iloc[k::n]
    by_img = dict(tuple(neus.groupby("image_id")))
    t0 = time.time()
    for _, img in imgs.iterrows():
        if os.path.exists(os.path.join(out_dir, f"{img.image_id}.json")):
            continue                                        # resumable
        neu = by_img.get(img.image_id, neus.iloc[:0])
        meta = render_row(img, neu, pool_dir, cache, out_dir, cfg, save_mip=save_mip)
        log(f"[{time.time() - t0:7.0f}s] {img.image_id} planned {len(neu)} "
            f"rendered {len(meta['instances'])} ({img.overlap_mode}, {img.count_bin})")
