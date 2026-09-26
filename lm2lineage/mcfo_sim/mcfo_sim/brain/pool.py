"""Neuron pool: which male CNS brain neurons can appear, and where they are.

For every pool neuron we download its SWC skeleton that the male CNS release
already registered to the JRC2018 unisex brain template (microns; the same
space FlyLight/FISBe imagery is registered to) and store:
  * annotations (type, superclass, hemilineages, side) and a train/val/test
    split assigned by cell type (so test neurons are never seen in training);
  * a coarse occupancy footprint (4 um cells) used for fast crop membership
    and overlap-aware selection in the manifest step;
  * cable length, bounding box, soma position (largest-radius node);
and, over the whole pool, a tissue grid (neurite occupancy + soma density)
that the renderer turns into neuropil haze and the cell-body rind.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import io
import os
import urllib.request

import numpy as np
import pandas as pd

from .config import BrainConfig


def _get(url, path, retries=3):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return path
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
    return path


def split_of(type_name, seed, fractions):
    h = int(hashlib.sha1(f"{seed}:{type_name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    c = np.cumsum(fractions)
    return ("train", "val", "test")[int(np.searchsorted(c, h, side="right").clip(0, 2))]


def parse_swc(text):
    a = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    ids = a[:, 0].astype(np.int64)
    lut = {n: i for i, n in enumerate(ids)}
    parent = np.array([lut.get(int(p), -1) for p in a[:, 6]], dtype=np.int64)
    return a[:, 2:5], a[:, 5], parent


def resample(xyz, parent, step):
    c = np.flatnonzero(parent >= 0)
    p0, p1 = xyz[c], xyz[parent[c]]
    L = np.linalg.norm(p1 - p0, axis=1)
    n = np.maximum(1, np.ceil(L / step).astype(int))
    i = np.repeat(np.arange(len(c)), n)
    t = (np.arange(i.size) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[i]
    return np.vstack([p0[i] + (p1[i] - p0[i]) * t[:, None], xyz]), float(L.sum())


def grid_shape(cfg: BrainConfig):
    return tuple(int(np.ceil(b / cfg.footprint_um)) + 1 for b in cfg.template_bbox_um)  # x,y,z


def neuron_summary(body_id, swc_path, cfg: BrainConfig):
    xyz, r, parent = parse_swc(open(swc_path).read())
    pts, cable = resample(xyz, parent, cfg.footprint_um / 2)
    g = np.array(grid_shape(cfg))
    cells = np.clip(np.floor(pts / cfg.footprint_um).astype(np.int64), 0, g - 1)
    flat = np.unique(np.ravel_multi_index(cells.T, g))
    k = int(np.argmax(r))
    soma = xyz[k] if r[k] >= 1.5 else np.full(3, np.nan)
    return dict(bodyId=int(body_id), cable_um=round(cable, 1), n_nodes=len(xyz),
                n_cells=len(flat), soma_x=soma[0], soma_y=soma[1], soma_z=soma[2],
                soma_r=float(r[k]) if r[k] >= 1.5 else np.nan,
                bb_x0=xyz[:, 0].min(), bb_y0=xyz[:, 1].min(), bb_z0=xyz[:, 2].min(),
                bb_x1=xyz[:, 0].max(), bb_y1=xyz[:, 1].max(), bb_z1=xyz[:, 2].max()), flat


def build_pool(cfg: BrainConfig, cache: str, out: str, max_neurons: int = 0, workers: int = 16,
               log=print):
    os.makedirs(out, exist_ok=True)
    ann_fp = _get(f"{cfg.bucket}/{cfg.annotations}", os.path.join(cache, "annotations.feather"))
    ann = pd.read_feather(ann_fp)
    pool = ann[ann.superclass.isin(cfg.superclasses) & ann.status.isin(cfg.statuses)].copy()
    pool["type"] = pool["type"].fillna(pool["bodyId"].map(lambda b: f"untyped_{b}"))
    if max_neurons and len(pool) > max_neurons:
        # keep whole cell types so "natural" (GAL4-line-like) images stay possible
        rng = np.random.default_rng(cfg.seed)
        types = np.array(pool.type.unique(), dtype=object)
        rng.shuffle(types)
        sizes = pool.groupby("type").size()
        keep = types[np.cumsum(sizes.loc[types].values) <= max_neurons]
        pool = pool[pool.type.isin(keep)]
    log(f"pool: {len(pool)} neurons, {pool.type.nunique()} types")

    def fetch(b):
        p = os.path.join(cache, "swc_jrc2018u", f"{b}.swc")
        try:
            _get(f"{cfg.bucket}/{cfg.swc_dir}/{b}.swc", p)
            return b, p
        except Exception:  # noqa: BLE001
            return b, None

    rows, fps, done = [], [], 0
    g = grid_shape(cfg)
    occ = np.zeros(g, np.int32)
    soma_grid = np.zeros(g, np.int32)
    with cf.ThreadPoolExecutor(workers) as ex:
        for b, p in ex.map(fetch, pool.bodyId.tolist()):
            done += 1
            if p is None:
                continue
            try:
                s, flat = neuron_summary(b, p, cfg)
            except Exception:  # noqa: BLE001
                continue
            rows.append(s)
            fps.append(flat.astype(np.int32))
            occ.ravel()[flat] += 1
            if np.isfinite(s["soma_x"]):
                c = np.clip(np.floor(np.array([s["soma_x"], s["soma_y"], s["soma_z"]])
                                     / cfg.footprint_um).astype(int), 0, np.array(g) - 1)
                soma_grid[tuple(c)] += 1
            if done % 2000 == 0:
                log(f"  {done}/{len(pool)} skeletons")
    summ = pd.DataFrame(rows)
    meta = pool[["bodyId", "type", "instance", "superclass", "itoleeHl", "trumanHl",
                 "somaSide", "flywireType", "statusLabel"]]
    df = summ.merge(meta, on="bodyId", how="left")
    df["split"] = [split_of(t, cfg.seed, cfg.split_fractions) for t in df.type]
    lens = np.array([len(f) for f in fps])
    df["fp_offset"] = np.concatenate([[0], np.cumsum(lens)[:-1]])
    df["fp_len"] = lens
    df.to_parquet(os.path.join(out, "pool.parquet"), index=False)
    np.save(os.path.join(out, "footprints.npy"), np.concatenate(fps) if fps else np.zeros(0, np.int32))
    np.savez_compressed(os.path.join(out, "tissue_grid.npz"), occupancy=occ, soma=soma_grid,
                        cell_um=cfg.footprint_um)
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(cfg.to_json())
    log(f"pool written: {len(df)} neurons with skeletons -> {out}")
    return df
