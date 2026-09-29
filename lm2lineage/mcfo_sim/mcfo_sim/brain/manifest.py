"""Deterministic design of the whole image set (no pixels rendered here).

For each image the manifest fixes: split, crop, target neuron count and overlap
mode (from a balanced design grid), the chosen body ids, their MCFO tag
combinations and expression levels, the placement jitter, all imaging
parameters and the render seed. Rendering any row later reproduces the same
image bit-for-bit (same code version + pool).

Overlap modes
  isolated : no two neurons come within one 4 um footprint cell of each other
  natural  : GAL4-line-like - pick cell types, all their in-crop members express,
             then sample the target number (co-expressed types overlap naturally)
  crowded  : grow the set around a seed neuron, preferring candidates that share
             the most footprint cells with those already chosen; optionally force
             an overlapping pair to share the same MCFO colour
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..imaging import ImagingParams
from .config import BrainConfig
from .pool import grid_shape


def design_grid(cfg: BrainConfig, rng):
    """Largest-remainder allocation of n_images over (count bin x overlap mode)."""
    cells, w = [], []
    for bi, bw in enumerate(cfg.count_weights):
        lo, hi = cfg.count_bins[bi]
        for mi, mw in enumerate(cfg.overlap_weights):
            mode = cfg.overlap_modes[mi]
            if mode == "isolated" and lo > cfg.isolated_max_n:
                continue
            cells.append((bi, mode))
            w.append(bw * mw)
    w = np.array(w) / np.sum(w)
    raw = w * cfg.n_images
    n = np.floor(raw).astype(int)
    n[np.argsort(-(raw - n))[:cfg.n_images - n.sum()]] += 1
    plan = [c for c, k in zip(cells, n) for _ in range(k)]
    rng.shuffle(plan)
    return plan


def sample_imaging(rng, voxel, cal=None):
    p = ImagingParams.random(rng, voxel)
    if cal:  # ranges measured from FISBe by `validate calibrate`
        u = rng.uniform
        p.offset_adu = u(*cal["offset_adu"])
        p.read_noise_adu = u(*cal["read_noise_adu"])
        p.gain_target = u(*cal["gain_target"])
        p.max_adu = int(cal["max_adu"])
        if "bg_scale" in cal:
            s = u(*cal["bg_scale"])
            p.haze_photons = tuple(h * s for h in p.haze_photons)
            p.cortex_photons = tuple(h * s for h in p.cortex_photons)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()}


class Pool:
    def __init__(self, pool_dir, cfg: BrainConfig):
        self.cfg = cfg
        self.df = pd.read_parquet(os.path.join(pool_dir, "pool.parquet")).reset_index(drop=True)
        self.fp = np.load(os.path.join(pool_dir, "footprints.npy"))
        self.g = np.array(grid_shape(cfg))
        xyz = np.stack(np.unravel_index(self.fp.astype(np.int64), self.g), 1).astype(np.int16)
        self.cx, self.cy, self.cz = xyz.T
        self.owner = np.repeat(np.arange(len(self.df)), self.df.fp_len.values)
        self.off = self.df.fp_offset.values
        self.len = self.df.fp_len.values
        self.cell_order = np.argsort(self.fp, kind="stable")   # global cell-sorted order, once

    def cells(self, i):
        return self.fp[self.off[i]:self.off[i] + self.len[i]]

    def in_crop_counts(self, lo_um, hi_um):
        lo = np.floor(np.array(lo_um) / self.cfg.footprint_um).astype(int)
        hi = np.ceil(np.array(hi_um) / self.cfg.footprint_um).astype(int)
        inside = ((self.cx >= lo[0]) & (self.cx < hi[0]) & (self.cy >= lo[1]) & (self.cy < hi[1])
                  & (self.cz >= lo[2]) & (self.cz < hi[2]))
        return np.bincount(self.owner[inside], minlength=len(self.df)), inside


def _dilate(cells, g, k):
    if k <= 0:
        return set(cells.tolist())
    xyz = np.stack(np.unravel_index(cells.astype(np.int64), g), 1)
    r = np.arange(-k, k + 1)
    off = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    d = np.clip((xyz[:, None, :] + off[None]).reshape(-1, 3), 0, g - 1)
    return set(np.unique(np.ravel_multi_index(d.T, g)).tolist())


class CropIndex:
    """In-crop footprint entries, indexed both by neuron and by cell."""

    def __init__(self, pool: Pool, inside):
        # footprints are stored grouped by neuron, so masking keeps owner order
        self.own_s, self.cells_by_own = pool.owner[inside], pool.fp[inside]
        idx = pool.cell_order[inside[pool.cell_order]]
        self.cell_s, self.own_by_cell = pool.fp[idx], pool.owner[idx]
        self._cache = {}

    def __getitem__(self, i):
        i = int(i)
        if i not in self._cache:
            a, b = np.searchsorted(self.own_s, [i, i + 1])
            self._cache[i] = set(self.cells_by_own[a:b].tolist())
        return self._cache[i]

    def owners_of(self, cells):
        cells = np.fromiter(cells, np.int64) if not isinstance(cells, np.ndarray) else cells
        if cells.size == 0:
            return np.zeros(0, np.int64)
        a = np.searchsorted(self.cell_s, cells, "left")
        b = np.searchsorted(self.cell_s, cells, "right")
        idx = np.concatenate([np.arange(x, y) for x, y in zip(a, b) if y > x] or [np.zeros(0, int)])
        return self.own_by_cell[idx]


def choose_neurons(pool: Pool, cand, crop_cells: CropIndex, n, mode, rng, cfg):
    """Returns (list of pool row indices, info dict)."""
    info = {}
    if len(cand) == 0:
        return [], info
    if mode == "natural":
        types = pool.df.type.values[cand]
        uniq = np.unique(types)
        rng.shuffle(uniq)
        expr, chosen_types = [], []
        want = n * rng.uniform(1.5, 4.0)
        for t in uniq:
            expr.extend(cand[types == t].tolist())
            chosen_types.append(str(t))
            if len(expr) >= want:
                break
        expr = np.array(expr)
        pick = rng.choice(expr, size=min(n, len(expr)), replace=False)
        info.update(expression_set_size=int(len(expr)), expressed_types=chosen_types)
        return [int(x) for x in pick], info
    order = rng.permutation(cand)
    if mode == "isolated":
        chosen, blocked = [], set()
        for i in order[:20000]:
            c = crop_cells[i]
            if blocked.isdisjoint(c):
                chosen.append(int(i))
                blocked |= _dilate(np.fromiter(c, np.int64), pool.g, cfg.isolation_margin_cells)
                if len(chosen) >= n:
                    break
        return chosen, info
    # crowded: incremental shared-cell counts via the cell -> owners index
    is_cand = np.zeros(len(pool.df), bool)
    is_cand[cand] = True
    ov = np.zeros(len(pool.df), np.int64)
    chosen, union = [], set()

    def add(i):
        new = crop_cells[i] - union
        union.update(new)
        np.add.at(ov, crop_cells.owners_of(new), 1)
        chosen.append(int(i))
        is_cand[i] = False

    add(order[0])
    while len(chosen) < n and is_cand.any():
        score = np.where(is_cand, ov, -1)
        top = np.argsort(-score)[:10]
        top = top[score[top] > 0] if (score[top] > 0).any() else top[score[top] >= 0]
        if top.size == 0:
            break
        add(int(rng.choice(top)))
    return chosen, info


def overlap_stats(ids, crop_cells, tags):
    n = len(ids)
    pairs, same = [], 0
    deg = np.zeros(n, int)
    for a in range(n):
        for b in range(a + 1, n):
            k = len(crop_cells[ids[a]] & crop_cells[ids[b]])
            if k:
                pairs.append((a, b, k))
                deg[a] += 1
                deg[b] += 1
                same += tags[a] == tags[b]
    return dict(overlapping_pairs=len(pairs), same_colour_overlapping_pairs=int(same),
                frac_neurons_overlapping=round(float((deg > 0).mean()) if n else 0.0, 3),
                max_overlap_degree=int(deg.max()) if n else 0,
                shared_cells_total=int(sum(p[2] for p in pairs))), pairs, deg


def build_manifest(pool_dir, out, cfg: BrainConfig, calibration=None, log=print):
    rng = np.random.default_rng(cfg.seed)
    pool = Pool(pool_dir, cfg)
    cal = json.load(open(calibration)) if calibration else None
    plan = design_grid(cfg, rng)
    split_of_image = rng.choice(["train", "val", "test"], size=len(plan), p=cfg.split_fractions)
    splits = pool.df.split.values
    img_rows, neu_rows = [], []
    tb = np.array(cfg.template_bbox_um)
    crop = np.array(cfg.crop_um)
    for k, (bi, mode) in enumerate(plan):
        lo_n, hi_n = cfg.count_bins[bi]
        n_target = int(rng.integers(lo_n, hi_n + 1))
        cx = rng.uniform(*cfg.crop_center_x_um)
        center = np.array([cx, tb[1] / 2 + rng.uniform(-10, 10), tb[2] / 2 + rng.uniform(-5, 5)])
        lo_um = center - crop / 2
        hi_um = center + crop / 2
        counts, inside = pool.in_crop_counts(lo_um, hi_um)
        ok = (counts * cfg.footprint_um >= cfg.min_cable_in_crop_um) & (splits == split_of_image[k])
        cand = np.flatnonzero(ok)
        crop_cells = CropIndex(pool, inside)
        ids, info = choose_neurons(pool, cand, crop_cells, n_target, mode, rng, cfg)
        # MCFO colours + expression levels
        q = rng.uniform(*(cal["q_excision"] if cal and "q_excision" in cal else (0.2, 0.6)))
        tags = []
        for _ in ids:
            t = rng.random(3) < q
            while not t.any():
                t = rng.random(3) < q
            tags.append(tuple(bool(x) for x in t))
        forced = False
        if mode == "crowded" and len(ids) > 1 and rng.random() < cfg.same_colour_overlap_p:
            st, pairs, _ = overlap_stats(ids, crop_cells, tags)
            if pairs:
                a, b, _ = pairs[int(rng.integers(len(pairs)))]
                tags[b] = tags[a]
                forced = True
        st, pairs, deg = overlap_stats(ids, crop_cells, tags)
        type_level = {}
        img_id = f"brain_{k:05d}"
        for j, i in enumerate(ids):
            r = pool.df.iloc[i]
            tl = type_level.setdefault(r.type, float(rng.lognormal(0, 0.6)))
            cell = tl * float(rng.lognormal(0, 0.35))
            levels = [round(cell * float(rng.lognormal(0, 0.25)), 4) if on else 0.0 for on in tags[j]]
            neu_rows.append(dict(image_id=img_id, instance=j + 1, bodyId=int(r.bodyId), type=r.type,
                                 superclass=r.superclass, itoleeHl=r.itoleeHl, trumanHl=r.trumanHl,
                                 somaSide=r.somaSide, split=r.split,
                                 cable_in_crop_um=int(counts[i] * cfg.footprint_um),
                                 tags="+".join(t for t, on in zip(cfg.tags, tags[j]) if on),
                                 level_HA=levels[0], level_V5=levels[1], level_FLAG=levels[2],
                                 overlap_degree=int(deg[j])))
        img_rows.append(dict(
            image_id=img_id, split=split_of_image[k], overlap_mode=mode,
            count_bin=f"{lo_n}-{hi_n}", n_target=n_target, n_neurons=len(ids),
            n_candidates=int(len(cand)), n_types=len(set(pool.df.type.values[ids])) if ids else 0,
            body_ids=" ".join(str(int(pool.df.bodyId.values[i])) for i in ids),
            crop_lo_um=json.dumps(np.round(lo_um, 2).tolist()), crop_hi_um=json.dumps(np.round(hi_um, 2).tolist()),
            q_excision=round(q, 3), forced_same_colour_pair=forced,
            expression_set_size=info.get("expression_set_size"),
            expressed_types=";".join(info.get("expressed_types", [])),
            render_seed=int(rng.integers(2 ** 31)),
            scale=json.dumps(np.round(1 + rng.uniform(-cfg.scale_jitter, cfg.scale_jitter, 3), 4).tolist()),
            rot_deg=json.dumps(np.round(rng.uniform(-cfg.max_rot_deg, cfg.max_rot_deg, 3), 3).tolist()),
            imaging=json.dumps(sample_imaging(rng, cfg.voxel_um, cal)),
            **st))
        if (k + 1) % 250 == 0:
            log(f"  designed {k + 1}/{len(plan)} images")
    os.makedirs(out, exist_ok=True)
    imgs, neus = pd.DataFrame(img_rows), pd.DataFrame(neu_rows)
    imgs.to_parquet(os.path.join(out, "manifest_images.parquet"), index=False)
    neus.to_parquet(os.path.join(out, "manifest_neurons.parquet"), index=False)
    imgs.drop(columns=["imaging"]).to_csv(os.path.join(out, "manifest_images.csv"), index=False)
    neus.to_csv(os.path.join(out, "manifest_neurons.csv"), index=False)
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(cfg.to_json())
    log(f"manifest: {len(imgs)} images, {len(neus)} neuron instances -> {out}")
    return imgs, neus
