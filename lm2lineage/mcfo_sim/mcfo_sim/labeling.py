"""Virtual GAL4 lines and the MultiColor FlpOut (MCFO) labelling model.

A GAL4 line expresses in a handful of cell types, usually in every copy of
each type (left/right, sometimes serial homologues). MCFO then labels a random
subset: each of the three FRT-stop reporters (HA, V5, FLAG in MCFO-1) is
excised independently in each expressing cell, so a cell ends up with one of
seven tag combinations, or is unlabelled. Expression strength varies by line /
cell type and between cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates

from .data import VNC_SUPERCLASSES

TAGS = ("HA", "V5", "FLAG")


def virtual_line(annot: pd.DataFrame, neuromeres, n_types: int, rng,
                 superclasses=VNC_SUPERCLASSES, p_descending=0.25,
                 exclude_sensory=True) -> pd.DataFrame:
    """Pick `n_types` cell types and return every neuron of those types."""
    sc = [s for s in superclasses if not (exclude_sensory and "sensory" in s)]
    pool = annot[annot.superclass.isin(sc) & annot.type.notna()
                 & (annot.status == "Traced")]
    local = pool[pool.somaNeuromere.isin(list(neuromeres))]
    local_types = local.type.unique()
    dn_types = pool[pool.superclass == "descending_neuron"].type.unique()
    chosen = []
    for _ in range(n_types):
        src = dn_types if (rng.random() < p_descending and len(dn_types)) else local_types
        chosen.append(rng.choice(src))
    chosen = list(dict.fromkeys(chosen))
    return pool[pool.type.isin(chosen)].copy()


def mcfo_label(expr: pd.DataFrame, rng, target_visible=(4, 15),
               type_sigma=0.6, cell_sigma=0.35, tag_sigma=0.25,
               q_max=0.45) -> pd.DataFrame:
    """Assign stochastic MCFO tag combinations and expression levels.

    The per-reporter excision probability q is set so the expected number of
    visibly labelled cells matches a draw from `target_visible`:
        P(visible) = 1 - (1 - q)^3
    """
    n = len(expr)
    tgt = min(rng.integers(target_visible[0], target_visible[1] + 1), n)
    q = 1 - (1 - tgt / max(n, 1)) ** (1 / 3)
    q = min(q, q_max)  # small expression sets: keep colours mixed, accept fewer cells
    tags = rng.random((n, len(TAGS))) < q
    type_level = {t: rng.lognormal(0, type_sigma) for t in expr.type.unique()}
    base = np.array([type_level[t] for t in expr.type]) * rng.lognormal(0, cell_sigma, n)
    levels = base[:, None] * rng.lognormal(0, tag_sigma, (n, len(TAGS))) * tags
    out = expr.assign(**{f"tag_{t}": tags[:, i] for i, t in enumerate(TAGS)},
                      **{f"level_{t}": levels[:, i] for i, t in enumerate(TAGS)})
    out["q_excision"] = q
    return out[tags.any(axis=1)].reset_index(drop=True)


def smooth_noise(points_um, spacing_um, rng, smooth=1.0):
    """Unit-variance smooth random field evaluated at points."""
    lo = points_um.min(0) - 2 * spacing_um
    shape = np.ceil((points_um.max(0) - lo) / spacing_um).astype(int) + 3
    g = gaussian_filter(rng.normal(size=shape), smooth)
    v = map_coordinates(g, ((points_um - lo) / spacing_um).T, order=1, mode="nearest")
    return (v - v.mean()) / (v.std() + 1e-9)


def along_cable_modulation(points_um, rng, sigma=0.35, spacing_um=4.0,
                           gap_fraction=0.04, gap_depth=0.85):
    """Multiplicative brightness variation along neurites, incl. dim gaps."""
    m = np.exp(sigma * smooth_noise(points_um, spacing_um, rng))
    if gap_fraction > 0:
        g = smooth_noise(points_um, spacing_um * 0.75, rng)
        thr = np.quantile(g, gap_fraction)
        m = m * np.where(g < thr, 1 - gap_depth, 1.0)
    return m
