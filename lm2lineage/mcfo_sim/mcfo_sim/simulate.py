"""Generate one synthetic MCFO VNC stack with instance ground truth.

Output (FISBe-compatible zarr, dims CZYX):
    raw           uint16 (3, Z, Y, X)   three MCFO channels (HA, V5, FLAG)
    gt_instances  uint8  (N, Z, Y, X)   one binary mask per neuron (overlaps allowed)
    gt_labels     uint16 (Z, Y, X)      single label map (brightest wins overlaps)
plus <name>.json (all parameters + per-neuron bodyId/type/hemilineage/tags),
<name>_skeletons.npz (centreline voxel coords per instance) and <name>_mip.png.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .data import VNC_SUPERCLASSES, load_annotations, load_skeleton
from .geometry import Placement, region_box_em
from .imaging import ImagingParams, build_tissue
from .scene import render_scene
from .labeling import TAGS, mcfo_label, virtual_line


def _peak_norm(sig_vox):
    """Weight that makes a blurred point source peak at 1."""
    return float((2 * np.pi) ** 1.5 * np.prod(sig_vox))


def simulate_sample(annot, out_dir, name, seed, cache_dir, neuromeres=("T2",),
                    n_types=(3, 20), target_visible=(4, 25), voxel=0.44,
                    randomize=True, vis_snr=2.0, bead_prob=0.6, bead_per_um=0.25,
                    placement_kwargs=None, verbose=True):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    p = ImagingParams.random(rng, voxel) if randomize else ImagingParams(voxel_um=voxel)
    lo_em, hi_em = region_box_em(annot, neuromeres)
    place = Placement.random(lo_em, hi_em, rng, **(placement_kwargs or {}))
    shape = tuple(np.ceil(place.shape_um / voxel).astype(int)[::-1])  # Z, Y, X
    shape_um_xyz = np.array(shape[::-1]) * voxel
    log = (lambda *a: print(f"[{name} {time.time() - t0:6.1f}s]", *a, flush=True)) \
        if verbose else (lambda *a: None)
    log("volume ZYX", shape, "voxel", voxel)

    # Tissue background from every VNC soma in the connectome.
    somas = annot[annot.superclass.isin(VNC_SUPERCLASSES)].dropna(subset=["soma_x_um"])
    soma_img = place(somas[["soma_x_um", "soma_y_um", "soma_z_um"]].values)
    keep = np.all((soma_img >= 0) & (soma_img < shape_um_xyz), axis=1)
    tissue = build_tissue(soma_img[keep], shape, voxel, rng, p)

    # Virtual GAL4 line + MCFO labelling.
    k = int(rng.integers(n_types[0], n_types[1] + 1))
    lab = mcfo_label(virtual_line(annot, neuromeres, k, rng), rng, target_visible)
    log(f"{k} cell types -> {len(lab)} MCFO-labelled neurons")

    def neurons():
        for row in lab.itertuples(index=False):
            soma = np.array([row.soma_x_um, row.soma_y_um, row.soma_z_um], float)
            sk = load_skeleton(row.bodyId, cache_dir, soma_um=soma)
            yield dict(xyz=place(sk.xyz), radius=sk.radius * place.radius_scale, edges=sk.edges,
                       soma_idx=sk.soma_idx,
                       levels=[getattr(row, f"level_{t}") for t in TAGS],
                       meta=dict(bodyId=int(row.bodyId), type=row.type, instance=row.instance,
                                 superclass=row.superclass, hemilineage=_s(row.hemilineage),
                                 trumanHl=_s(row.trumanHl), itoleeHl=_s(row.itoleeHl),
                                 somaSide=_s(row.somaSide), somaNeuromere=_s(row.somaNeuromere)))

    meta = render_scene(neurons(), tissue, p, shape, rng, out_dir, name,
                        extra_meta=dict(seed=seed, neuromeres=list(neuromeres), n_cell_types=k,
                                        source="male-cns v1.0 (JRCFIB2022M) VNC",
                                        placement=dict(center_em_um=place.center_em.tolist(),
                                                       scale=place.scale.tolist(),
                                                       rot=place.rot.tolist()),
                                        q_excision=float(lab.q_excision.iloc[0]) if len(lab) else None),
                        vis_snr=vis_snr, bead_prob=bead_prob, bead_per_um=bead_per_um,
                        log=log if verbose else None)
    log(f"done: {len(meta['instances'])} GT instances -> {out_dir}/{name}.zarr")
    return meta


def _s(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--n", type=int, default=1, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neuromeres", default="T2", help="comma list, e.g. T1,T2")
    ap.add_argument("--voxel", type=float, default=0.44)
    ap.add_argument("--types", default="3,20", help="min,max cell types per line")
    ap.add_argument("--visible", default="4,25", help="min,max labelled neurons")
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--inplane-rot", type=float, default=0.0,
                    help="max in-plane rotation (deg); ~180 mimics unaligned tiles")
    a = ap.parse_args()
    annot = load_annotations(a.cache)
    for i in range(a.n):
        simulate_sample(annot, a.out, f"synth_{a.seed + i:05d}", a.seed + i, a.cache,
                        neuromeres=tuple(a.neuromeres.split(",")),
                        n_types=tuple(map(int, a.types.split(","))),
                        target_visible=tuple(map(int, a.visible.split(","))),
                        voxel=a.voxel, randomize=not a.no_randomize,
                        placement_kwargs=dict(inplane_rot_deg=a.inplane_rot))


if __name__ == "__main__":
    main()
