"""Dataset-level statistics over the manifest (and, optionally, rendered output)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd


def _desc(s):
    s = pd.Series(s).dropna()
    if s.empty:
        return {}
    return dict(n=int(s.size), mean=round(float(s.mean()), 3), sd=round(float(s.std()), 3),
                min=float(s.min()), p25=float(s.quantile(.25)), median=float(s.median()),
                p75=float(s.quantile(.75)), max=float(s.max()))


def manifest_stats(manifest_dir, rendered_dir=None, plots=True):
    imgs = pd.read_parquet(os.path.join(manifest_dir, "manifest_images.parquet"))
    neus = pd.read_parquet(os.path.join(manifest_dir, "manifest_neurons.parquet"))
    out = dict(
        n_images=len(imgs), n_neuron_instances=len(neus),
        unique_body_ids=int(neus.bodyId.nunique()), unique_types=int(neus.type.nunique()),
        neurons_per_image=_desc(imgs.n_neurons),
        neurons_per_image_histogram={int(k): int(v) for k, v in imgs.n_neurons.value_counts().sort_index().items()},
        images_by_split=imgs.split.value_counts().to_dict(),
        images_by_overlap_mode=imgs.overlap_mode.value_counts().to_dict(),
        images_by_count_bin=imgs.count_bin.value_counts().to_dict(),
        neurons_per_image_by_mode={m: _desc(g.n_neurons) for m, g in imgs.groupby("overlap_mode")},
        shortfall_images=int((imgs.n_neurons < imgs.n_target).sum()),
        empty_images=int((imgs.n_neurons == 0).sum()),
        overlapping_pairs_per_image=_desc(imgs.overlapping_pairs),
        frac_neurons_overlapping=_desc(imgs.frac_neurons_overlapping),
        same_colour_overlapping_pairs=_desc(imgs.same_colour_overlapping_pairs),
        images_with_same_colour_overlap=int((imgs.same_colour_overlapping_pairs > 0).sum()),
        images_with_no_overlap=int((imgs.overlapping_pairs == 0).sum()),
        body_id_reuse=_desc(neus.bodyId.value_counts()),
        tag_combinations=neus.tags.value_counts().to_dict(),
        superclass_counts=neus.superclass.value_counts().to_dict(),
        top_itoleeHl=neus.itoleeHl.value_counts().head(30).to_dict(),
        split_leakage_types=int(neus.groupby("type").split.nunique().gt(1).sum()),
    )
    if rendered_dir:
        rows = []
        for fp in glob.glob(os.path.join(rendered_dir, "*.json")):
            m = json.load(open(fp))
            if "image_id" not in m:
                continue
            ov = m.get("overlap", {})
            rows.append(dict(image_id=m["image_id"], planned=m["n_neurons_planned"],
                             rendered=len(m["instances"]), not_rendered=len(m.get("not_rendered", [])),
                             **{k: ov.get(k) for k in ("pairs_touching_3d", "pairs_overlapping_mip",
                                                       "same_colour_pairs_touching_3d",
                                                       "frac_fg_voxels_shared_3d", "frac_fg_pixels_shared_mip")}))
        if rows:
            r = pd.DataFrame(rows)
            r.to_csv(os.path.join(manifest_dir, "rendered_counts.csv"), index=False)
            out["rendered"] = dict(n_images=len(r), rendered_neurons_per_image=_desc(r.rendered),
                                   fraction_planned_visible=round(float(r.rendered.sum() / max(r.planned.sum(), 1)), 4),
                                   pairs_touching_3d=_desc(r.pairs_touching_3d),
                                   pairs_overlapping_mip=_desc(r.pairs_overlapping_mip),
                                   same_colour_pairs_touching_3d=_desc(r.same_colour_pairs_touching_3d),
                                   frac_fg_voxels_shared_3d=_desc(r.frac_fg_voxels_shared_3d),
                                   images_without_3d_contact=int((r.pairs_touching_3d == 0).sum()))
    with open(os.path.join(manifest_dir, "stats.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    if plots:
        _plots(imgs, neus, manifest_dir)
    return out


def _plots(imgs, neus, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for m, g in imgs.groupby("overlap_mode"):
        ax[0].hist(g.n_neurons, bins=np.arange(0, imgs.n_neurons.max() + 2) - 0.5,
                   histtype="step", label=m)
    ax[0].set(xlabel="neurons per image", ylabel="images")
    ax[0].legend(frameon=False)
    ax[1].scatter(imgs.n_neurons, imgs.frac_neurons_overlapping, s=4, alpha=.3)
    ax[1].set(xlabel="neurons per image", ylabel="fraction of neurons overlapping another")
    reuse = neus.bodyId.value_counts()
    ax[2].hist(reuse, bins=np.arange(1, reuse.max() + 2) - 0.5)
    ax[2].set(xlabel="images a neuron appears in", ylabel="neurons", yscale="log")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stats.png"), dpi=130)
