"""Sanity checks on a generated sample: GT <-> image <-> metadata consistency."""
from __future__ import annotations

import json
import sys

import numpy as np

from .labeling import TAGS
from .scene import open_sample


def check(zarr_path: str) -> dict:
    meta = json.load(open(zarr_path.replace(".zarr", ".json")))
    raw, gt, labels = open_sample(zarr_path)
    labels = labels[:]
    n = len(meta["instances"])
    out = dict(sample=meta["name"], n_instances=n, gt_channels=gt.shape[0],
               label_max=int(labels.max()))
    assert n == 0 or gt.shape[0] == n, "gt_instances count != metadata"
    assert labels.max() <= n, "label ids exceed instance count"
    img = raw[:].astype(np.float32) - meta["imaging"]["offset_adu"]
    bg = np.median(img[:, labels == 0], axis=1)
    union = np.zeros(labels.shape, np.uint8)
    color_ok, contrast = 0, []
    for i, ins in enumerate(meta["instances"]):
        m = gt[i][:].astype(bool)
        union += m
        on = [TAGS.index(t) for t in ins["tags"]]
        off = [c for c in range(3) if c not in on]
        mean = img[:, m].mean(axis=1)
        if not off or mean[on].min() > mean[off].max():
            color_ok += 1
        contrast.append(float(np.median(img[on][:, m].max(0)) / (bg[on].max() + 1.0)))
    fg = union > 0
    out.update(
        colour_consistent=f"{color_ok}/{n}",
        median_contrast_fg_over_bg=round(float(np.median(contrast)), 1) if contrast else None,
        min_contrast=round(float(np.min(contrast)), 2) if contrast else None,
        overlap_fraction_of_fg=round(float((union > 1).sum() / max(fg.sum(), 1)), 4),
        fg_fraction=round(float(fg.mean()), 5),
    )
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(check(p)))
