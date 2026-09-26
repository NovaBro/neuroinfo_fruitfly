"""Validation: integrity, FISBe calibration, realism comparison, two-sample test.

All functions read FISBe-format zarr directories (raw CZYX + gt_instances NZYX),
so they work identically on FISBe itself and on the synthetic output.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, wasserstein_distance

from ..labeling import TAGS
from ..scene import open_sample


# ------------------------------------------------------------------ integrity
def integrity(rendered_dir, manifest_dir=None):
    man = None
    if manifest_dir:
        neus = pd.read_parquet(os.path.join(manifest_dir, "manifest_neurons.parquet"))
        man = neus.groupby("image_id").bodyId.apply(set).to_dict()
    rows = []
    for zp in sorted(glob.glob(os.path.join(rendered_dir, "*.zarr"))):
        meta = json.load(open(zp[:-5] + ".json"))
        raw, gt, lab = open_sample(zp)
        lab = lab[:] if lab is not None else np.zeros(1, int)
        n = len(meta["instances"])
        img = raw[:].astype(np.float32) - meta["imaging"]["offset_adu"]
        colour_ok, masks = 0, [gt[i][:].astype(bool) for i in range(n)]
        cnt = np.sum(masks, axis=0) if n else None
        for i, ins in enumerate(meta["instances"]):
            m = masks[i] & (cnt == 1)          # voxels not shared with another neuron
            if m.sum() < 20:
                colour_ok += 1                  # nothing exclusive to test
                continue
            on = [TAGS.index(t) for t in ins["tags"]]
            off = [c for c in range(3) if c not in on]
            mean = img[:, m].mean(1)
            colour_ok += (not off) or mean[on].min() > mean[off].max()
        rendered = {ins["bodyId"] for ins in meta["instances"]}
        missing = {d["bodyId"] for d in meta.get("not_rendered", [])}
        row = dict(image_id=meta.get("image_id", meta["name"]), n_rendered=n,
                   n_not_rendered=len(missing), gt_channels_match=(n == 0 or gt.shape[0] == n),
                   labels_ok=int(lab.max()) <= n, colour_consistent=f"{colour_ok}/{n}")
        if man is not None:
            planned = man.get(row["image_id"], set())
            row["ids_match_manifest"] = (rendered | missing) == planned
        row["pass"] = bool(row["gt_channels_match"] and row["labels_ok"] and colour_ok == n
                           and row.get("ids_match_manifest", True))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rendered_dir, "integrity.csv"), index=False)
    return df


# ------------------------------------------------------------------ features
def image_features(zarr_path, max_instances=60, stride=2):
    raw, gt, _ = open_sample(zarr_path)
    r = raw[:, ::stride, ::stride, ::stride].astype(np.float32)
    # effective bit depth (12-bit data is usually stored as uint16)
    obs = float(r.max())
    dmax = next((m for m in (255.0, 4095.0, 65535.0) if obs <= m), obs)
    n = gt.shape[0]
    union = np.zeros(r.shape[1:], bool)
    inst = []
    for i in range(min(n, max_instances)):
        m = gt[i][:].astype(bool)
        if not m.any():
            continue
        union |= m[::stride, ::stride, ::stride]
        inst.append(_instance_features(raw, m))
    mx = r.max(0)
    dark = [float(np.percentile(r[c], 1)) for c in range(r.shape[0])]
    lowmask = ~ndi.binary_dilation(union, iterations=3)
    low = [r[c][lowmask & (r[c] <= np.percentile(r[c][lowmask], 50))] for c in range(r.shape[0])]
    # robust sd; integer data with a flat dark level can have MAD == 0, fall back to sd
    dark_sd = [float(max(1.4826 * np.median(np.abs(v - np.median(v))), np.std(v), 0.5)) if v.size else np.nan
               for v in low]
    bg = mx[lowmask]
    fg = mx[union]
    hp = mx - ndi.gaussian_filter(mx, 2)
    f = dict(path=zarr_path, dtype_max=dmax, n_instances=int(n),
             dark_level=float(np.mean(dark)), dark_sd=float(np.nanmean(dark_sd)),
             bg_median=float(np.median(bg)) if bg.size else np.nan,
             fg_median=float(np.median(fg)) if fg.size else np.nan,
             fg_p99=float(np.percentile(fg, 99)) if fg.size else np.nan,
             fg_fraction=float(union.mean()),
             bg_texture=float(np.std(hp[lowmask])) if lowmask.any() else np.nan)
    d = f["dark_level"]
    f["fg_p99_rel"] = (f["fg_p99"] - d) / dmax
    f["contrast"] = (f["fg_median"] - f["bg_median"]) / max(f["dark_sd"], 1e-3)
    f["bg_to_fg"] = (f["bg_median"] - d) / max(f["fg_median"] - d, 1e-3)
    f["texture_rel"] = f["bg_texture"] / max(f["dark_sd"], 1e-3)
    return f, inst


def _instance_features(raw, m):
    from skimage.morphology import skeletonize
    zz, yy, xx = np.nonzero(m)
    lo = np.array([zz.min(), yy.min(), xx.min()])
    hi = np.array([zz.max(), yy.max(), xx.max()]) + 1
    sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sk = skeletonize(sub)
    edt = ndi.distance_transform_edt(sub)
    vals = raw[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][:, sub].astype(np.float32)
    core = vals.max(0) >= np.percentile(vals.max(0), 80)       # brightest 20% of the neuron
    mean = vals[:, core].mean(1)
    mean = mean - mean.min()
    col = mean / (mean.max() + 1e-6)
    return dict(voxels=int(sub.sum()), skeleton_voxels=int(sk.sum()),
                diameter_vox=float(2 * np.median(edt[sk])) if sk.any() else np.nan,
                n_channels_on=int((col >= 0.35).sum()), dominant_fraction=float(col.max() / (col.sum() + 1e-6)))


def features_for_dir(d, pattern="*.zarr", limit=None):
    imgs, insts = [], []
    for zp in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True))[:limit]:
        try:
            f, inst = image_features(zp)
        except Exception as e:  # noqa: BLE001
            print("skip", zp, e)
            continue
        imgs.append(f)
        insts += [dict(i, path=zp) for i in inst]
    return pd.DataFrame(imgs), pd.DataFrame(insts)


# ------------------------------------------------------------------ calibration
def _q_from_mean_channels(m):
    """Invert E[k | k>=1] = 3q / (1 - (1-q)^3) for the per-tag excision probability q."""
    qs = np.linspace(0.01, 0.99, 981)
    e = 3 * qs / (1 - (1 - qs) ** 3)
    return float(qs[np.argmin(np.abs(e - m))])


def calibrate(fisbe_dir, out_json, limit=None):
    f, inst = features_for_dir(fisbe_dir, limit=limit)
    q = lambda s, a, b: [float(np.nanpercentile(s, a)), float(np.nanpercentile(s, b))]  # noqa: E731
    cal = dict(max_adu=int(f.dtype_max.max()),
               offset_adu=q(f.dark_level, 5, 95), read_noise_adu=q(f.dark_sd, 5, 95),
               gain_target=[max(0.02, v) for v in q(f.fg_p99_rel, 5, 95)],
               fisbe_bg_to_fg=q(f.bg_to_fg, 5, 95), fisbe_contrast=q(f.contrast, 5, 95),
               n_images=len(f))
    if len(inst):
        per_img = inst.groupby("path").n_channels_on.mean()
        cal["q_excision"] = [_q_from_mean_channels(v) for v in np.nanpercentile(per_img, [5, 95])]
        cal["fisbe_channels_on_freq"] = inst.n_channels_on.value_counts(normalize=True).round(3).to_dict()
    json.dump(cal, open(out_json, "w"), indent=1)
    return cal


# ------------------------------------------------------------------ comparison
IMG_METRICS = ["n_instances", "fg_fraction", "fg_p99_rel", "contrast", "bg_to_fg", "texture_rel", "dark_sd"]
INST_METRICS = ["voxels", "skeleton_voxels", "diameter_vox", "n_channels_on", "dominant_fraction"]


def compare(fisbe_dir, synth_dir, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    fr, fi = features_for_dir(fisbe_dir, limit=limit)
    sr, si = features_for_dir(synth_dir, limit=limit)
    for df, name in ((fr, "fisbe_images"), (fi, "fisbe_instances"), (sr, "synth_images"), (si, "synth_instances")):
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    rows = []
    for level, a, b, metrics in (("image", fr, sr, IMG_METRICS), ("instance", fi, si, INST_METRICS)):
        for m in metrics:
            x, y = a[m].dropna().values, b[m].dropna().values
            if len(x) < 3 or len(y) < 3:
                continue
            scale = np.subtract(*np.percentile(np.r_[x, y], [95, 5])) or 1.0
            ks = ks_2samp(x, y)
            rows.append(dict(level=level, metric=m, fisbe_median=np.median(x), synth_median=np.median(y),
                             ks=round(ks.statistic, 3), ks_p=ks.pvalue,
                             wasserstein_rel=round(wasserstein_distance(x, y) / scale, 3),
                             flag="mismatch" if ks.statistic > 0.3 else "ok"))
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(out_dir, "compare_report.csv"), index=False)
    sug = {}
    if "bg_to_fg" in fr and "bg_to_fg" in sr:
        sug["bg_scale"] = float(np.nanmedian(fr.bg_to_fg) / max(np.nanmedian(sr.bg_to_fg), 1e-6))
    json.dump(dict(suggestions=sug), open(os.path.join(out_dir, "suggestions.json"), "w"), indent=1)
    return rep, sug


# ------------------------------------------------------------------ two-sample test
def _patch_features(raw, z, y, x, size=(24, 48, 48)):
    sl = tuple(slice(max(0, c - s // 2), max(0, c - s // 2) + s) for c, s in zip((z, y, x), size))
    p = raw[(slice(None),) + sl].astype(np.float32)
    p /= np.percentile(p, 99.9) + 1e-6
    feats = []
    for c in range(p.shape[0]):
        v = p[c]
        g = np.sqrt(sum(np.gradient(v, axis=a) ** 2 for a in range(3)))
        feats += list(np.percentile(v, [5, 25, 50, 75, 95, 99])) + list(np.percentile(g, [50, 90, 99]))
        feats.append(float(ndi.laplace(v).std()))
    mx = p.max(0).ravel()
    for a in range(p.shape[0]):
        for b in range(a + 1, p.shape[0]):
            feats.append(float(np.corrcoef(p[a].ravel(), p[b].ravel())[0, 1]) if mx.std() > 0 else 0.0)
    return np.nan_to_num(np.array(feats, np.float32))


def c2st(fisbe_dir, synth_dir, patches_per_image=40, limit=None, seed=0):
    """Classifier two-sample test on random patches (grouped CV by image).
    AUC ~0.5: these features cannot tell synthetic from real; AUC ~1: easy to tell."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for label, d in ((0, fisbe_dir), (1, synth_dir)):
        for gi, zp in enumerate(sorted(glob.glob(os.path.join(d, "**", "*.zarr"), recursive=True))[:limit]):
            raw, gt, _ = open_sample(zp)
            Z, Y, Xs = raw.shape[1:]
            fgpts = None
            if gt.shape[0]:
                m = gt[0][::4, ::4, ::4]
                nz = np.argwhere(m)
                fgpts = nz * 4 if len(nz) else None
            for k in range(patches_per_image):
                if fgpts is not None and k % 2 == 0:
                    z, yy, x = fgpts[rng.integers(len(fgpts))]
                else:
                    z, yy, x = rng.integers(0, Z), rng.integers(0, Y), rng.integers(0, Xs)
                X.append(_patch_features(raw, z, yy, x))
                y.append(label)
                g.append(f"{label}_{gi}")
    X, y, g = np.stack(X), np.array(y), np.array(g)
    n_groups = min(len(set(g[y == 0])), len(set(g[y == 1])))
    if n_groups < 2:
        raise ValueError("c2st needs at least 2 images per set")
    aucs = []
    for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(X, y, g):
        clf = HistGradientBoostingClassifier(max_iter=200).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return dict(auc_mean=float(np.mean(aucs)), auc_folds=[round(a, 3) for a in aucs], n_patches=len(y))
