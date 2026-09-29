"""Known-identity evaluation set from published split-GAL4 lines (option 2).

Input: split/split_lines_hemilineage_avail.csv, built from the split-GAL4 line
resource (Meissner et al. 2025, eLife 98405, Figure 1-source data 1): every
adult line whose listed cell types resolve to a single VNC hemilineage, via
(in priority order) the listed MANC body ids, MANC type names, or the
hemilineage written in the paper's own cell-type label.

For each line, one aligned VNC stack (JRC2018_VNC_Unisex_40x_DS) is fetched
from FlyLight/NeuronBridge and a foreground mask is made:
  * lines with MANC body ids: EM-guided proposals for each listed body (this
    also checks the published identity against the image: 'em_coverage');
  * other lines: robust threshold of the signal, small components removed.
The mask is labelled with the line's hemilineage (all labelled cells share it).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .nb import NB
from .propose import Params, propose


def threshold_mask(stack, kinds, z=12.0, min_vox=200, rel=2.5):
    """Foreground of thin labelled structures above smooth tissue haze.

    Local background = heavy Gaussian blur; foreground = residual above z robust
    noise units AND at least `rel` x the local background (so uniformly bright
    tissue haze is not picked up).
    """
    sig = [i for i, k in enumerate(kinds) if k == "signal"]
    m = np.zeros(stack.shape[1:], bool)
    for c in sig:
        v = ndi.gaussian_filter(stack[c].astype(np.float32), (0.5, 1, 1))
        b = ndi.gaussian_filter(v[::2, ::4, ::4], (3, 4, 4))
        b = ndi.zoom(b, np.array(v.shape) / np.array(b.shape), order=1)[:v.shape[0], :v.shape[1], :v.shape[2]]
        if b.shape != v.shape:
            b = np.pad(b, [(0, s1 - s0) for s0, s1 in zip(b.shape, v.shape)], mode="edge")
        r = v - b
        tissue = b > np.percentile(b, 50)
        rs = r[tissue][::7]
        noise = max(1.4826 * np.median(np.abs(rs - np.median(rs))), 1.0)
        m |= (r > z * noise) & (v > rel * (b + 1))
        del v, b, r
    lab, n = ndi.label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_vox
        keep[0] = False
        m = keep[lab]
    return m


def qc(stack, kinds, mask, title, skel=None):
    sig = [i for i, k in enumerate(kinds) if k == "signal"]
    mip = np.stack([stack[c].max(0).astype(np.float32) for c in sig])
    for i in range(len(mip)):
        mip[i] /= np.percentile(mip[i], 99.8) + 1e-6
    mip = np.clip(mip, 0, 1) ** 0.7
    if len(sig) == 1:
        rgb = np.repeat(mip[:1], 3, 0)
    else:
        rgb = np.concatenate([mip[:3], np.zeros((max(0, 3 - len(mip)),) + mip.shape[1:])])
    rgb = np.moveaxis(rgb, 0, -1)
    mm = mask.max(0)
    ov = rgb * 0.5
    ov[mm] = ov[mm] * 0.3 + np.array([1.0, 0.25, 0.8]) * 0.7
    if skel is not None:
        for (y, x) in skel:
            ov[y, x] = [1, 1, 1]
    panel = np.concatenate([rgb, np.ones((rgb.shape[0], 4, 3)), ov], axis=1)
    img = Image.fromarray((panel * 255).astype(np.uint8))
    img = img.resize((img.width // 2, img.height // 2))
    out = Image.new("RGB", (max(img.width, 500), img.height + 34), (20, 20, 20))
    out.paste(img, (0, 34))
    d = ImageDraw.Draw(out)
    for i, t in enumerate(title):
        d.text((4, 3 + 14 * i), t, fill=(255, 230, 120))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="split/split_lines_hemilineage_avail.csv")
    ap.add_argument("--out", default="splitset")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--per-hemilineage", type=int, default=0, help="0 = all lines")
    ap.add_argument("--max-bodies", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=3, help="slides tried per EM line")
    a = ap.parse_args()
    nb = NB(a.cache)
    t = pd.read_csv(a.table)
    t = t[t.single_hemilineage & (t.n_vnc_stacks > 0)]
    if a.per_hemilineage:
        t = (t.sort_values(["label_source", "quality"])
             .groupby("hemilineage", group_keys=False).head(a.per_hemilineage))
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "review"), exist_ok=True)
    rows, t0 = [], time.time()
    csv_fp = os.path.join(a.out, "splitset.csv")
    if os.path.exists(csv_fp):                     # resume
        rows = pd.read_csv(csv_fp).to_dict("records")
    done = {x["line"] for x in rows}
    for _, r in t.iterrows():
        if r.line in done:
            continue
        recs = [x for x in nb.meta(f"by_line/{r.line}.json")["results"]
                if x["anatomicalArea"] == "VNC" and x["files"].get("VisuallyLosslessStack")]
        recs = sorted({x["slideCode"]: x for x in recs}.values(), key=lambda x: x["slideCode"])
        ids = [int(i) for i in str(r.em_ids).split()] if isinstance(r.em_ids, str) else []
        swcs = []
        for bid in ids[:a.max_bodies]:
            em = [e for e in nb.em_images(bid) if e["libraryName"] == "FlyEM_MANC_v1.2.1"]
            if em and em[0]["files"].get("AlignedBodySWC"):
                swcs.append((bid, nb.swc(em[0]["files"]["AlignedBodySWC"])))
        best = None
        # EM lines: try up to --max-samples slides, keep the one that best shows the
        # published neurons; threshold lines: first slide.
        for rec in recs[:a.max_samples if swcs else 1]:
            try:
                stack, kinds, vox, space = nb.stack(rec)
            except Exception as e:  # noqa: BLE001
                print("skip", r.line, rec["slideCode"], e)
                continue
            mask = np.zeros(stack.shape[1:], bool)
            covs, skel_yx = [], []
            for bid, swc in swcs:
                # both body orientations: split lines label left and right copies
                for mir in (False, True):
                    p = propose(stack, kinds, vox, swc, p=Params(z_thresh=6.0), orientations=(mir,))
                    if p is None:
                        continue
                    covs.append(dict(body=bid, mirrored=mir, coverage=round(p.coverage, 3),
                                     presence=round(p.presence, 3)))
                    if p.presence >= 0.7 or not mir:
                        lo = p.offset
                        mask[lo[0]:lo[0] + p.mask.shape[0], lo[1]:lo[1] + p.mask.shape[1],
                             lo[2]:lo[2] + p.mask.shape[2]] |= p.mask
                        iv = p.skel_vox * [1, p.bin_xy, p.bin_xy] + lo
                        skel_yx += [tuple(v) for v in np.unique(iv[:, 1:], axis=0)]
            both = [max(c["presence"] for c in covs if c["body"] == b) for b in {c["body"] for c in covs}]
            score = float(np.mean(both)) if both else -1
            if best is None or score > best[0]:
                best = (score, rec, stack, kinds, vox, mask, covs, skel_yx)
            else:
                del stack
            if score >= 0.7:
                break
        if best is None:
            continue
        _, rec, stack, kinds, vox, mask, covs, skel_yx = best
        method = "em_guided" if covs else "threshold"
        if method == "threshold":
            mask = threshold_mask(stack, kinds)
        per_body = {}
        for c in covs:
            if c["body"] not in per_body or c["presence"] > per_body[c["body"]]["presence"]:
                per_body[c["body"]] = c
        em_cov = float(np.mean([c["coverage"] for c in per_body.values()])) if covs else None
        em_pres = float(np.mean([c["presence"] for c in per_body.values()])) if covs else None
        sid = f"{r.line}__{rec['slideCode']}"
        row = dict(sample_id=sid, line=r.line, slide=rec["slideCode"], objective=rec["objective"],
                   sex=rec.get("gender"), hemilineage=r.hemilineage, label_source=r.label_source,
                   cell_types=r.cell_types, superclass=r.superclass, reference=r.ref,
                   mask_method=method, mask_voxels=int(mask.sum()), em_bodies=len(per_body),
                   samples_tried=min(len(recs), a.max_samples if covs else 1),
                   em_coverage=None if em_cov is None else round(em_cov, 3),
                   em_presence=None if em_pres is None else round(em_pres, 3),
                   identity_check=("confirmed" if em_pres is not None and em_pres >= 0.7 else
                                   "doubtful" if em_pres is not None else "not_checked"),
                   stack=rec["files"]["VisuallyLosslessStack"])
        rows.append(row)
        np.savez_compressed(os.path.join(a.out, "masks", sid + ".npz"),
                            mask_bits=np.packbits(mask), shape=mask.shape, voxel_xyz=vox,
                            meta=json.dumps(row), em=json.dumps(covs))
        yx = [(y, x) for (y, x) in skel_yx if y < mask.shape[1] and x < mask.shape[2]]
        qc(stack, kinds, mask, [f"{r.line} {rec['slideCode']}  hemilineage {r.hemilineage} "
                                f"({r.label_source})  {str(r.cell_types)[:60]}",
                                f"mask: {method}  EM coverage {row['em_coverage']}  presence "
                                f"{row['em_presence']}  -> {row['identity_check']}"],
           yx if yx else None).save(os.path.join(a.out, "review", sid + ".png"))
        print(f"[{time.time() - t0:6.0f}s] {sid} {r.hemilineage} {method} "
              f"cov={row['em_coverage']} pres={row['em_presence']} {row['identity_check']}", flush=True)
        del stack
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "splitset.csv"), index=False)
    print(f"wrote {len(rows)} samples")


if __name__ == "__main__":
    main()
