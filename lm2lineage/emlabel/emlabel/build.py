"""Build a reviewable set of EM-guided mask proposals.

Two entry points:
  by-hemilineage : pick MANC neurons per hemilineage, take their top PPPM
                   matches in Gen1 MCFO VNC samples, propose masks.
  by-sample      : for one of *your* Gen1 MCFO VNC samples (line + slide code),
                   take the top colour-depth-search EM candidates, propose masks.

Outputs (in --out):
  proposals.csv            one row per (EM neuron, MCFO sample) with scores
  masks/<id>.npz           mask (packed bits) + crop offset in stack voxels
  review/<id>.png          raw | raw + EM skeleton | proposal
  review_sheet_*.png       contact sheets
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd

from .nb import NB
from .propose import Params, propose
from .qc import contact_sheet, review_image

ANNOT_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
             "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather")


def manc_hemilineage_table(cache):
    """MANC bodyId -> type / hemilineage via the male CNS cross-matching."""
    fp = os.path.join(cache, "male_cns_annotations.feather")
    if not os.path.exists(fp):
        import urllib.request
        urllib.request.urlretrieve(ANNOT_URL, fp)
    df = pd.read_feather(fp)
    m = df[df.mancBodyid.notna()].copy()
    m["mancBodyid"] = m.mancBodyid.astype(np.int64)
    m = m.drop_duplicates("mancBodyid")
    return m[["mancBodyid", "mancType", "trumanHl", "itoleeHl", "superclass",
              "somaNeuromere", "somaSide", "bodyId"]].rename(columns={"bodyId": "mcnsBodyId"})


def auto_flag(s):
    ok = s["coverage"] >= 0.6 and s["precision"] >= 0.6 and s["presence"] >= 0.7
    weak = s["coverage"] < 0.3 or s["presence"] < 0.4
    return "likely_good" if ok else ("likely_bad" if weak else "check")


def run_pair(nb, match, info, out, params, stacks_cache):
    key = match["lm"]["files"]["VisuallyLosslessStack"]
    if key not in stacks_cache:
        stacks_cache.clear()                      # keep one stack in memory
        stacks_cache[key] = nb.stack(match["lm"])
    stack, kinds, vox, space = stacks_cache[key]
    swc = nb.swc(match["em_swc"])
    prop = propose(stack, kinds, vox, swc, mirrored_hint=match["mirrored"], p=params)
    if prop is None:
        return None
    lm = match["lm"]
    pid = f"{info['mancBodyid']}__{lm['publishedName']}__{lm['slideCode']}"
    s = prop.summary()
    row = dict(proposal_id=pid, manc_body=info["mancBodyid"], type=info.get("mancType"),
               hemilineage=info.get("trumanHl"), superclass=info.get("superclass"),
               line=lm["publishedName"], slide=lm["slideCode"], sex=lm.get("gender"),
               method=match["method"], rank=match.get("rank"), score=match.get("score"),
               mirrored_meta=match["mirrored"], **{k: s[k] for k in
               ("mirrored", "coverage", "precision", "presence", "mask_voxels")},
               channels_on="+".join(map(str, s["channels_on"])),
               stack=key, em_swc=match["em_swc"])
    row["auto_flag"] = auto_flag(row)
    row["human_verdict"] = ""
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out, "review"), exist_ok=True)
    np.savez_compressed(os.path.join(out, "masks", pid + ".npz"),
                        mask_bits=np.packbits(prop.mask), shape=prop.mask.shape,
                        offset_zyx=prop.offset, stack_shape=stack.shape[1:],
                        voxel_xyz=vox, meta=json.dumps(row))
    img = review_image(prop, [
        f"MANC {info['mancBodyid']}  {info.get('mancType')}  hemilineage {info.get('trumanHl')}",
        f"{lm['publishedName']} {lm['slideCode']}  {match['method']} rank {match.get('rank')} "
        f"score {match.get('score')}",
        f"coverage {row['coverage']:.2f}  precision {row['precision']:.2f}  presence "
        f"{row['presence']:.2f}  -> {row['auto_flag']}"])
    img.save(os.path.join(out, "review", pid + ".png"))
    return row, img


def by_hemilineage(args):
    nb = NB(args.cache)
    table = manc_hemilineage_table(args.cache)
    clean = table[table.trumanHl.notna() & table.trumanHl.str.fullmatch(r"\d{2}[AB]")
                  & table.superclass.isin(["vnc_intrinsic", "ascending_neuron", "vnc_motor"])]
    hls = args.hemilineages.split(",") if args.hemilineages else sorted(clean.trumanHl.unique())
    rng = np.random.default_rng(args.seed)
    rows, imgs, cache = [], [], {}
    t0 = time.time()
    for hl in hls:
        pool = clean[clean.trumanHl == hl].sample(frac=1, random_state=int(rng.integers(1e9)))
        got = 0
        for _, info in pool.iterrows():
            if got >= args.per_hemilineage:
                break
            matches = [m for m in nb.pppm_matches(info.mancBodyid) if m["rank"] is not None]
            matches = sorted(matches, key=lambda m: m["rank"])[:args.top]
            if not matches:
                continue
            got += 1
            for m in matches:
                r = run_pair(nb, m, info.to_dict(), args.out, Params(), cache)
                if r:
                    rows.append(r[0]); imgs.append(r[1])
                    print(f"[{time.time() - t0:6.0f}s] {hl} {r[0]['proposal_id']} "
                          f"cov={r[0]['coverage']:.2f} prec={r[0]['precision']:.2f} "
                          f"-> {r[0]['auto_flag']}", flush=True)
    _write(args.out, rows, imgs)


def by_sample(args):
    nb = NB(args.cache)
    table = manc_hemilineage_table(args.cache).set_index("mancBodyid")
    ms = nb.cds_matches_lm(args.line, args.slide)
    ms = [m for m in ms if m["em_library"].startswith("FlyEM_MANC") and m["em_swc"]]
    ms = sorted(ms, key=lambda m: -m["score"])
    seen, rows, imgs, cache = set(), [], [], {}
    for m in ms:
        bid = int(m["em_name"].split(":")[-1])
        if bid in seen:
            continue
        seen.add(bid)
        info = table.loc[bid].to_dict() if bid in table.index else {}
        info["mancBodyid"] = bid
        info.setdefault("mancType", m["em_type"])
        r = run_pair(nb, m, info, args.out, Params(), cache)
        if r:
            rows.append(r[0]); imgs.append(r[1])
            print(f"{r[0]['proposal_id']} {info.get('mancType')} {info.get('trumanHl')} "
                  f"cov={r[0]['coverage']:.2f} prec={r[0]['precision']:.2f} -> {r[0]['auto_flag']}")
        if len(seen) >= args.top:
            break
    _write(args.out, rows, imgs)


def _write(out, rows, imgs):
    if not rows:
        print("no proposals")
        return
    with open(os.path.join(out, "proposals.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for i in range(0, len(imgs), 12):
        contact_sheet(imgs[i:i + 12], cols=3).save(
            os.path.join(out, f"review_sheet_{i // 12 + 1:02d}.png"))
    print(f"wrote {len(rows)} proposals to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("by-hemilineage")
    a.add_argument("--hemilineages", default="", help="e.g. 03A,06B,19A (default: all)")
    a.add_argument("--per-hemilineage", type=int, default=1, help="EM neurons per hemilineage")
    a.add_argument("--top", type=int, default=2, help="PPPM matches per EM neuron")
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("by-sample")
    b.add_argument("--line", required=True)
    b.add_argument("--slide", required=True)
    b.add_argument("--top", type=int, default=10, help="EM candidates to try")
    for p in (a, b):
        p.add_argument("--out", default="proposals")
        p.add_argument("--cache", default="cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (by_hemilineage if args.cmd == "by-hemilineage" else by_sample)(args)


if __name__ == "__main__":
    main()
