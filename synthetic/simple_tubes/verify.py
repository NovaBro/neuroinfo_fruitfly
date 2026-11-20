"""Check simple_v1 zarrs against the invariants in plan §6.

  python -m synthetic.simple_tubes.verify fisbe/simple_v1 --meta-dir fisbe/simple_v1/meta
  python -m synthetic.simple_tubes.verify <one>.zarr --regen      # also re-render and compare

A path can be one .zarr, a split dir, or a dataset dir (searched two levels deep).
Inside a container, the dataset dir works once the source parts are mounted.

Raw vs GT: every GT voxel of neuron n has, on each of its "on" channels, raw >= its level
(haze, blobs and bright confetti only raise values). Where confetti is baked (val/test),
raw <= the dark-confetti level (1) is also allowed. With noise off, raw must equal the
per-channel max of level x GT. Also checks raw's clipmax attr, gt_numinst, gt_fg_rm_5
(prepare_fisbe_for_ppp.fg_rm_mask), the raw_normalized / gt_instances_rm_5 links and,
with --fg-dir, the fg point CSV.
GT is read one channel at a time, so 400x600x600 volumes with 30 neurons stay small.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

RAW_CHUNKS = (3, 64, 64, 64)
GT_CHUNKS = (1, 64, 64, 64)


def verify_zarr(zpath: str, meta_dir: str | None = None, regen: bool = False,
                min_voxels: int = 500, fg_dir: str | None = None) -> list[str]:
    """Return a list of problems (empty = OK)."""
    import zarr

    from .render_util import count_components

    bad = []
    root = zarr.open_group(zpath, mode="r")
    info = root.attrs.get("simple_tubes")
    if info is None:
        return [f"{zpath}: no 'simple_tubes' attrs"]
    raw_a = root["volumes/raw"]
    gt_a = root["volumes/gt_instances"]
    raw = raw_a[...]
    shape = raw.shape[1:]
    noise_on = info.get("noise") is not None
    confetti_on = noise_on and info["params"]["render"].get("confetti", True)
    dark_max = int(info["params"]["noise"]["dark_level"][1]) if confetti_on else -1
    if raw_a.attrs.get("clipmax") != 15:
        bad.append(f"raw clipmax attr {raw_a.attrs.get('clipmax')}, want 15")

    if raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[0] != 3:
        bad.append(f"raw {raw.dtype} {raw.shape}, want uint8 (3, Z, Y, X)")
    if list(shape) != list(info.get("shape", shape)):
        bad.append(f"raw shape {shape}, attrs say {info['shape']}")
    if tuple(raw_a.chunks) != RAW_CHUNKS:
        bad.append(f"raw chunks {raw_a.chunks}, want {RAW_CHUNKS}")
    if raw.max() > 15:
        bad.append(f"raw max {raw.max()} > 15")
    N = gt_a.shape[0]
    if gt_a.dtype != np.uint8 or len(gt_a.shape) != 4 or tuple(gt_a.shape[1:]) != shape or N < 1:
        bad.append(f"gt_instances {gt_a.dtype} {gt_a.shape}, want uint8 (N>=1, {shape})")
        return [f"{zpath}: {b}" for b in bad]
    if tuple(gt_a.chunks) != GT_CHUNKS:
        bad.append(f"gt chunks {gt_a.chunks}, want {GT_CHUNKS}")
    inst = info["instances"]
    if len(inst) != N:
        return [f"{zpath}: {len(inst)} instances in attrs, {N} gt channels"]

    sk = None
    if meta_dir:
        name = os.path.basename(zpath.rstrip("/"))[:-len(".zarr")]
        sk_path = os.path.join(meta_dir, name + "_skeletons.npz")
        if os.path.exists(sk_path):
            sk = np.load(sk_path)
        else:
            bad.append(f"missing {sk_path}")

    regen_sample = None
    if regen:
        from . import noise
        from .geometry import GeometryParams
        from .render import RenderParams, make_sample

        p = info["params"]
        npar = noise.NoiseParams(**p["noise"]) if p.get("noise") else None
        regen_sample = make_sample(info["index"], info["base_seed"],
                                   GeometryParams(**p["geometry"]),
                                   RenderParams(**p["render"]), npar)
        if not np.array_equal(regen_sample["raw"], raw):
            bad.append("re-rendering from the seed does not reproduce raw")
        if len(regen_sample["flats"]) != N:
            bad.append(f"re-rendering gives {len(regen_sample['flats'])} instances, not {N}")
            regen_sample = None

    raw_flat = raw.reshape(3, -1)
    # simple_v2 varies brightness per branch / along the cable: only the neuron's minimum level is
    # guaranteed on its voxels, and exact equality is left to --regen
    varied = any("level_min" in i for i in inst)
    expected = None if (noise_on or varied) else np.zeros_like(raw_flat)
    count = np.zeros(shape, dtype=np.uint8)
    for n in range(N):
        chan = gt_a[n]
        count += chan > 0
        flat = np.flatnonzero(chan)
        if np.any(chan.reshape(-1)[flat] != n + 1):
            bad.append(f"gt[{n}] has values other than 0 and {n + 1}")
        nvox = int(flat.size)
        if nvox < min_voxels:
            bad.append(f"gt[{n}] has {nvox} voxels < {min_voxels}")
        if nvox != inst[n]["voxels"]:
            bad.append(f"gt[{n}] has {nvox} voxels, attrs say {inst[n]['voxels']}")
        n_cc = count_components(flat, shape)
        if n_cc != 1:
            bad.append(f"gt[{n}] has {n_cc} 26-connected pieces, want 1")
        level = inst[n].get("level_min", inst[n]["level"])
        for ch in range(3):
            if not inst[n]["colour"][ch]:
                continue
            v = raw_flat[ch, flat]
            if noise_on:
                wrong = int(((v < level) & (v > dark_max)).sum())
                if wrong:
                    bad.append(f"gt[{n}] ch{ch}: {wrong} voxels below level {level}"
                               + (f" and above the dark-confetti level {dark_max}"
                                  if confetti_on else " (no confetti baked)"))
            elif expected is not None:
                expected[ch, flat] = np.maximum(expected[ch, flat], level)
            else:
                wrong = int((v < level).sum())
                if wrong:
                    bad.append(f"gt[{n}] ch{ch}: {wrong} voxels below its minimum level {level}")
        if sk is not None:
            key = f"instance_{n + 1:03d}"
            if key not in sk:
                bad.append(f"{key} missing from skeleton npz")
            else:
                s = sk[key].astype(np.int64)
                if len(s) == 0:
                    bad.append(f"{key} skeleton is empty")
                else:
                    out = int((chan[s[:, 0], s[:, 1], s[:, 2]] == 0).sum())
                    if out:
                        bad.append(f"{key}: {out}/{len(s)} skeleton voxels outside the mask")
        if regen_sample is not None and not np.array_equal(regen_sample["flats"][n], flat):
            bad.append(f"re-rendering does not reproduce gt[{n}]")
    if expected is not None and not np.array_equal(raw_flat, expected):
        diff = int((raw_flat != expected).sum())
        bad.append(f"raw differs from max(level x GT) per channel at {diff} voxels")
    bad += _check_ppp_arrays(root, zpath, count, fg_dir)
    return [f"{zpath}: {b}" for b in bad]


def _check_ppp_arrays(root, zpath, count, fg_dir):
    """gt_numinst, gt_fg_rm_5, alias links and fg points, as PatchPerPix reads them."""
    from .generate import ALIASES, ppp_module

    bad = []
    vol = root["volumes"]
    if "gt_numinst" not in vol or not np.array_equal(vol["gt_numinst"][...], count):
        bad.append("gt_numinst missing or not the per-voxel instance count")
    fg = ppp_module("prepare_fisbe_for_ppp").fg_rm_mask(count, opening_radius=1)
    if "gt_fg_rm_5" not in vol or not np.array_equal(vol["gt_fg_rm_5"][...], fg):
        bad.append("gt_fg_rm_5 missing or not fg_rm_mask(gt_numinst, 1)")
    for alias, target in ALIASES.items():
        link = os.path.join(zpath, "volumes", alias)
        if not os.path.islink(link) or os.readlink(link) != target:
            bad.append(f"volumes/{alias} is not a link to {target}")
    if fg_dir:
        name = os.path.basename(zpath.rstrip("/"))[:-len(".zarr")]
        csv = os.path.join(fg_dir, name + ".csv")
        if not os.path.exists(csv):
            bad.append(f"missing {csv}")
        else:
            pts = np.loadtxt(csv, dtype=np.int64, ndmin=2)
            if len(pts) == 0 or not (fg[pts[:, 0], pts[:, 1], pts[:, 2]] > 0).all():
                bad.append(f"{csv}: empty or has points outside gt_fg_rm_5")
    return bad


def find_zarrs(paths):
    out = []
    for p in paths:
        p = p.rstrip("/")
        if p.endswith(".zarr"):
            out.append(p)
        else:
            out += sorted(glob.glob(os.path.join(p, "*.zarr")))
            out += sorted(glob.glob(os.path.join(p, "*", "*.zarr")))
    return out


def _one(job):
    zpath, meta_dir, regen, fg_dir = job
    try:
        return zpath, verify_zarr(zpath, meta_dir, regen, fg_dir=fg_dir)
    except Exception as e:  # report and keep going over the batch
        return zpath, [f"{zpath}: {type(e).__name__}: {e}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--meta-dir", default=None, help="dir with tube_XXXXX_skeletons.npz")
    ap.add_argument("--regen", action="store_true", help="re-render and compare (slow)")
    ap.add_argument("--fg-dir", default=None, help="dir with tube_XXXXX.csv fg points")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    zarrs = find_zarrs(args.paths)
    if not zarrs:
        sys.exit(f"no .zarr under {args.paths}")
    jobs = [(z, args.meta_dir, args.regen, args.fg_dir) for z in zarrs]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_one, jobs))
    else:
        results = [_one(j) for j in jobs]
    n_bad = 0
    for zpath, problems in sorted(results):
        if problems:
            n_bad += 1
            for p in problems:
                print(p)
    print(f"verified {len(zarrs)} zarrs: {len(zarrs) - n_bad} OK, {n_bad} with problems")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
