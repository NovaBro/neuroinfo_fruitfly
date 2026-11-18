"""Generate the simple_v1 synthetic tube dataset (md_guides/dataset/simple_dataset_plan.md).

Run from the repo root:

  # render samples [start, start + count) into <out>/{split}/tube_XXXXX.zarr
  python -m synthetic.simple_tubes.generate render --out $SLURM_TMPDIR/simple_v1 \
      --meta-out $SLURM_TMPDIR/simple_v1/meta --fg-out $SLURM_TMPDIR/simple_v1/fg_points \
      --confetti-splits val,test --start 0 --count 250 --workers 16 --verify

  # fisbe/<dataset> tree + sample list (links resolve once the parts are mounted)
  python -m synthetic.simple_tubes.generate skeleton --dataset-dir fisbe/simple_v1 \
      --fg-dir fisbe/fg_points/simple_v1

Each zarr is self-contained for PatchPerPix (no prepare_fisbe_for_ppp.py / pack step):
  raw            uint8 (3,Z,Y,X) 0..15, attrs clipmax=15 (train.py / predict_no_gp.py scale it)
  gt_instances   uint8 (N,Z,Y,X), channel n holds n + 1
  gt_numinst     uint8 (Z,Y,X) instances per voxel
  gt_fg_rm_5     float32 0/1, foreground opened with ball(1) (prepare_fisbe_for_ppp.fg_rm_mask)
  raw_normalized -> raw, gt_instances_rm_5 -> gt_instances  (relative symlinks; default config keys)

sbatch/ppp/prep/simple/generate_simple_sbatch.sh wraps both and packs SquashFS parts.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from . import layout

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPMAX = 15                      # 4-bit raw
SPATIAL_CHUNKS = (64, 64, 64)     # as prepare_fisbe_for_ppp.py
ALIASES = {"raw_normalized": "raw", "gt_instances_rm_5": "gt_instances"}


@functools.lru_cache(maxsize=None)
def ppp_module(name: str):
    """Import PatchPerPix/experiments/flylight/<name>.py (not a package) by path."""
    path = REPO_ROOT / "PatchPerPix" / "experiments" / "flylight" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ppp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pair(text, cast=float):
    a, b = (cast(v) for v in text.split(","))
    return (a, b)


def _splits(text):
    return set(s for s in text.split(",") if s) if text else set()


def _params(args, split):
    from .geometry import GeometryParams
    from .render import RenderParams

    shape = tuple(int(v) for v in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError(f"--shape needs Z,Y,X, got {args.shape!r}")
    gp = GeometryParams(shape=shape, spline=args.spline,
                        root_radius=_pair(args.root_radius),
                        radius_floor=args.radius_floor)
    rp = RenderParams(neurons=_pair(args.neurons, int), noise=not args.no_noise,
                      confetti=split in _splits(args.confetti_splits))
    return gp, rp


def write_sample(sample: dict, out_dir: str, meta_dir: str | None, counts, mip=True,
                 dataset: str = layout.DATASET, fg_dir: str | None = None) -> str:
    """Write one sample as a self-contained zarr plus sidecars. Returns the zarr path."""
    import numcodecs
    import numpy as np
    import zarr

    from .render_util import gt_channel

    index = sample["index"]
    name = layout.sample_name(index)
    shape = sample["shape"]
    zpath = os.path.join(out_dir, layout.zarr_relpath(index, counts))
    comp = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    root = zarr.open_group(zpath, mode="w")
    root.attrs.update({
        **layout.BASE_ATTRS,
        "simple_tubes": {
            "dataset": dataset,
            "index": index,
            "base_seed": sample["base_seed"],
            "seed": sample["seed"],
            "split_counts": list(counts),
            "shape": list(shape),
            "params": sample["params"],
            "noise": sample["noise"],
            "instances": [{k: inst[k] for k in ("label", "colour", "level", "voxels")}
                          for inst in sample["instances"]],
        },
    })
    vol = root.create_group("volumes")
    raw = vol.create_dataset("raw", data=sample["raw"], chunks=(3, *SPATIAL_CHUNKS),
                             compressor=comp, write_empty_chunks=False)
    raw.attrs["clipmax"] = CLIPMAX
    gt = vol.create_dataset("gt_instances", shape=(len(sample["flats"]), *shape),
                            dtype="u1", chunks=(1, *SPATIAL_CHUNKS), fill_value=0,
                            compressor=comp, write_empty_chunks=False)
    numinst = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    for n, flat in enumerate(sample["flats"]):
        gt[n] = gt_channel(flat, n + 1, shape)   # one dense channel at a time
        numinst[flat] += 1                        # flats are unique per instance
    numinst = numinst.reshape(shape)
    vol.create_dataset("gt_numinst", data=numinst, chunks=SPATIAL_CHUNKS,
                       compressor=comp, write_empty_chunks=False)
    fg = ppp_module("prepare_fisbe_for_ppp").fg_rm_mask(numinst, opening_radius=1)
    del numinst
    vol.create_dataset("gt_fg_rm_5", data=fg, chunks=SPATIAL_CHUNKS,
                       compressor=comp, write_empty_chunks=False)
    del fg
    vdir = os.path.join(zpath, "volumes")
    for alias, target in ALIASES.items():
        os.symlink(target, os.path.join(vdir, alias))   # relative, valid inside the image

    if fg_dir:
        os.makedirs(fg_dir, exist_ok=True)
        ppp_module("prepare_fg_points_csv").write_sample_csv(
            zpath, os.path.join(fg_dir, name + ".csv"), stride=4, overwrite=True)

    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)
        meta = {k: sample[k] for k in ("index", "seed", "base_seed", "stats", "params",
                                         "noise", "instances")}
        meta["shape"] = list(shape)
        meta["name"] = name
        meta["split"] = layout.split_of(index, counts)
        with open(os.path.join(meta_dir, name + ".json"), "w") as f:
            json.dump(meta, f)
        np.savez_compressed(os.path.join(meta_dir, name + "_skeletons.npz"),
                            **sample["skeletons"])
        if mip:
            save_mip(sample["raw"], sample["flats"], shape,
                     os.path.join(meta_dir, name + "_mip.png"))
    return zpath


def save_mip(raw, flats, shape, path):
    """Colour MIP of raw (left) and of the GT instances (right), max over Z."""
    import numpy as np
    from PIL import Image

    rgb = (raw.max(axis=1).transpose(1, 2, 0).astype(np.uint16) * 17).astype(np.uint8)
    N = len(flats)
    ny, nx = shape[1], shape[2]
    rng = np.random.default_rng(12345)
    hues = (np.arange(N) / max(N, 1) + rng.uniform(0, 1)) % 1.0
    palette = np.stack([_hsv_to_rgb(h, 0.85, 1.0) for h in hues]) if N else np.zeros((0, 3))
    lab = np.zeros(rgb.shape, dtype=np.uint8)
    for n in range(N):
        yx = np.unique(flats[n] % (ny * nx))
        lab.reshape(-1, 3)[yx] = palette[n]
    sep = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    Image.fromarray(np.concatenate([rgb, sep, lab], axis=1)).save(path)


def _hsv_to_rgb(h, s, v):
    import colorsys

    return [int(round(255 * c)) for c in colorsys.hsv_to_rgb(h, s, v)]


def _render_one(job):
    index, args, counts = job
    from .render import make_sample

    t0 = time.time()
    split = layout.split_of(index, counts)
    gp, rp = _params(args, split)
    sample = make_sample(index, args.seed, gp, rp)
    zpath = write_sample(sample, args.out, args.meta_out, counts,
                         mip=split in _splits(args.mip_splits), dataset=args.dataset,
                         fg_dir=args.fg_out)
    problems = []
    if args.verify:
        from .verify import verify_zarr

        problems = verify_zarr(zpath, meta_dir=args.meta_out, fg_dir=args.fg_out)
    st = sample["stats"]
    return (index, zpath, st, problems, time.time() - t0)


def cmd_render(args):
    counts = layout.parse_split_counts(args.splits)
    total = sum(counts)
    end = min(args.start + args.count, total)
    if args.start < 0 or args.start >= end:
        sys.exit(f"nothing to do: start={args.start} count={args.count} total={total}")
    jobs = [(i, args, counts) for i in range(args.start, end)]
    print(f"rendering {len(jobs)} samples [{args.start}, {end}) -> {args.out} "
          f"(workers={args.workers}, seed={args.seed}, confetti in {args.confetti_splits})",
          flush=True)
    bad = 0
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers, maxtasksperchild=20) as pool:
            results = pool.imap_unordered(_render_one, jobs)
            bad = _report(results, len(jobs))
    else:
        bad = _report(map(_render_one, jobs), len(jobs))
    print(f"done {len(jobs)} samples in {time.time() - t0:.0f} s; {bad} failed verify",
          flush=True)
    if bad:
        sys.exit(1)


def _report(results, n):
    bad = 0
    for k, (index, zpath, st, problems, dt) in enumerate(results, 1):
        flag = "OK" if not problems else "FAIL"
        print(f"[{k}/{n}] {layout.sample_name(index)} {flag} N={st['n_instances']}"
              f"/{st['n_target']} redraws={st['redraws']} skipped={st['skipped']} "
              f"{dt:.1f}s", flush=True)
        for p in problems:
            print(f"    {p}", flush=True)
        bad += bool(problems)
    return bad


def cmd_skeleton(args):
    counts = layout.parse_split_counts(args.splits)
    os.makedirs(args.dataset_dir, exist_ok=True)
    mount = args.mount or "/" + args.dataset
    n = layout.write_skeleton(args.dataset_dir, counts, mount=mount, dataset=args.dataset,
                              overwrite=args.overwrite, fg_dir=args.fg_dir)
    print(f"skeleton: {n} loose val/test zarrs under {args.dataset_dir}; train, meta"
          f"{' and fg points' if args.fg_dir else ''} link into {mount}")
    if args.sources:
        path = os.path.join(args.dataset_dir, "source_sqf.txt")
        with open(path, "w") as f:
            f.write(f"# SquashFS parts holding the {args.dataset} zarrs, meta and fg points "
                    f"(mount at {mount})\n")
            for s in args.sources:
                f.write(os.path.abspath(s) + "\n")
        print(f"wrote {path} ({len(args.sources)} parts)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="generate samples [start, start+count)")
    r.add_argument("--out", required=True, help="dataset root; writes <out>/{split}/*.zarr")
    r.add_argument("--dataset", default=layout.DATASET, help="name stored in the zarr attrs")
    r.add_argument("--meta-out", default=None, help="JSON / skeleton npz / MIP png dir")
    r.add_argument("--fg-out", default=None, help="fg point CSV dir (stride-4 grid of gt_fg_rm_5)")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--count", type=int, default=1)
    r.add_argument("--seed", type=int, default=0, help="base seed; sample seed = seed*1e6 + index")
    r.add_argument("--splits", default="800,100,100", help="train,val,test counts")
    r.add_argument("--shape", default="400,600,600", help="volume Z,Y,X")
    r.add_argument("--neurons", default="10,30", help="neurons per volume, inclusive")
    r.add_argument("--no-noise", action="store_true",
                   help="tubes only: no haze, distractor blobs or confetti")
    r.add_argument("--confetti-splits", default="train,val,test",
                   help="splits that get baked confetti; train usually gets it on the GPU "
                        "instead ([training.augmentation.gpu_confetti])")
    r.add_argument("--mip-splits", default="train,val,test", help="splits that get a MIP png")
    r.add_argument("--root-radius", default="1.5,3.0")
    r.add_argument("--radius-floor", type=float, default=1.5)
    r.add_argument("--spline", default="catmull-rom", choices=["catmull-rom", "bspline"])
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--verify", action="store_true", help="run verify.py checks on each zarr")
    r.set_defaults(func=cmd_render)

    s = sub.add_parser("skeleton", help="write the fisbe/<dataset> tree + sample list")
    s.add_argument("--dataset-dir", required=True, help="e.g. fisbe/simple_v1")
    s.add_argument("--splits", default="800,100,100")
    s.add_argument("--dataset", default=layout.DATASET, help="name in the sample list header")
    s.add_argument("--mount", default=None, help="where the parts mount (default /<dataset>)")
    s.add_argument("--fg-dir", default=None,
                   help="fg points dir to link into the image, e.g. fisbe/fg_points/simple_v1")
    s.add_argument("--sources", nargs="*", default=None,
                   help="SquashFS parts to list in source_sqf.txt")
    s.add_argument("--overwrite", action="store_true", help="replace links with other targets")
    s.set_defaults(func=cmd_skeleton)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
