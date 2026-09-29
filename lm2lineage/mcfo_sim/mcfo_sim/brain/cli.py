"""Command line: python -m mcfo_sim.brain.cli <command> ...

  pool      build the neuron pool (downloads JRC2018U skeletons, ~1 per neuron)
  manifest  design N images (neuron ids, counts, overlap, imaging) - no pixels
  stats     neuron-count / overlap / reuse statistics (+ rendered counts)
  render    render manifest rows (resumable, shardable across machines)
  validate  integrity | calibrate | compare | c2st
"""
from __future__ import annotations

import argparse
import json
import os

from .config import BrainConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("pool")
    a.add_argument("--cache", default="cache")
    a.add_argument("--out", default="pool")
    a.add_argument("--max-neurons", type=int, default=0, help="0 = all eligible neurons")
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--config", help="JSON overriding BrainConfig fields")

    b = sub.add_parser("manifest")
    b.add_argument("--pool", default="pool")
    b.add_argument("--out", default="manifest")
    b.add_argument("--n-images", type=int, default=5000)
    b.add_argument("--seed", type=int)
    b.add_argument("--calibration", help="JSON from `validate calibrate`")
    b.add_argument("--config")

    c = sub.add_parser("stats")
    c.add_argument("--manifest", default="manifest")
    c.add_argument("--rendered")

    d = sub.add_parser("render")
    d.add_argument("--manifest", default="manifest")
    d.add_argument("--pool", default="pool")
    d.add_argument("--cache", default="cache")
    d.add_argument("--out", default="rendered")
    d.add_argument("--ids", nargs="*", help="image ids (default: all)")
    d.add_argument("--first", type=int, help="render only the first N rows")
    d.add_argument("--shard", help="k/n: render rows k, k+n, k+2n ... (for parallel jobs)")
    d.add_argument("--no-mip", action="store_true")

    e = sub.add_parser("validate")
    e.add_argument("what", choices=["integrity", "calibrate", "compare", "c2st"])
    e.add_argument("--rendered", default="rendered")
    e.add_argument("--manifest")
    e.add_argument("--fisbe", help="directory of FISBe .zarr samples")
    e.add_argument("--out", default="validation")
    e.add_argument("--limit", type=int)

    args = ap.parse_args()

    def cfg_from(path, **kw):
        cfg = BrainConfig.from_json(open(path).read()) if path else BrainConfig()
        for k, v in kw.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    if args.cmd == "pool":
        from .pool import build_pool
        build_pool(cfg_from(args.config), args.cache, args.out, args.max_neurons, args.workers)
    elif args.cmd == "manifest":
        from .manifest import build_manifest
        base = args.config or os.path.join(args.pool, "config.json")
        cfg = cfg_from(base, n_images=args.n_images, seed=args.seed)
        build_manifest(args.pool, args.out, cfg, calibration=args.calibration)
        from .stats import manifest_stats
        s = manifest_stats(args.out)
        print(json.dumps({k: s[k] for k in ("n_images", "n_neuron_instances", "unique_body_ids",
                                            "neurons_per_image", "images_by_overlap_mode")}, indent=1))
    elif args.cmd == "stats":
        from .stats import manifest_stats
        print(json.dumps(manifest_stats(args.manifest, args.rendered), indent=1, default=str))
    elif args.cmd == "render":
        import pandas as pd

        from .render import render_many
        ids = args.ids
        if args.first:
            ids = pd.read_parquet(os.path.join(args.manifest, "manifest_images.parquet")).image_id[:args.first].tolist()
        shard = tuple(map(int, args.shard.split("/"))) if args.shard else None
        render_many(args.manifest, args.pool, args.cache, args.out, rows=ids, shard=shard,
                    save_mip=not args.no_mip)
    elif args.cmd == "validate":
        from . import validate as V
        os.makedirs(args.out, exist_ok=True)
        if args.what == "integrity":
            df = V.integrity(args.rendered, args.manifest)
            print(df.to_string(index=False))
            print(f"passed {int(df['pass'].sum())}/{len(df)}")
        elif args.what == "calibrate":
            print(json.dumps(V.calibrate(args.fisbe, os.path.join(args.out, "calibration.json"), args.limit), indent=1))
        elif args.what == "compare":
            rep, sug = V.compare(args.fisbe, args.rendered, args.out, args.limit)
            print(rep.to_string(index=False))
            print("suggestions:", sug)
        elif args.what == "c2st":
            r = V.c2st(args.fisbe, args.rendered, limit=args.limit)
            json.dump(r, open(os.path.join(args.out, "c2st.json"), "w"), indent=1)
            print(r)


if __name__ == "__main__":
    main()
