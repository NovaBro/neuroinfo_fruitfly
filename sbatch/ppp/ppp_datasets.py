#!/usr/bin/env python3
"""Print fisbe/ dataset names or sample IDs for a PPP split.

Merges configs left-to-right (later wins), matching run_ppp.merge_dicts.
Looks under PatchPerPix/experiments/flylight/setups/setup01/ for bare names.

  train      train_datasets + val_datasets (staging extracts both tars;
             interleaved validation reads val_datasets)
  val/test   <split>_datasets, else train_datasets (run_ppp.dataset_lists)

Legacy configs without train_datasets print 'completely'.

Default mode prints '+'-joined dataset names (DATASETS format).

  --list-samples  print sorted {dataset}/{stem} lines (respects
                  data/max_test_samples for split=test)
  --count         print the number of samples (--list-samples length)

Usage:
  python3 sbatch/ppp/ppp_datasets.py --split train default_l40s.toml basic-ppp-noenc-cp.toml
  python3 sbatch/ppp/ppp_datasets.py --split test --list-samples \\
    default_l40s.toml datasets_test_himanshu.toml
  python3 sbatch/ppp/ppp_datasets.py --split test --count --datasets himanshu \\
    default_l40s.toml datasets_test_himanshu.toml
"""
from __future__ import annotations

import argparse
import os
import sys

from needs_decode import load_toml, merge_dicts, resolve_path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_FISBE = os.path.join(_REPO_ROOT, "fisbe")


def datasets_for_split(cfg, split):
    data = cfg.get("data", {})
    train = list(data.get("train_datasets") or [])
    if not train:
        return ["completely"]
    if split == "train":
        names = train + list(data.get("val_datasets") or train)
    else:
        names = list(data.get("%s_datasets" % split) or train)
    return list(dict.fromkeys(names))


def parse_datasets_override(raw):
    """Split DATASETS-style '+'/',' strings into folder names."""
    if not raw:
        return None
    names = []
    for part in raw.replace("+", ",").split(","):
        part = part.strip()
        if part:
            names.append(part)
    return names or None


def list_samples(cfg, split, fisbe_root=None, datasets=None):
    """Sorted ``{dataset}/{stem}`` IDs; apply max_test_samples for test."""
    data = cfg.get("data", {})
    root = fisbe_root or data.get("fisbe_root") or _DEFAULT_FISBE
    fmt = data.get("input_format", "zarr")
    names = parse_datasets_override(datasets) or datasets_for_split(cfg, split)
    samples = []
    for name in names:
        abs_dir = os.path.join(root, name, split)
        if not os.path.isdir(abs_dir):
            continue
        for fn in sorted(
                f for f in os.listdir(abs_dir) if f.endswith("." + fmt)):
            stem = os.path.splitext(fn)[0]
            samples.append(os.path.join(name, stem))
    if split == "test":
        max_n = data.get("max_test_samples")
        if max_n is not None:
            samples = samples[:int(max_n)]
    return samples


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument(
        "--list-samples",
        action="store_true",
        help="Print sorted {dataset}/{stem} lines instead of dataset names",
    )
    ap.add_argument(
        "--count",
        action="store_true",
        help="Print sample count (same list as --list-samples)",
    )
    ap.add_argument(
        "--datasets",
        default=None,
        help="Override dataset folders (+ or , separated); else from TOML",
    )
    ap.add_argument(
        "--fisbe-root",
        default=None,
        help="Override data/fisbe_root (default: from TOML or repo fisbe/)",
    )
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args(argv[1:])
    cfg = {}
    for name in args.configs:
        path = resolve_path(name)
        if not os.path.isfile(path):
            print(f"Error: config not found: {path}", file=sys.stderr)
            return 2
        cfg = merge_dicts(cfg, load_toml(path))

    if args.list_samples or args.count:
        samples = list_samples(
            cfg,
            args.split,
            fisbe_root=args.fisbe_root,
            datasets=args.datasets,
        )
        if args.count:
            print(len(samples))
        else:
            for s in samples:
                print(s)
        return 0

    names = parse_datasets_override(args.datasets) or datasets_for_split(
        cfg, args.split)
    print("+".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
