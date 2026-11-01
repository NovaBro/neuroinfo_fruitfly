#!/usr/bin/env python3
"""Print 1 if merged PPP TOMLs have train_code, else 0.

Merges configs left-to-right (later wins), matching run_ppp.merge_dicts.
Looks under PatchPerPix/experiments/flylight/setups/setup01/ for bare names.

Usage:
  python3 sbatch/ppp/needs_decode.py default_l40s.toml basic-ppp-decoder.toml
  python3 sbatch/ppp/needs_decode.py original_default_train_code.toml basic-ppp-decoder.toml
"""
from __future__ import annotations

import os
import sys

try:
    import tomllib
except ImportError:
    tomllib = None
    try:
        import toml
    except ImportError:
        print("Error: need tomllib (3.11+) or toml package", file=sys.stderr)
        sys.exit(2)


SETUP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "PatchPerPix",
    "experiments",
    "flylight",
    "setups",
    "setup01",
)


def merge_dicts(sink, source):
    if not isinstance(sink, dict) or not isinstance(source, dict):
        raise TypeError("Args to merge_dicts should be dicts")
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(sink.get(k), dict):
            sink[k] = merge_dicts(sink[k], v)
        else:
            sink[k] = v
    return sink


def load_toml(path):
    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)
    with open(path, "r") as f:
        return toml.load(f)


def resolve_path(name):
    if os.path.isabs(name) or os.path.sep in name:
        return name
    return os.path.join(SETUP_DIR, os.path.basename(name))


def main(argv):
    if len(argv) < 2:
        print("Usage: needs_decode.py CONFIG.toml [CONFIG.toml …]", file=sys.stderr)
        return 2
    cfg = {}
    for name in argv[1:]:
        path = resolve_path(name)
        if not os.path.isfile(path):
            print(f"Error: config not found: {path}", file=sys.stderr)
            return 2
        cfg = merge_dicts(cfg, load_toml(path))
    train_code = bool(
        cfg.get("model", {}).get("train_code")
        or cfg.get("training", {}).get("train_code")
    )
    print("1" if train_code else "0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
