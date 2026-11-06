#!/usr/bin/env python3
"""Resolve PPP config names to TOML paths under setup01/configs/.

Configs live in role subfolders (base/, presets/, data/, train/, vi/, eval/,
exp/; see setup01/configs/README.md). Callers keep passing bare file names
(e.g. ``vi_th_0_3.toml``); this finds the single match under configs/.

Lookup order for NAME:
  1. absolute path, or a path containing '/' that exists relative to the
     cwd → returned unchanged;
  2. NAME relative to configs/ (e.g. ``vi/vi_th_0_3.toml``);
  3. bare file name → the unique match anywhere under configs/, except
     archive/ (more than one match is an error);
  4. legacy flat file directly in setup01/ (old checkouts / compat symlinks).

The setup dir defaults to flylight/setups/setup01 and honours $SETUP_DIR
(relative to PatchPerPix/experiments), like ppp_job.sh.

CLI (one resolved path per line, exit 1 if any is missing):
  python3 sbatch/ppp/ppp_config_path.py vi_th_0_3.toml eval_rm200.toml
  python3 sbatch/ppp/ppp_config_path.py --relative-to PatchPerPix/experiments NAME...
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_DIR = os.path.join(_REPO_ROOT, "PatchPerPix", "experiments")
SETUP_DIR = os.path.join(
    EXPERIMENTS_DIR, os.environ.get("SETUP_DIR") or os.path.join("flylight", "setups", "setup01"))
CONFIG_DIR = os.path.join(SETUP_DIR, "configs")
ARCHIVE_DIR = "archive"


class ConfigNotFound(FileNotFoundError):
    pass


def _find_by_name(name):
    hits = []
    for root, dirs, files in os.walk(CONFIG_DIR):
        if os.path.relpath(root, CONFIG_DIR) == ".":
            dirs[:] = [d for d in dirs if d != ARCHIVE_DIR]
        if name in files:
            hits.append(os.path.join(root, name))
    return sorted(hits)


def resolve_path(name):
    """Return the TOML path for NAME (see module doc). Raises ConfigNotFound."""
    if os.path.isabs(name):
        if os.path.isfile(name):
            return name
        raise ConfigNotFound(f"config not found: {name}")
    if os.sep in name:
        if os.path.isfile(name):
            return name
        cand = os.path.join(CONFIG_DIR, name)
        if os.path.isfile(cand):
            return cand
        raise ConfigNotFound(f"config not found: {name} (cwd or {CONFIG_DIR})")
    hits = _find_by_name(name)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        rel = ", ".join(os.path.relpath(h, CONFIG_DIR) for h in hits)
        raise ConfigNotFound(f"ambiguous config name {name}: {rel} (pass the subfolder path)")
    legacy = os.path.join(SETUP_DIR, name)
    if os.path.isfile(legacy):
        return legacy
    raise ConfigNotFound(f"config not found: {name} (searched {CONFIG_DIR})")


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relative-to", default=None,
                    help="Print paths relative to this directory")
    ap.add_argument("names", nargs="+")
    args = ap.parse_args(argv[1:])
    rc = 0
    for name in args.names:
        try:
            path = os.path.abspath(resolve_path(name))
        except ConfigNotFound as e:
            print(f"Error: {e}", file=sys.stderr)
            rc = 1
            continue
        print(os.path.relpath(path, args.relative_to) if args.relative_to else path)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
