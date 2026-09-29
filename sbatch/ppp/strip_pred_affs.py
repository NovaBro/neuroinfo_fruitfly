#!/usr/bin/env python3
"""Remove volumes/pred_affs from PPP processed zarrs, keeping pred_numinst.

Keeps full pred_affs on samples listed in --keep-name (comma-separated).
Default keeps one FISBe-val and one FISBe-test stem when present:
  R22C03-20180918_66_J2.zarr,JRC_SS04989-20160318_24_A2.zarr
If none of the keep names exist in a ckpt dir, falls back to the first *.zarr.

Examples:
  # Dry-run entire metrics/ppp tree
  python3 sbatch/ppp/strip_pred_affs.py --dry-run

  # One-time cleanup
  python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp

  # Post-evaluate job hook (one sample, unless it is a keep-sample)
  python3 sbatch/ppp/strip_pred_affs.py \\
    --root metrics/ppp --exp ppp_basic_fmaps24 --split test \\
    --ckpt 80000 --sample R14A02-20180905_65_A6
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_KEEP = (
    "R22C03-20180918_66_J2.zarr,JRC_SS04989-20160318_24_A2.zarr"
)
AFF_REL = Path("volumes") / "pred_affs"
SPLITS = ("test", "val")


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}TiB"


def list_sample_zarrs(ckpt_dir: Path) -> list[Path]:
    return sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.endswith(".zarr"))


def normalize_keep_names(keep_name: str) -> list[str]:
    """Parse comma-separated keep names; ensure .zarr suffix."""
    names: list[str] = []
    for part in keep_name.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.endswith(".zarr"):
            part = f"{part}.zarr"
        names.append(part)
    return names


def resolve_keep_set(samples: list[Path], keep_name: str) -> set[Path]:
    """Samples to retain. Prefer matching keep names; else first sample."""
    if not samples:
        return set()
    want = set(normalize_keep_names(keep_name))
    kept = {s for s in samples if s.name in want}
    if kept:
        return kept
    return {samples[0]}


def iter_ckpt_dirs(root: Path, exp: str | None, split: str | None, ckpt: str | None):
    if exp:
        exp_dirs = [root / exp] if (root / exp).is_dir() else []
    else:
        exp_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    splits = (split,) if split else SPLITS
    for exp_dir in exp_dirs:
        for sp in splits:
            processed = exp_dir / sp / "processed"
            if not processed.is_dir():
                continue
            if ckpt:
                d = processed / str(ckpt)
                if d.is_dir():
                    yield exp_dir.name, sp, d
            else:
                for d in sorted(p for p in processed.iterdir() if p.is_dir()):
                    yield exp_dir.name, sp, d


def strip_affs(aff_path: Path, dry_run: bool) -> int:
    """Remove pred_affs dir. Returns bytes freed (pre-delete size)."""
    size = dir_size_bytes(aff_path)
    if dry_run:
        print(f"DRY_RUN rm -rf {aff_path}  ({fmt_bytes(size)})")
        return size
    print(f"rm -rf {aff_path}  ({fmt_bytes(size)})")
    shutil.rmtree(aff_path)
    return size


def process_ckpt_dir(
    exp: str,
    split: str,
    ckpt_dir: Path,
    keep_name: str,
    sample_filter: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Returns (n_stripped, bytes_freed)."""
    samples = list_sample_zarrs(ckpt_dir)
    if not samples:
        return 0, 0
    keep = resolve_keep_set(samples, keep_name)
    keep_names = ", ".join(sorted(s.name for s in keep))
    print(f"[{exp}/{split}/processed/{ckpt_dir.name}] keep={keep_names}")

    n_stripped = 0
    freed = 0
    for sample in samples:
        stem = sample.name
        if sample_filter:
            # Accept with or without .zarr suffix.
            want = sample_filter if sample_filter.endswith(".zarr") else f"{sample_filter}.zarr"
            if stem != want:
                continue
        aff = sample / AFF_REL
        if not aff.exists():
            print(f"  skip {stem}: no pred_affs")
            continue
        if sample in keep:
            print(f"  keep  {stem}: pred_affs retained ({fmt_bytes(dir_size_bytes(aff))})")
            continue
        freed += strip_affs(aff, dry_run=dry_run)
        n_stripped += 1
    return n_stripped, freed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="metrics/ppp", help="PPP metrics root (default: metrics/ppp)")
    p.add_argument("--exp", default=None, help="Limit to one experiment id")
    p.add_argument("--split", choices=["test", "val"], default=None, help="Limit to one split")
    p.add_argument("--ckpt", default=None, help="Limit to one checkpoint dir name")
    p.add_argument(
        "--sample",
        default=None,
        help="Only consider this sample stem (still skipped if it is a keep-sample)",
    )
    p.add_argument(
        "--keep-name",
        default=DEFAULT_KEEP,
        help=f"Comma-separated keep samples (default: {DEFAULT_KEEP})",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without deleting")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Error: root not found: {root}", file=sys.stderr)
        return 1

    total_n = 0
    total_bytes = 0
    for exp, split, ckpt_dir in iter_ckpt_dirs(root, args.exp, args.split, args.ckpt):
        n, b = process_ckpt_dir(
            exp, split, ckpt_dir, args.keep_name, args.sample, args.dry_run
        )
        total_n += n
        total_bytes += b

    action = "would free" if args.dry_run else "freed"
    print(f"Done: stripped {total_n} pred_affs dirs, {action} {fmt_bytes(total_bytes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
