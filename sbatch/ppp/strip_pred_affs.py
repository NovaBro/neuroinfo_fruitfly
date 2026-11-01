#!/usr/bin/env python3
"""Remove large prediction arrays from PPP processed zarrs, keeping pred_numinst.

By default strips volumes/pred_affs and volumes/pred_code on every sample.
Pass --keep-name to retain those arrays on listed samples (comma-separated).
An empty keep list retains nothing. Unmatched names do not fall back to the
first *.zarr.

Examples:
  # Dry-run entire metrics/ppp tree
  python3 sbatch/ppp/strip_pred_affs.py --dry-run

  # One-time cleanup
  python3 sbatch/ppp/strip_pred_affs.py --root metrics/ppp

  # Affs only (legacy)
  python3 sbatch/ppp/strip_pred_affs.py --arrays volumes/pred_affs

  # Post-evaluate job hook (strips every sample unless --keep-name is set)
  python3 sbatch/ppp/strip_pred_affs.py \\
    --root metrics/ppp --exp ppp_basic_fmaps24 --split test \\
    --ckpt 80000 --keep-name ""
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_KEEP = ""
DEFAULT_ARRAYS = ("volumes/pred_affs", "volumes/pred_code")
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
    """Flat ``*.zarr`` at ckpt root plus nested ``{dataset}/*.zarr`` (list mode)."""
    flat = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.endswith(".zarr")
    ]
    nested = [
        p for p in ckpt_dir.glob("*/*.zarr")
        if p.is_dir()
    ]
    return sorted(set(flat) | set(nested))


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


def parse_arrays(raw: str) -> list[Path]:
    """Parse comma-separated relative array paths (e.g. volumes/pred_affs)."""
    arrays: list[Path] = []
    for part in raw.split(","):
        part = part.strip().strip("/")
        if not part:
            continue
        arrays.append(Path(part))
    if not arrays:
        raise ValueError("--arrays must list at least one path")
    return arrays


def resolve_keep_set(samples: list[Path], keep_name: str) -> set[Path]:
    """Samples to retain. Empty or unmatched keep names retain nothing.

    Keep names may be bare ``stem.zarr`` or namespaced ``dataset/stem.zarr``.
    """
    if not samples or not keep_name.strip():
        return set()
    want = set(normalize_keep_names(keep_name))
    if not want:
        return set()
    kept = set()
    for s in samples:
        if s.name in want:
            kept.add(s)
            continue
        # Nested list-mode: match ``completely/stem.zarr`` or bare stem.
        rel = s.as_posix()
        for w in want:
            if rel.endswith("/" + w) or rel.endswith(w):
                kept.add(s)
                break
    return kept


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


def strip_array(arr_path: Path, dry_run: bool) -> int:
    """Remove one array dir. Returns bytes freed (pre-delete size)."""
    size = dir_size_bytes(arr_path)
    if dry_run:
        print(f"DRY_RUN rm -rf {arr_path}  ({fmt_bytes(size)})")
        return size
    print(f"rm -rf {arr_path}  ({fmt_bytes(size)})")
    shutil.rmtree(arr_path)
    return size


def array_label(rel: Path) -> str:
    return rel.name


def process_ckpt_dir(
    exp: str,
    split: str,
    ckpt_dir: Path,
    keep_name: str,
    sample_filter: str | None,
    arrays: list[Path],
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
        rel_stem = sample.relative_to(ckpt_dir).as_posix().removesuffix(".zarr")
        if sample_filter:
            want = sample_filter.removesuffix(".zarr")
            if want not in (rel_stem, stem.removesuffix(".zarr"), Path(want).name):
                continue
        for rel in arrays:
            label = array_label(rel)
            arr = sample / rel
            if not arr.exists():
                print(f"  skip {stem}: no {label}")
                continue
            if sample in keep:
                print(
                    f"  keep  {stem}: {label} retained "
                    f"({fmt_bytes(dir_size_bytes(arr))})"
                )
                continue
            freed += strip_array(arr, dry_run=dry_run)
            n_stripped += 1
    return n_stripped, freed


def main() -> int:
    default_arrays = ",".join(DEFAULT_ARRAYS)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        help=(
            "Comma-separated samples to retain. "
            "Default is empty: strip every sample. "
            "Unmatched names retain nothing."
        ),
    )
    p.add_argument(
        "--arrays",
        default=default_arrays,
        help=(
            "Comma-separated relative array paths to strip "
            f"(default: {default_arrays})"
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without deleting")
    args = p.parse_args()

    try:
        arrays = parse_arrays(args.arrays)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    root = Path(args.root)
    if not root.is_dir():
        print(f"Error: root not found: {root}", file=sys.stderr)
        return 1

    total_n = 0
    total_bytes = 0
    for exp, split, ckpt_dir in iter_ckpt_dirs(root, args.exp, args.split, args.ckpt):
        n, b = process_ckpt_dir(
            exp, split, ckpt_dir, args.keep_name, args.sample, arrays, args.dry_run
        )
        total_n += n
        total_bytes += b

    labels = ", ".join(array_label(a) for a in arrays)
    action = "would free" if args.dry_run else "freed"
    print(
        f"Done: stripped {total_n} dirs ({labels}), "
        f"{action} {fmt_bytes(total_bytes)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
