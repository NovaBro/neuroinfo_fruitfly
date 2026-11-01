"""Check that each FISBe completely gt_instances channel is a single 26-CC."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

STRUCT26 = np.ones((3, 3, 3), dtype=np.uint8)
DEFAULT_OUT = Path("imaging_helpers_hpc/output/gt_instance_connectivity.csv")
CSV_FIELDS = (
    "split",
    "sample",
    "channel",
    "label_id",
    "voxels",
    "n_cc",
    "largest_frac",
)


def check_sample(zarr_path: Path, split: str) -> list[dict]:
    """Return one row per non-empty gt_instances channel."""
    group = zarr.open_group(str(zarr_path), mode="r")
    arr = group["volumes/gt_instances"]
    sample = zarr_path.name
    rows: list[dict] = []

    for c in range(arr.shape[0]):
        channel = np.asarray(arr[c])
        mask = channel > 0
        if not mask.any():
            continue

        labels = np.unique(channel)
        labels = labels[labels > 0]
        label_id = int(labels[0])

        labeled, n_cc = ndimage.label(mask, structure=STRUCT26)
        sizes = np.bincount(labeled.ravel())[1:]
        largest_frac = float(sizes.max() / sizes.sum()) if sizes.size else 0.0

        rows.append(
            {
                "split": split,
                "sample": sample,
                "channel": c,
                "label_id": label_id,
                "voxels": int(mask.sum()),
                "n_cc": int(n_cc),
                "largest_frac": largest_frac,
            }
        )
    return rows


def _check_sample_job(args: tuple[str, str]) -> tuple[str, str, list[dict]]:
    """Worker entry: (split, zarr_path_str) -> (split, sample_name, rows)."""
    split, zarr_path_str = args
    zarr_path = Path(zarr_path_str)
    rows = check_sample(zarr_path, split)
    return split, zarr_path.name, rows


def iter_zarrs(fisbe_root: Path, splits: list[str]) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = fisbe_root / split
        if not split_dir.is_dir():
            print(f"warning: missing split dir {split_dir}", file=sys.stderr)
            continue
        for zp in sorted(split_dir.glob("*.zarr")):
            pairs.append((split, zp))
    return pairs


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> list[dict]:
    offenders = [r for r in rows if r["n_cc"] != 1]
    print(f"instances checked: {len(rows)}")
    print(f"multi-component:   {len(offenders)}")
    for r in offenders:
        print(
            f"  {r['split']}/{r['sample']} ch={r['channel']} "
            f"label={r['label_id']} n_cc={r['n_cc']} "
            f"voxels={r['voxels']} largest_frac={r['largest_frac']:.4f}"
        )
    return offenders


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check each volumes/gt_instances channel in fisbe/completely "
            "for a single 26-connected component."
        )
    )
    parser.add_argument(
        "--fisbe-root",
        type=Path,
        default=Path("fisbe/completely"),
        help="Root containing train/val/test zarr dirs (default: fisbe/completely)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to scan (default: train val test)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"CSV output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Optional basename filter, e.g. R38F04-20181005_63_G5.zarr",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel workers across samples "
            "(default: SLURM_CPUS_PER_TASK or min(8, cpu_count))"
        ),
    )
    return parser.parse_args(argv)


def _default_workers() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return max(1, int(slurm))
    return max(1, min(8, os.cpu_count() or 1))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fisbe_root: Path = args.fisbe_root
    if not fisbe_root.is_dir():
        print(f"error: fisbe root not found: {fisbe_root}", file=sys.stderr)
        return 2

    pairs = iter_zarrs(fisbe_root, args.splits)
    if args.sample is not None:
        name = args.sample if args.sample.endswith(".zarr") else f"{args.sample}.zarr"
        pairs = [(s, p) for s, p in pairs if p.name == name]
        if not pairs:
            print(f"error: sample not found: {name}", file=sys.stderr)
            return 2

    workers = args.workers if args.workers is not None else _default_workers()
    workers = max(1, min(workers, len(pairs)))
    print(f"scanning {len(pairs)} samples with {workers} workers", flush=True)

    jobs = [(split, str(zp)) for split, zp in pairs]
    rows: list[dict] = []

    if workers == 1:
        for split, zp_str in jobs:
            print(f"checking {split}/{Path(zp_str).name} ...", flush=True)
            rows.extend(check_sample(Path(zp_str), split))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_check_sample_job, job): job for job in jobs}
            for fut in as_completed(futures):
                split, sample, sample_rows = fut.result()
                print(
                    f"done {split}/{sample} ({len(sample_rows)} instances)",
                    flush=True,
                )
                rows.extend(sample_rows)

    # Stable CSV order: split, sample, channel
    rows.sort(key=lambda r: (r["split"], r["sample"], r["channel"]))

    write_csv(args.out, rows)
    print(f"wrote {args.out}")
    offenders = print_summary(rows)
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
