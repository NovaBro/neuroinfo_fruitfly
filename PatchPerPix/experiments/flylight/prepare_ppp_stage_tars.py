"""Pack PPP training arrays into uncompressed tars for fast SLURM_TMPDIR staging.

rsync of post-rechunk 64^3 zarrs is metadata-bound (~25+ min). One sequential
tar extract per split is much faster. Includes only arrays train.py needs;
excludes volumes/raw.

Per-dataset layout (Phase 2)::

  fisbe/<dataset>/train_ppp.tar
  fisbe/<dataset>/val_ppp.tar

Staging extracts to ``$SLURM_TMPDIR/ppp_data/<dataset>/{train,val}/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Arrays used by flylight setup01 training (gt_instances is hardcoded for shape).
PPP_ARRAYS = (
    "raw_normalized",
    "gt_instances_rm_5",
    "gt_numinst",
    "gt_fg_rm_5",
    "gt_instances",
)

META_NAMES = (".zgroup", ".zattrs")
DEFAULT_SPLITS = ("train", "val")
REQUIRED_ARRAYS = (
    "raw_normalized",
    "gt_instances_rm_5",
    "gt_numinst",
    "gt_fg_rm_5",
    "gt_instances",
)


def _add_if_exists(tar: tarfile.TarFile, path: Path, arcname: str) -> int:
    if not path.exists():
        return 0
    tar.add(path, arcname=arcname, recursive=False)
    return 1


def _add_tree(tar: tarfile.TarFile, path: Path, arcname: str) -> tuple[int, int]:
    """Add directory tree; return (n_files, total_bytes)."""
    n_files = 0
    n_bytes = 0
    if not path.exists():
        return 0, 0
    for root, _dirs, files in os.walk(path):
        root_p = Path(root)
        rel = root_p.relative_to(path)
        for name in files:
            fp = root_p / name
            if rel == Path("."):
                arc = f"{arcname}/{name}"
            else:
                arc = f"{arcname}/{rel.as_posix()}/{name}"
            tar.add(fp, arcname=arc, recursive=False)
            n_files += 1
            n_bytes += fp.stat().st_size
        # ensure directory entries exist even if empty
        if not files and not list(root_p.iterdir()):
            tar.add(root_p, arcname=(
                arcname if rel == Path(".") else f"{arcname}/{rel.as_posix()}"
            ), recursive=False)
    return n_files, n_bytes


def _sample_missing_arrays(zarr_path: Path) -> list[str]:
    vol = zarr_path / "volumes"
    return [a for a in REQUIRED_ARRAYS if not (vol / a).exists()]


def pack_split(split_dir: Path, out_tar: Path, overwrite: bool = False) -> None:
    if out_tar.exists() and not overwrite:
        print(f"skip (exists): {out_tar}", flush=True)
        return
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir missing: {split_dir}")

    zarrs = sorted(split_dir.glob("*.zarr"))
    if not zarrs:
        raise FileNotFoundError(f"no *.zarr under {split_dir}")

    missing_samples = []
    for zarr_path in zarrs:
        miss = _sample_missing_arrays(zarr_path)
        if miss:
            missing_samples.append((zarr_path.name, miss))
    if missing_samples:
        for name, miss in missing_samples[:5]:
            print(
                f"error: {split_dir / name} missing {miss}; "
                "run prepare_fisbe_for_ppp.py first",
                flush=True,
            )
        if len(missing_samples) > 5:
            print(
                f"error: … and {len(missing_samples) - 5} more samples",
                flush=True,
            )
        raise SystemExit(
            f"{len(missing_samples)} sample(s) under {split_dir} lack PPP arrays"
        )

    out_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_tar.with_suffix(out_tar.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    t0 = time.time()
    n_files = 0
    n_bytes = 0
    with tarfile.open(tmp, mode="w") as tar:
        for zarr_path in zarrs:
            sample = zarr_path.name
            print(f"  + {sample}", flush=True)
            for meta in META_NAMES:
                n_files += _add_if_exists(tar, zarr_path / meta, f"{sample}/{meta}")
            vol = zarr_path / "volumes"
            for meta in META_NAMES:
                n_files += _add_if_exists(
                    tar, vol / meta, f"{sample}/volumes/{meta}"
                )
            for arr in PPP_ARRAYS:
                arr_path = vol / arr
                if not arr_path.exists():
                    print(f"    warn: missing {arr_path}", flush=True)
                    continue
                nf, nb = _add_tree(tar, arr_path, f"{sample}/volumes/{arr}")
                n_files += nf
                n_bytes += nb

    os.replace(tmp, out_tar)
    elapsed = time.time() - t0
    size = out_tar.stat().st_size
    print(
        f"wrote {out_tar}  samples={len(zarrs)}  files={n_files}  "
        f"payload={n_bytes / 1e9:.2f}GB  tar={size / 1e9:.2f}GB  "
        f"in {elapsed:.0f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-dataset train_ppp.tar / val_ppp.tar for staging"
    )
    parser.add_argument(
        "--fisbe-root",
        default="/scratch/wmz2007/neuroinfo_fruitfly/fisbe",
        help="FISBe root holding <dataset>/{train,val}/",
    )
    parser.add_argument(
        "--datasets",
        default="completely",
        help="Comma-separated dataset folder names under fisbe-root",
    )
    parser.add_argument(
        "--completely-subdir",
        default=None,
        help=argparse.SUPPRESS,  # legacy alias → single dataset
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated split names under each dataset/",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_ppp.tar files",
    )
    args = parser.parse_args()

    if args.completely_subdir:
        datasets = [args.completely_subdir]
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    fisbe_root = Path(args.fisbe_root)

    for dataset in datasets:
        base = fisbe_root / dataset
        if not base.is_dir():
            print(f"warn: dataset dir missing, skip: {base}", flush=True)
            continue
        for split in splits:
            split_dir = base / split
            out_tar = base / f"{split}_ppp.tar"
            if not split_dir.is_dir():
                print(
                    f"warn: split dir missing, skip: {split_dir}",
                    flush=True,
                )
                continue
            print(f"packing {split_dir} -> {out_tar}", flush=True)
            pack_split(split_dir, out_tar, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
