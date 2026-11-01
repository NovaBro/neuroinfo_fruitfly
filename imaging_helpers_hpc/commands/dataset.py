"""Inventory and MIP sqf-backed datasets under fisbe/<dataset>."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from imaging_helpers_hpc.imaging import (
    gen_basic_mip,
    gen_instance_projection,
    gen_scalar_mip,
)
from imaging_helpers_hpc.loading import (
    SynthMountMissing,
    load_zarr_volume,
    require_ppp_mount,
    require_synth_mount,
    zarr_array_metadata,
)
from imaging_helpers_hpc.paths import AnalysisOutputPaths, FisbeDataPaths

CSV_FIELDS = (
    "dataset",
    "split",
    "sample",
    "array",
    "shape",
    "dtype",
    "chunks",
    "symlink_target",
    "status",
)
VALID_SPLITS = ("train", "val", "test")
RGB_ARRAYS = ("raw", "raw_normalized")
LABEL_ARRAYS = ("gt_instances", "gt_instances_rm_5")
SCALAR_ARRAYS = ("gt_numinst", "gt_fg_rm_5")
PPP_PREPARED = (
    "raw_normalized",
    "gt_instances_rm_5",
    "gt_numinst",
    "gt_fg_rm_5",
)
MIP_ARRAYS = RGB_ARRAYS + LABEL_ARRAYS + SCALAR_ARRAYS
DEFAULT_MIP_ARRAYS = (
    "raw,gt_instances,raw_normalized,gt_instances_rm_5,gt_numinst,gt_fg_rm_5"
)
DEFAULT_INVENTORY_CSV = Path("imaging_helpers_hpc/output/sqf_dataset_inventory.csv")
PACK_SCRIPT = "sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh"


def parse_csv_list(value: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def iter_zarrs(
    dataset: str,
    splits: list[str],
    sample: str | None,
) -> list[tuple[str, Path]]:
    """Return (split, zarr_path) under fisbe/<dataset>, optionally one stem."""
    root = FisbeDataPaths(data_root=dataset)
    wanted = None
    if sample:
        wanted = sample[:-5] if sample.endswith(".zarr") else sample

    pairs: list[tuple[str, Path]] = []
    for split in splits:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {', '.join(VALID_SPLITS)}"
            )
        split_dir = root.paths[split]
        if not split_dir.is_dir():
            print(f"warning: missing split dir {split_dir}", file=sys.stderr)
            continue
        for zarr_path in sorted(split_dir.glob("*.zarr")):
            if wanted is not None and zarr_path.stem != wanted:
                continue
            pairs.append((split, zarr_path))
    return pairs


def _require_mounts(datasets: list[str], arrays: list[str]) -> None:
    prepared = [name for name in arrays if name in PPP_PREPARED]
    try:
        for dataset in datasets:
            require_synth_mount(dataset)
            if prepared and _prepared_symlinks_need_mount(dataset, prepared):
                require_ppp_mount(dataset)
    except SynthMountMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _prepared_symlinks_need_mount(dataset: str, arrays: list[str]) -> bool:
    """True when a prepared array is a symlink into /synth_MCFO/."""
    root = FisbeDataPaths(data_root=dataset)
    for split in VALID_SPLITS:
        split_dir = root.paths[split]
        if not split_dir.is_dir():
            continue
        for zarr_path in split_dir.glob("*.zarr"):
            for array in arrays:
                vol = zarr_path / "volumes" / array
                if not vol.is_symlink():
                    continue
                try:
                    target = os.readlink(vol)
                except OSError:
                    continue
                if target.startswith("/synth_MCFO/"):
                    return True
    return False


def _ensure_array(dataset: str, zarr_path: Path, array: str) -> None:
    """Exit if this volumes/<array> is missing or its symlink does not resolve."""
    vol = zarr_path / "volumes" / array
    pack = f"DATASETS={dataset} sbatch {PACK_SCRIPT}"
    if vol.is_symlink():
        try:
            target = os.readlink(vol)
        except OSError as exc:
            print(f"error: {vol}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if not vol.exists():
            print(f"error: dangling symlink {vol} -> {target}", file=sys.stderr)
            if array in PPP_PREPARED:
                print(pack, file=sys.stderr)
            raise SystemExit(1)
        return
    if not vol.exists():
        print(f"error: missing {vol}", file=sys.stderr)
        if array in PPP_PREPARED:
            print(pack, file=sys.stderr)
        raise SystemExit(1)


def run_inventory(
    datasets: list[str],
    splits: list[str],
    sample: str | None,
    arrays: list[str],
    out_path: Path,
) -> None:
    _require_mounts(datasets, arrays)
    rows: list[dict] = []
    n_samples = 0
    for dataset in datasets:
        pairs = iter_zarrs(dataset, splits, sample)
        if not pairs:
            print(
                f"warning: no zarrs for {dataset} splits={','.join(splits)}",
                file=sys.stderr,
            )
        for split, zarr_path in pairs:
            n_samples += 1
            print(f"{dataset}/{split}/{zarr_path.stem}", flush=True)
            for array in arrays:
                meta = zarr_array_metadata(zarr_path, array)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "sample": zarr_path.stem,
                        "array": array,
                        **meta,
                    }
                )

    if n_samples == 0:
        print("error: no zarrs matched", file=sys.stderr)
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_err = sum(1 for row in rows if row["status"] != "ok")
    print(f"samples: {n_samples}")
    print(f"rows: {len(rows)}  errors: {n_err}")
    print(f"wrote {out_path}")


def run_mip(
    datasets: list[str],
    splits: list[str],
    sample: str | None,
    arrays: list[str],
    limit: int,
    dpi: int,
    output_paths: AnalysisOutputPaths,
) -> None:
    if limit < 1:
        print("error: --limit must be >= 1", file=sys.stderr)
        raise SystemExit(1)
    unknown = [name for name in arrays if name not in MIP_ARRAYS]
    if unknown:
        print(
            f"error: unknown array {', '.join(unknown)}; "
            f"mip renders {', '.join(MIP_ARRAYS)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _require_mounts(datasets, arrays)
    n_written = 0
    for dataset in datasets:
        pairs = iter_zarrs(dataset, splits, sample)[:limit]
        if not pairs:
            print(
                f"warning: no zarrs for {dataset} splits={','.join(splits)}",
                file=sys.stderr,
            )
        for split, zarr_path in pairs:
            stem = zarr_path.stem
            print(f"{dataset}/{split}/{stem}", flush=True)
            image_dir = (
                output_paths.output_root / dataset / "images" / split / stem
            )
            for array in arrays:
                _ensure_array(dataset, zarr_path, array)
                volume = load_zarr_volume(zarr_path, array)
                out_path = image_dir / f"{stem}_{array}.png"
                title = f"{array}_{split}_{stem}"
                if array in RGB_ARRAYS:
                    gen_basic_mip(
                        volume,
                        title,
                        output_paths,
                        axis=1,
                        output_path=out_path,
                        dpi=dpi,
                    )
                elif array in LABEL_ARRAYS:
                    gen_instance_projection(
                        volume,
                        title,
                        output_paths,
                        axis=1,
                        output_path=out_path,
                        dpi=dpi,
                    )
                else:
                    gen_scalar_mip(
                        volume,
                        title,
                        output_paths,
                        axis=0,
                        output_path=out_path,
                        dpi=dpi,
                    )
                del volume
                n_written += 1

    if n_written == 0:
        print("error: no images written", file=sys.stderr)
        raise SystemExit(1)
    print(f"images: {n_written}")
