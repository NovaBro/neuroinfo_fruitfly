from __future__ import annotations

import logging
import re
from pathlib import Path

from imaging_helpers_hpc.imaging import gen_ppp_snapshot_mip
from imaging_helpers_hpc.loading import list_hdf_volume_datasets, load_any_volume
from imaging_helpers_hpc.paths import AnalysisOutputPaths

logger = logging.getLogger(__name__)

_SNAPSHOT_KEYS = (
    "volumes/raw",
    "volumes/pred_numinst",
    "volumes/gt_numinst",
    "volumes/pred_affs",
    "volumes/gt_affs",
)

_ITERATION_RE = re.compile(r"(\d+)$")


def _try_load(path: Path, dataset: str):
    available = set(list_hdf_volume_datasets(path))
    if dataset not in available:
        logger.warning("%s not in %s; available=%s", dataset, path, sorted(available))
        return None
    return load_any_volume(path, dataset=dataset, reduce_wide_channels=False)


def _iteration_key(path: Path) -> tuple[int, str]:
    match = _ITERATION_RE.search(path.stem)
    iteration = int(match.group(1)) if match else 10**18
    return (iteration, path.name)


def _exp_id(path: Path) -> str:
    """metrics/ppp/<exp_id>/... -> <exp_id>."""
    parts = path.parts
    idxs = [i for i, part in enumerate(parts) if part == "ppp"]
    if not idxs or idxs[-1] + 1 >= len(parts):
        raise ValueError(
            f"Cannot infer exp_id from {path}; "
            "expected metrics/ppp/<exp_id>/..."
        )
    return parts[idxs[-1] + 1]


def _snapshot_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("snapshot_*.hdf"), key=_iteration_key)
    if not files:
        raise ValueError(f"No snapshot_*.hdf files in {path}")
    return files


def _run_one_snapshot(
    path: Path,
    output_paths: AnalysisOutputPaths,
    output_path: Path | None = None,
) -> None:
    raw = _try_load(path, "volumes/raw")
    pred_numinst = _try_load(path, "volumes/pred_numinst")
    gt_numinst = _try_load(path, "volumes/gt_numinst")
    pred_affs = _try_load(path, "volumes/pred_affs")
    gt_affs = _try_load(path, "volumes/gt_affs")

    if all(v is None for v in (raw, pred_numinst, gt_numinst, pred_affs, gt_affs)):
        raise ValueError(
            f"No expected snapshot datasets {_SNAPSHOT_KEYS} found in {path}"
        )

    gen_ppp_snapshot_mip(
        raw,
        pred_numinst,
        gt_numinst,
        pred_affs,
        gt_affs,
        output_paths,
        title_prefix=path.stem,
        axis=0,
        output_path=output_path,
    )


def run_snapshot_mip(
    input_file: str | Path,
    output_paths: AnalysisOutputPaths,
) -> None:
    path = Path(input_file)
    files = _snapshot_files(path)
    if path.is_file():
        _run_one_snapshot(files[0], output_paths)
        return

    exp_id = _exp_id(path)
    out_dir = output_paths.output_root / f"{exp_id}_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d snapshot MIPs to %s", len(files), out_dir)

    failures: list[Path] = []
    for snapshot in files:
        png = out_dir / f"{snapshot.stem}.png"
        try:
            _run_one_snapshot(snapshot, output_paths, png)
        except Exception:
            logger.exception("Failed snapshot MIP for %s", snapshot)
            failures.append(snapshot)

    if failures and len(failures) == len(files):
        raise RuntimeError(
            f"All {len(files)} snapshot MIP(s) failed; first: {failures[0]}"
        )
    if failures:
        logger.warning(
            "%d of %d snapshot MIP(s) failed", len(failures), len(files)
        )
