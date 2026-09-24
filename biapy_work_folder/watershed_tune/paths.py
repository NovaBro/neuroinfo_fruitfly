"""Resolve setup/run → per_image dir, config YAML, and GT dir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WatershedPaths:
    """Resolved paths for one BiaPy setup/run."""

    repo_root: Path
    setup: str
    run: str
    per_image_dir: Path
    config_yaml: Path
    gt_dir: Path
    cfg: dict[str, Any]


def resolve_paths(
    setup: str,
    run: str,
    *,
    repo_root: Path | None = None,
) -> WatershedPaths:
    """Resolve per_image / config / GT for ``--setup`` / ``--run``.

    Raises
    ------
    FileNotFoundError
        If Stage-0 contract paths are missing or ``per_image`` has no ``*.zarr``.
    """
    root = (repo_root or _REPO_ROOT).resolve()
    per_image_dir = root / "metrics" / "biapy" / setup / "results" / f"{setup}_{run}" / "per_image"
    config_yaml = root / "biapy_work_folder" / "configs" / f"{run}.yaml"

    if not config_yaml.is_file():
        raise FileNotFoundError(f"Missing run config YAML: {config_yaml}")
    if not per_image_dir.is_dir():
        raise FileNotFoundError(
            f"Missing per_image dir (Stage 0 valpred required): {per_image_dir}"
        )

    zarrs = sorted(p for p in per_image_dir.iterdir() if p.is_dir() and p.suffix == ".zarr")
    if not zarrs:
        raise FileNotFoundError(f"No *.zarr channel preds in {per_image_dir}")

    with config_yaml.open() as f:
        cfg = yaml.safe_load(f)

    try:
        gt_rel = cfg["DATA"]["TEST"]["GT_PATH"]
    except (KeyError, TypeError) as exc:
        raise FileNotFoundError(
            f"Config missing DATA.TEST.GT_PATH: {config_yaml}"
        ) from exc

    gt_dir = Path(gt_rel)
    if not gt_dir.is_absolute():
        gt_dir = root / gt_dir
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing GT dir: {gt_dir}")

    return WatershedPaths(
        repo_root=root,
        setup=setup,
        run=run,
        per_image_dir=per_image_dir,
        config_yaml=config_yaml,
        gt_dir=gt_dir,
        cfg=cfg,
    )


def list_sample_stems(per_image_dir: Path) -> list[str]:
    """Return sorted sample stems from ``*.zarr`` basenames under ``per_image_dir``."""
    stems = sorted(
        p.stem for p in per_image_dir.iterdir() if p.is_dir() and p.suffix == ".zarr"
    )
    if not stems:
        raise FileNotFoundError(f"No *.zarr channel preds in {per_image_dir}")
    return stems


def instance_seg_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return ``PROBLEM.INSTANCE_SEG`` block from a BiaPy YAML dict."""
    try:
        return cfg["PROBLEM"]["INSTANCE_SEG"]
    except (KeyError, TypeError) as exc:
        raise KeyError("Config missing PROBLEM.INSTANCE_SEG") from exc
