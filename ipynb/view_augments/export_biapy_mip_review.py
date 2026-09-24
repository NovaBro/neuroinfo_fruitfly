#!/usr/bin/env python3
"""Headless export of augmentation5/6 BiaPy MIP review grids to PNG.

Reuses ``plot_per_image_predictions`` (pred channels) and
``plot_image_grid`` + ``mip_biapy_gt_instance`` (instance labels) from
``metric_review.py``. Run from the repo root (or let this script chdir).

Default: FDb dice + skel Ray Tune wc_nopat_short winners →
``imaging_helpers_hpc/output/<config>/<run>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_repo_root() -> Path:
    """Ensure cwd is repo root and ``ipynb/`` is importable (notebook-style)."""
    here = Path(__file__).resolve()
    # .../neuroinfo_fruitfly/ipynb/view_augments/export_biapy_mip_review.py
    root = here.parents[2]
    os.chdir(root)
    ipynb = root / "ipynb"
    if str(ipynb) not in sys.path:
        sys.path.insert(0, str(ipynb))
    return root


# Agg before any pyplot import via metric_review.
import matplotlib

matplotlib.use("Agg")

_ROOT = _bootstrap_repo_root()

from view_augments.metric_review import (  # noqa: E402
    get_metric_paths,
    mip_biapy_gt_instance,
    plot_image_grid,
    plot_per_image_predictions,
)
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_PAIRS = (
    (
        "biapy-aug-zarr-seunet-FDb-dice",
        "biapy_fdb_dice_l40s_wc_nopat_short_winner",
    ),
    (
        "biapy-aug-zarr-seunet-FDb-skel",
        "biapy_fdb_skel_l40s_wc_nopat_short_winner",
    ),
)
DEFAULT_CHANNELS = ("F", "Db")
DEFAULT_THRESHOLDS = (0.05, 0.5, 0.95)
DEFAULT_OUT = Path("imaging_helpers_hpc/output")


def _parse_pair(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"expected config:run, got {text!r}"
        )
    config, run = text.split(":", 1)
    config, run = config.strip(), run.strip()
    if not config or not run:
        raise argparse.ArgumentTypeError(
            f"expected non-empty config:run, got {text!r}"
        )
    return config, run


def _save_fig(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _channel_threshold_dict(
    channel_names: list[str], thr: float
) -> dict[str, float]:
    return {name: float(thr) for name in channel_names}


def export_pred_channels(
    config: str,
    run: str,
    out_dir: Path,
    *,
    channel_names: list[str],
    thresholds: list[float],
    n_samples: int | None,
    dpi: int,
    figsize_cell: tuple[float, float],
) -> None:
    fig, _ = plot_per_image_predictions(
        config,
        run=run,
        n_samples=n_samples,
        threshold=None,
        figsize_cell=figsize_cell,
        channel_names=channel_names,
        dpi=dpi,
    )
    _save_fig(fig, out_dir / "per_image_raw.png", dpi)

    for thr in thresholds:
        fig, _ = plot_per_image_predictions(
            config,
            run=run,
            n_samples=n_samples,
            threshold=_channel_threshold_dict(channel_names, thr),
            figsize_cell=figsize_cell,
            channel_names=channel_names,
            dpi=dpi,
        )
        # Keep filenames simple / shell-friendly (0.05 → thr0.05).
        thr_tag = f"{thr:g}"
        _save_fig(fig, out_dir / f"per_image_thr{thr_tag}.png", dpi)


def export_instances(
    config: str,
    run: str,
    out_dir: Path,
    *,
    n_samples: int | None,
    dpi: int,
    figsize_cell: tuple[float, float],
    ncols: int,
) -> None:
    for sub_folder, filename in (
        ("per_image_instances", "instances.png"),
        ("per_image_post_processing", "instances_post.png"),
    ):
        paths = sorted(get_metric_paths(sub_folder, config, run))
        if not paths:
            raise FileNotFoundError(
                f"No .tif files in metrics/biapy/{config}/results/"
                f"{config}_{run}/{sub_folder}"
            )
        if n_samples is not None:
            paths = paths[: max(0, int(n_samples))]
            if not paths:
                raise ValueError("n_samples resolved to an empty path list")
        fig, _ = plot_image_grid(
            paths,
            mip_biapy_gt_instance,
            ncols=ncols,
            figsize_cell=figsize_cell,
            dpi=dpi,
            font_siz=6,
        )
        _save_fig(fig, out_dir / filename, dpi)


def export_run(
    config: str,
    run: str,
    out_root: Path,
    *,
    channel_names: list[str],
    thresholds: list[float],
    n_samples: int | None,
    dpi: int,
    skip_pred: bool,
    skip_instances: bool,
) -> None:
    out_dir = out_root / config / run
    print(f"=== {config} / {run} → {out_dir} ===")
    if not skip_pred:
        export_pred_channels(
            config,
            run,
            out_dir,
            channel_names=channel_names,
            thresholds=thresholds,
            n_samples=n_samples,
            dpi=dpi,
            figsize_cell=(3.0, 3.0),
        )
    if not skip_instances:
        export_instances(
            config,
            run,
            out_dir,
            n_samples=n_samples,
            dpi=dpi,
            figsize_cell=(4.0, 4.0),
            ncols=4,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export augmentation5/6 BiaPy MIP review grids as PNGs "
            "(headless Agg backend)."
        )
    )
    p.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        dest="pairs",
        metavar="CONFIG:RUN",
        help=(
            "Config/run to export (repeatable). Default: FDb dice + skel "
            "wc_nopat_short winners."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output root (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Only plot the first N samples (faster preview).",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_THRESHOLDS),
        help=(
            "Per-channel threshold sweep for pred MIPs "
            f"(default: {list(DEFAULT_THRESHOLDS)}). Pass nothing after "
            "the flag to skip the sweep (raw only)."
        ),
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=list(DEFAULT_CHANNELS),
        help=f"BiaPy DATA_CHANNELS codes (default: {list(DEFAULT_CHANNELS)}).",
    )
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument(
        "--skip-pred",
        action="store_true",
        help="Skip augmentation5 per_image channel grids.",
    )
    p.add_argument(
        "--skip-instances",
        action="store_true",
        help="Skip augmentation6 instance / post-processing grids.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = args.pairs if args.pairs else list(DEFAULT_PAIRS)
    out_root = args.out if args.out.is_absolute() else _ROOT / args.out

    for config, run in pairs:
        export_run(
            config,
            run,
            out_root,
            channel_names=list(args.channels),
            thresholds=list(args.thresholds),
            n_samples=args.n_samples,
            dpi=args.dpi,
            skip_pred=args.skip_pred,
            skip_instances=args.skip_instances,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
