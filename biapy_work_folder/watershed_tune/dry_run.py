"""CLI smoke: disk → watershed_by_channels → n_pred / n_true on one volume.

Examples
--------
Single threshold::

    python biapy_work_folder/watershed_tune/dry_run.py \\
      --setup biapy-aug-zarr-seunet-FDb-dice \\
      --run biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred \\
      --sample R22C03-20180918_66_J2 \\
      --seed-th 0.05 --growth-th auto

Exit-gate compare (baseline 0.05 vs stricter 0.4)::

    python biapy_work_folder/watershed_tune/dry_run.py \\
      --setup biapy-aug-zarr-seunet-FDb-dice \\
      --run biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred \\
      --compare-baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biapy_work_folder.watershed_tune.load import (  # noqa: E402
    count_instances,
    load_channel_pred,
    load_gt_instances,
)
from biapy_work_folder.watershed_tune.paths import (  # noqa: E402
    instance_seg_cfg,
    list_sample_stems,
    resolve_paths,
)
from biapy_work_folder.watershed_tune.run_ws import parse_thresh, run_watershed  # noqa: E402

DEFAULT_SETUP = "biapy-aug-zarr-seunet-FDb-dice"
DEFAULT_RUN = "biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred"
DEFAULT_SAMPLE = "R22C03-20180918_66_J2"


def _print_result(tag: str, labels, n_true: int, seed_th, growth_th) -> None:
    n_pred = count_instances(labels)
    print(
        f"[{tag}] seed_th={seed_th!r} growth_th={growth_th!r} "
        f"labels.shape={tuple(labels.shape)} n_pred={n_pred} n_true={n_true}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", default=DEFAULT_SETUP, help="BiaPy job / setup name")
    p.add_argument("--run", default=DEFAULT_RUN, help="Run id (config stem + results suffix)")
    p.add_argument(
        "--sample",
        default=DEFAULT_SAMPLE,
        help="Val sample stem under per_image/ (default: first val stem)",
    )
    p.add_argument("--seed-th", default="0.05", help="Seed threshold float or 'auto'")
    p.add_argument("--growth-th", default="auto", help="Growth-mask threshold float or 'auto'")
    p.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Run seed 0.05 vs 0.4 with growth=auto (Stage 1 exit gate)",
    )
    p.add_argument(
        "--verbose-ws",
        action="store_true",
        help="Pass verbose=True to watershed_by_channels",
    )
    args = p.parse_args(argv)

    paths = resolve_paths(args.setup, args.run, repo_root=_REPO_ROOT)
    stems = list_sample_stems(paths.per_image_dir)
    if args.sample not in stems:
        raise SystemExit(
            f"Sample {args.sample!r} not in {paths.per_image_dir} "
            f"(available: {stems})"
        )

    inst = instance_seg_cfg(paths.cfg)
    print(f"setup={paths.setup} run={paths.run}")
    print(f"per_image={paths.per_image_dir}")
    print(f"gt_dir={paths.gt_dir}")
    print(f"DATA_CHANNELS={inst.get('DATA_CHANNELS')}")
    print(f"WATERSHED={inst.get('WATERSHED')}")

    print(f"Loading pred + GT for {args.sample} ...")
    pred = load_channel_pred(paths.per_image_dir, args.sample)
    gt = load_gt_instances(paths.gt_dir, args.sample)
    n_true = count_instances(gt)
    print(f"pred.shape={tuple(pred.shape)} gt.shape={tuple(gt.shape)} n_true={n_true}")

    if pred.shape[:3] != gt.shape:
        raise SystemExit(
            f"Spatial shape mismatch: pred {pred.shape[:3]} vs gt {gt.shape}"
        )

    if args.compare_baseline:
        trials = [
            ("baseline", parse_thresh(0.05), parse_thresh("auto")),
            ("stricter", parse_thresh(0.4), parse_thresh("auto")),
        ]
    else:
        trials = [
            ("run", parse_thresh(args.seed_th), parse_thresh(args.growth_th)),
        ]

    for tag, seed_th, growth_th in trials:
        print(f"Running watershed ({tag}) ...")
        labels = run_watershed(
            pred,
            inst,
            seed_ths=seed_th,
            growth_ths=growth_th,
            verbose=args.verbose_ws,
        )
        _print_result(tag, labels, n_true, seed_th, growth_th)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
