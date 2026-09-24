"""Evaluate watershed thresholds on all val volumes; print multi-metric suite.

Examples
--------
Exit-gate compare (baseline 0.05 vs strict 0.7)::

    python biapy_work_folder/watershed_tune/eval_val.py \\
      --setup biapy-aug-zarr-seunet-FDb-dice \\
      --run biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred \\
      --compare-baseline --strict-seed 0.7

Single setting::

    python biapy_work_folder/watershed_tune/eval_val.py \\
      --seed-th 0.05 --growth-th auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biapy_work_folder.watershed_tune.evaluate import evaluate_setting  # noqa: E402
from biapy_work_folder.watershed_tune.paths import (  # noqa: E402
    instance_seg_cfg,
    list_sample_stems,
    resolve_paths,
)
from biapy_work_folder.watershed_tune.run_ws import parse_thresh  # noqa: E402
from biapy_work_folder.watershed_tune.score import (  # noqa: E402
    METRIC_KEYS,
    parse_metric_weights,
)

DEFAULT_SETUP = "biapy-aug-zarr-seunet-FDb-dice"
DEFAULT_RUN = "biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred"


def _print_side_by_side(results: dict[str, dict[str, Any]]) -> None:
    names = list(results.keys())
    col_w = max(14, max(len(n) for n in names) + 2)
    header = f"{'metric':<22}" + "".join(f"{n:>{col_w}}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for key in METRIC_KEYS:
        row = f"{key:<22}"
        for n in names:
            v = results[n][key]
            if key in ("n_pred", "n_true"):
                row += f"{int(v):>{col_w}d}"
            else:
                row += f"{float(v):>{col_w}.6f}"
        print(row)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", default=DEFAULT_SETUP)
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--seed-th", default="0.05", help="Used when not --compare-baseline")
    p.add_argument("--growth-th", default="auto")
    p.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare seed 0.05 vs --strict-seed (growth=auto both)",
    )
    p.add_argument("--strict-seed", default="0.7", help="Stricter seed for --compare-baseline")
    p.add_argument(
        "--metric-weights",
        default=None,
        help="Override weights, e.g. pq03=1,pq05=0.5,f103=0.5,f105=0.25,count=0.25",
    )
    args = p.parse_args(argv)

    weights = parse_metric_weights(args.metric_weights)
    paths = resolve_paths(args.setup, args.run, repo_root=_REPO_ROOT)
    stems = list_sample_stems(paths.per_image_dir)
    inst = instance_seg_cfg(paths.cfg)

    print(f"setup={paths.setup} run={paths.run}")
    print(f"per_image={paths.per_image_dir}")
    print(f"gt_dir={paths.gt_dir}")
    print(f"n_samples={len(stems)} stems={stems}")
    print(f"weights={weights}")

    results: dict[str, dict[str, Any]] = {}
    if args.compare_baseline:
        settings = [
            ("baseline", parse_thresh(0.05), parse_thresh("auto")),
            ("strict", parse_thresh(args.strict_seed), parse_thresh("auto")),
        ]
    else:
        settings = [
            ("run", parse_thresh(args.seed_th), parse_thresh(args.growth_th)),
        ]

    for tag, seed_th, growth_th in settings:
        results[tag] = evaluate_setting(
            paths=paths,
            stems=stems,
            inst=inst,
            seed_th=seed_th,
            growth_th=growth_th,
            weights=weights,
            tag=tag,
        )

    _print_side_by_side(results)

    if args.compare_baseline and "baseline" in results and "strict" in results:
        b, s = results["baseline"]["score"], results["strict"]["score"]
        if s > b:
            print(f"\nEXIT_GATE: PASS  (strict score {s:.6f} > baseline {b:.6f})")
        else:
            print(
                f"\nEXIT_GATE: DOCUMENT  "
                f"(strict score {s:.6f} <= baseline {b:.6f}; "
                f"investigate before Ray)"
            )
    else:
        print("\nEXIT_GATE: N/A (single setting)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
