"""Shared sequential watershed + matching evaluation over a sample list."""

from __future__ import annotations

import gc
from typing import Any

from biapy_work_folder.watershed_tune.load import load_channel_pred, load_gt_instances
from biapy_work_folder.watershed_tune.run_ws import run_watershed
from biapy_work_folder.watershed_tune.score import MetricWeights, evaluate_dataset, score_volume


def evaluate_setting(
    *,
    paths,
    stems: list[str],
    inst: dict[str, Any],
    seed_th,
    growth_th,
    weights: MetricWeights,
    tag: str,
) -> dict[str, Any]:
    """Run watershed + matching on every stem; return full metric schema dict."""
    print(f"=== setting {tag}: seed_th={seed_th!r} growth_th={growth_th!r} ===")
    stats_list = []
    for i, stem in enumerate(stems):
        print(f"  [{i + 1}/{len(stems)}] {stem}: load ...", flush=True)
        pred = load_channel_pred(paths.per_image_dir, stem)
        gt = load_gt_instances(paths.gt_dir, stem)
        if pred.shape[:3] != gt.shape:
            raise SystemExit(
                f"Spatial mismatch for {stem}: pred {pred.shape[:3]} vs gt {gt.shape}"
            )
        print(f"  [{i + 1}/{len(stems)}] {stem}: watershed ...", flush=True)
        labels = run_watershed(
            pred,
            inst,
            seed_ths=seed_th,
            growth_ths=growth_th,
            verbose=False,
        )
        del pred
        print(f"  [{i + 1}/{len(stems)}] {stem}: matching ...", flush=True)
        stats_list.append(score_volume(gt, labels))
        del labels, gt
        gc.collect()

    metrics = evaluate_dataset(stats_list, weights=weights)
    print(
        f"=== {tag} done: score={metrics['score']:.6f} "
        f"pq_0.3={metrics['pq_0.3']:.6f} f1_0.3={metrics['f1_0.3']:.6f} "
        f"n_pred={metrics['n_pred']} n_true={metrics['n_true']} ==="
    )
    return metrics
