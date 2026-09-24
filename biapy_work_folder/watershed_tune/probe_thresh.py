"""Stage 2.5 seed / growth threshold probe for watershed_tune.

Phase A: seed ladder with growth=auto.
Phase B: growth ladder at best-score seed if Phase A max mean_n_pred <= 1.5.

Example::

    python biapy_work_folder/watershed_tune/probe_thresh.py \\
      --setup biapy-aug-zarr-seunet-FDb-dice \\
      --run biapy_fdb_dice_l40s_wc_nopat_short_winner_valpred \\
      --out biapy_work_folder/watershed_tune/probe_out/fdb_dice_valpred
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUT = "biapy_work_folder/watershed_tune/probe_out/fdb_dice_valpred"

SEED_LADDER: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
GROWTH_LADDER: tuple[Any, ...] = ("auto", 0.10, 0.20, 0.30, 0.40, 0.50)
PHASE_B_MEAN_NPRED_MAX = 1.5
CSV_FIELDS = ("phase", "seed_th", "growth_th", *METRIC_KEYS)


def _fmt_th(v: Any) -> str:
    if isinstance(v, str):
        return v
    return f"{float(v):g}"


def _row(phase: str, seed_th: Any, growth_th: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "seed_th": _fmt_th(seed_th),
        "growth_th": _fmt_th(growth_th),
    }
    for k in METRIC_KEYS:
        row[k] = metrics[k]
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _best_by_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: float(r["score"]))


def _best_npred_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer max f1_0.3 among rows with mean_n_pred >= 2; else closest n_pred to n_true."""
    candidates = [r for r in rows if float(r["mean_n_pred"]) >= 2.0]
    if candidates:
        return max(candidates, key=lambda r: float(r["f1_0.3"]))
    return min(
        rows,
        key=lambda r: abs(float(r["n_pred"]) - float(r["n_true"])),
    )


def _recommend_seed_range(best_score_row: dict[str, Any], baseline_row: dict[str, Any]) -> str:
    best_seed = float(best_score_row["seed_th"])
    baseline_score = float(baseline_row["score"])
    best_score = float(best_score_row["score"])
    if best_score <= baseline_score and abs(best_seed - 0.05) < 1e-9:
        return (
            "STUCK under-seg: no seed beat baseline 0.05 on score. "
            "Do not run Ray yet — inspect Db channel MIPs / histograms "
            "and reconsider polarity or growth."
        )
    lo = max(0.05, best_seed - 0.1)
    hi = min(0.60, best_seed + 0.1)
    return f"Recommended Ray seed_db float range: uniform({lo:.2f}, {hi:.2f}) (clipped to [0.05, 0.60])"


def _write_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    phase_b_ran: bool,
    phase_b_reason: str,
) -> None:
    phase_a = [r for r in rows if r["phase"] == "A"]
    best_score = _best_by_score(rows)
    best_npred = _best_npred_row(rows)
    phase_a_best = _best_by_score(phase_a)
    baseline = next(r for r in phase_a if float(r["seed_th"]) == 0.05 and r["growth_th"] == "auto")
    max_mean_npred = max(float(r["mean_n_pred"]) for r in phase_a)

    if phase_b_ran:
        growth_best = _best_by_score([r for r in rows if r["phase"] == "B"])
        growth_note = (
            "Include growth_f floats in Ray (Phase B ran; see CSV). "
            f"Best Phase-B: seed={growth_best['seed_th']} growth={growth_best['growth_th']} "
            f"score={float(growth_best['score']):.6f}."
        )
    else:
        growth_note = (
            "Growth: keep auto as default for Ray; floats optional later (Phase B skipped)."
        )

    lines = [
        "Stage 2.5 watershed threshold probe summary",
        f"n_rows={len(rows)} phase_a={len(phase_a)}",
        f"phase_b_ran={phase_b_ran}",
        f"phase_b_reason={phase_b_reason}",
        f"phase_a_max_mean_n_pred={max_mean_npred:.4f}",
        "",
        "Best by score (all phases):",
        f"  phase={best_score['phase']} seed={best_score['seed_th']} "
        f"growth={best_score['growth_th']} score={float(best_score['score']):.6f} "
        f"pq_0.3={float(best_score['pq_0.3']):.6f} f1_0.3={float(best_score['f1_0.3']):.6f} "
        f"n_pred={best_score['n_pred']} n_true={best_score['n_true']}",
        "",
        "Best n_pred / detection pick:",
        f"  phase={best_npred['phase']} seed={best_npred['seed_th']} "
        f"growth={best_npred['growth_th']} score={float(best_npred['score']):.6f} "
        f"pq_0.3={float(best_npred['pq_0.3']):.6f} f1_0.3={float(best_npred['f1_0.3']):.6f} "
        f"mean_n_pred={float(best_npred['mean_n_pred']):.4f}",
        "",
        f"Baseline (seed=0.05, growth=auto) score={float(baseline['score']):.6f}",
        f"Phase-A best seed for Ray bracket: {phase_a_best['seed_th']}",
        _recommend_seed_range(phase_a_best, baseline),
        growth_note,
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    print("\n----- summary.txt -----")
    print(path.read_text())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", default=DEFAULT_SETUP)
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--out", default=DEFAULT_OUT, help="Output directory for CSV + summary")
    p.add_argument("--metric-weights", default=None)
    p.add_argument(
        "--force-phase-b",
        action="store_true",
        help="Always run growth ladder (ignore mean_n_pred gate)",
    )
    args = p.parse_args(argv)

    weights = parse_metric_weights(args.metric_weights)
    paths = resolve_paths(args.setup, args.run, repo_root=_REPO_ROOT)
    stems = list_sample_stems(paths.per_image_dir)
    inst = instance_seg_cfg(paths.cfg)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"setup={paths.setup} run={paths.run}")
    print(f"out={out_dir}")
    print(f"n_samples={len(stems)} stems={stems}")
    print(f"seed_ladder={SEED_LADDER}")

    rows: list[dict[str, Any]] = []

    # Phase A
    for seed in SEED_LADDER:
        seed_th = parse_thresh(seed)
        growth_th = parse_thresh("auto")
        tag = f"A_seed{_fmt_th(seed)}"
        metrics = evaluate_setting(
            paths=paths,
            stems=stems,
            inst=inst,
            seed_th=seed_th,
            growth_th=growth_th,
            weights=weights,
            tag=tag,
        )
        rows.append(_row("A", seed_th, growth_th, metrics))
        _write_csv(out_dir / "leaderboard.csv", rows)  # incremental

    max_mean_npred = max(float(r["mean_n_pred"]) for r in rows)
    phase_a_best = _best_by_score(rows)
    best_seed = parse_thresh(phase_a_best["seed_th"])

    phase_b_ran = False
    if args.force_phase_b or max_mean_npred <= PHASE_B_MEAN_NPRED_MAX:
        phase_b_reason = (
            f"max_mean_n_pred={max_mean_npred:.4f} <= {PHASE_B_MEAN_NPRED_MAX}"
            if not args.force_phase_b
            else f"--force-phase-b (max_mean_n_pred={max_mean_npred:.4f})"
        )
        print(f"\nPhase B: {phase_b_reason}; seed={_fmt_th(best_seed)}")
        for growth in GROWTH_LADDER:
            growth_th = parse_thresh(growth)
            # Skip duplicate of Phase-A best (seed, auto) already logged
            if growth_th == "auto" and abs(float(best_seed) - float(phase_a_best["seed_th"])) < 1e-12:
                # Still record explicitly as Phase B for clarity if not already identical row
                rows.append(_row("B", best_seed, growth_th, {k: phase_a_best[k] for k in METRIC_KEYS}))
                continue
            tag = f"B_seed{_fmt_th(best_seed)}_g{_fmt_th(growth)}"
            metrics = evaluate_setting(
                paths=paths,
                stems=stems,
                inst=inst,
                seed_th=best_seed,
                growth_th=growth_th,
                weights=weights,
                tag=tag,
            )
            rows.append(_row("B", best_seed, growth_th, metrics))
            _write_csv(out_dir / "leaderboard.csv", rows)
        phase_b_ran = True
    else:
        phase_b_reason = (
            f"skipped: max_mean_n_pred={max_mean_npred:.4f} > {PHASE_B_MEAN_NPRED_MAX}"
        )
        print(f"\nPhase B: {phase_b_reason}")

    _write_csv(out_dir / "leaderboard.csv", rows)
    _write_summary(
        out_dir / "summary.txt",
        rows=rows,
        phase_b_ran=phase_b_ran,
        phase_b_reason=phase_b_reason,
    )
    print(f"Wrote {out_dir / 'leaderboard.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
