"""FDcDn-specific watershed threshold probe (separate Dc / Dn seeds).

Unlike ``probe_thresh.py`` (FDb single-seed broadcast), this script treats
``SEED_CHANNELS: ['Dc', 'Dn']`` independently and includes:

* Phase 0 — official YAML baseline (Dc=auto, Dn=0.05, growth=auto)
* Phase 1 — tune Dn with Dc=auto
* Phase 2 — tune Dc with Dn=0.05
* Phase 3 — compact 2D grid around Phase-1/2 winners
* Phase 4 — growth F ladder at best seed pair
* Phase 5 — inverted Dc/Dn diagnostic (``1 - channel``, still BiaPy ``<=``)

Example::

    python biapy_work_folder/watershed_tune/probe_fcdcdn.py \\
      --setup biapy-aug-zarr-seunet-FDcDn-skel \\
      --run biapy_fcdcdn_skel_l40s_c3_winner_valpred \\
      --out biapy_work_folder/watershed_tune/probe_out/fcdcdn_skel_c3_valpred_fdccdn
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biapy_work_folder.watershed_tune.load import (  # noqa: E402
    channel_index,
    load_channel_pred,
    load_gt_instances,
)
from biapy_work_folder.watershed_tune.paths import (  # noqa: E402
    instance_seg_cfg,
    list_sample_stems,
    resolve_paths,
)
from biapy_work_folder.watershed_tune.run_ws import parse_thresh, run_watershed  # noqa: E402
from biapy_work_folder.watershed_tune.score import (  # noqa: E402
    METRIC_KEYS,
    MetricWeights,
    evaluate_dataset,
    parse_metric_weights,
    score_volume,
)

DEFAULT_SETUP = "biapy-aug-zarr-seunet-FDcDn-skel"
DEFAULT_RUN = "biapy_fcdcdn_skel_l40s_c3_winner_valpred"
DEFAULT_OUT = "biapy_work_folder/watershed_tune/probe_out/fcdcdn_skel_c3_valpred_fdccdn"

# Phase ladders (plan).
DN_LADDER: tuple[Any, ...] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
DC_LADDER: tuple[Any, ...] = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, "auto")
GROWTH_LADDER: tuple[Any, ...] = ("auto", 0.10, 0.20, 0.30, 0.40, 0.50)
INVERT_DC: tuple[Any, ...] = (0.10, 0.30, 0.50)
INVERT_DN: tuple[Any, ...] = (0.05, 0.10)

CSV_FIELDS = (
    "phase",
    "polarity",
    "dc_th",
    "dn_th",
    "growth_th",
    *METRIC_KEYS,
)

# Failure-mode references from prior FDb / FDcDn test experience.
FDB_UNDERSEG_MEAN_NPRED = 1.0
FCDCDN_OVERSEG_MEAN_NPRED = 96.0


def _fmt_th(v: Any) -> str:
    if isinstance(v, str):
        return v
    return f"{float(v):g}"


def _parse_th_token(v: Any):
    return parse_thresh(v)


def _assert_dc_dn_seeds(inst: dict[str, Any]) -> None:
    seeds = list(inst["WATERSHED"]["SEED_CHANNELS"])
    if seeds != ["Dc", "Dn"]:
        raise SystemExit(
            f"Expected WATERSHED.SEED_CHANNELS == ['Dc', 'Dn'], got {seeds}. "
            "This probe is FDcDn-specific."
        )


def _invert_dc_dn(pred: np.ndarray, channel_names: list[str]) -> np.ndarray:
    """Return a copy with Dc and Dn channels inverted as ``1 - x``."""
    out = np.array(pred, dtype=np.float32, copy=True)
    for name in ("Dc", "Dn"):
        idx = channel_index(channel_names, name)
        out[..., idx] = 1.0 - out[..., idx]
    return out


def _evaluate_setting(
    *,
    paths,
    stems: list[str],
    inst: dict[str, Any],
    dc_th,
    dn_th,
    growth_th,
    weights: MetricWeights,
    tag: str,
    invert_dc_dn: bool = False,
) -> dict[str, Any]:
    """Watershed + matching for one (dc, dn, growth) setting."""
    polarity = "inverted" if invert_dc_dn else "normal"
    seed_ths = [dc_th, dn_th]
    print(
        f"=== setting {tag}: polarity={polarity} "
        f"dc={_fmt_th(dc_th)!r} dn={_fmt_th(dn_th)!r} growth={_fmt_th(growth_th)!r} ===",
        flush=True,
    )
    channel_names = list(inst["DATA_CHANNELS"])
    stats_list = []
    for i, stem in enumerate(stems):
        print(f"  [{i + 1}/{len(stems)}] {stem}: load ...", flush=True)
        pred = load_channel_pred(paths.per_image_dir, stem)
        if invert_dc_dn:
            pred = _invert_dc_dn(pred, channel_names)
        gt = load_gt_instances(paths.gt_dir, stem)
        if pred.shape[:3] != gt.shape:
            raise SystemExit(
                f"Spatial mismatch for {stem}: pred {pred.shape[:3]} vs gt {gt.shape}"
            )
        print(f"  [{i + 1}/{len(stems)}] {stem}: watershed ...", flush=True)
        labels = run_watershed(
            pred,
            inst,
            seed_ths=seed_ths,
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
        f"n_pred={metrics['n_pred']} n_true={metrics['n_true']} ===",
        flush=True,
    )
    return metrics


def _row(
    phase: str,
    polarity: str,
    dc_th: Any,
    dn_th: Any,
    growth_th: Any,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "polarity": polarity,
        "dc_th": _fmt_th(dc_th),
        "dn_th": _fmt_th(dn_th),
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


def _best_by_score(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: float(r["score"]))


def _best_count_balanced(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prefer rows with mean_n_pred near mean_n_true; break ties by score."""
    return min(
        rows,
        key=lambda r: (
            abs(float(r["mean_n_pred"]) - float(r["mean_n_true"])),
            -float(r["score"]),
        ),
    )


def _neighbors(ladder: Sequence[Any], best: Any, k: int = 3) -> list[Any]:
    """Return up to ``k`` ladder values centered on ``best`` (by index)."""
    best_s = _fmt_th(best)
    idxs = [i for i, v in enumerate(ladder) if _fmt_th(v) == best_s]
    if not idxs:
        # best may be from another phase; prefer closest float, else prepend best
        if _fmt_th(best) == "auto" or isinstance(best, str):
            out: list[Any] = [best]
            for v in ladder:
                if not any(_th_eq(v, x) for x in out):
                    out.append(v)
                if len(out) >= k:
                    break
            return out[:k]
        best_f = float(best)
        ranked = sorted(
            (
                (abs(float(v) - best_f) if not isinstance(v, str) else 1e9, i, v)
                for i, v in enumerate(ladder)
            )
        )
        return [v for _, _, v in ranked[:k]]

    i = idxs[0]
    half = k // 2
    start = max(0, i - half)
    end = min(len(ladder), start + k)
    start = max(0, end - k)
    return list(ladder[start:end])


def _th_eq(a: Any, b: Any) -> bool:
    sa, sb = _fmt_th(a), _fmt_th(b)
    if sa == sb:
        return True
    try:
        return abs(float(sa) - float(sb)) < 1e-12
    except ValueError:
        return False


def _recommend(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> str:
    normal = [r for r in rows if r["polarity"] == "normal"]
    inverted = [r for r in rows if r["polarity"] == "inverted"]
    best_n = _best_by_score(normal) if normal else None
    best_i = _best_by_score(inverted) if inverted else None

    candidates = [c for c in (best_n, best_i) if c is not None]
    if not candidates:
        return "No rows to recommend — check probe failed early."

    winner = max(candidates, key=lambda r: float(r["score"]))
    base_score = float(baseline["score"])
    base_pq = float(baseline["pq_0.3"])
    base_f1 = float(baseline["f1_0.3"])
    base_n = float(baseline["mean_n_pred"])
    base_true = float(baseline["mean_n_true"])

    w_score = float(winner["score"])
    w_pq = float(winner["pq_0.3"])
    w_f1 = float(winner["f1_0.3"])
    w_n = float(winner["mean_n_pred"])

    better_score = w_score > base_score
    better_quality = (w_pq > base_pq and w_pq > 0) or (w_f1 > base_f1 and w_f1 > 0)
    closer_count = abs(w_n - base_true) < abs(base_n - base_true)
    not_fdb_under = abs(w_n - FDB_UNDERSEG_MEAN_NPRED) > abs(w_n - base_true)
    not_test_over = abs(w_n - FCDCDN_OVERSEG_MEAN_NPRED) > abs(w_n - base_true)
    count_ok = closer_count and not_fdb_under and not_test_over

    if better_score and better_quality and count_ok:
        return (
            f"PROCEED to Ray: best {winner['polarity']} row "
            f"(dc={winner['dc_th']}, dn={winner['dn_th']}, growth={winner['growth_th']}) "
            f"beats baseline on score/quality with more sensible n_pred "
            f"(mean_n_pred={w_n:.2f} vs n_true={base_true:.2f})."
        )
    return (
        "STOP before Ray: no row clearly beats official baseline with "
        "nonzero PQ/F1 improvement and sensible instance counts. "
        "Inspect Dc/Dn GT-vs-pred polarity / channel quality next."
    )


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    baseline = next(
        r
        for r in rows
        if r["phase"] == "0"
        and r["polarity"] == "normal"
        and r["dc_th"] == "auto"
        and r["dn_th"] == "0.05"
        and r["growth_th"] == "auto"
    )
    normal = [r for r in rows if r["polarity"] == "normal"]
    inverted = [r for r in rows if r["polarity"] == "inverted"]
    best_normal = _best_by_score(normal)
    best_inverted = _best_by_score(inverted) if inverted else None
    best_count = _best_count_balanced(normal)

    def _fmt_row(label: str, r: dict[str, Any] | None) -> list[str]:
        if r is None:
            return [f"{label}: (none)"]
        return [
            f"{label}:",
            f"  phase={r['phase']} polarity={r['polarity']} "
            f"dc={r['dc_th']} dn={r['dn_th']} growth={r['growth_th']}",
            f"  score={float(r['score']):.6f} pq_0.3={float(r['pq_0.3']):.6f} "
            f"f1_0.3={float(r['f1_0.3']):.6f}",
            f"  n_pred={r['n_pred']} n_true={r['n_true']} "
            f"mean_n_pred={float(r['mean_n_pred']):.4f} "
            f"mean_n_true={float(r['mean_n_true']):.4f}",
        ]

    lines: list[str] = [
        "FDcDn-specific watershed threshold probe summary",
        f"n_rows={len(rows)}",
        "",
        *_fmt_row("Official baseline (Phase 0)", baseline),
        "",
        *_fmt_row("Best normal-polarity", best_normal),
        "",
        *_fmt_row("Best inverted-polarity diagnostic", best_inverted),
        "",
        *_fmt_row("Best count-balanced (normal)", best_count),
        "",
        _recommend(rows, baseline),
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    print("\n----- summary.txt -----")
    print(path.read_text())


def _run_row(
    *,
    paths,
    stems,
    inst,
    weights,
    rows: list[dict[str, Any]],
    out_csv: Path,
    phase: str,
    polarity: str,
    dc_th: Any,
    dn_th: Any,
    growth_th: Any,
    seen: set[tuple[str, str, str, str]],
    max_rows: int | None,
) -> bool:
    """Evaluate one row; return False if max_rows reached after append."""
    key = (polarity, _fmt_th(dc_th), _fmt_th(dn_th), _fmt_th(growth_th))
    if key in seen:
        return True
    if max_rows is not None and len(rows) >= max_rows:
        return False

    tag = (
        f"P{phase}_{polarity}_dc{_fmt_th(dc_th)}_dn{_fmt_th(dn_th)}_g{_fmt_th(growth_th)}"
    )
    metrics = _evaluate_setting(
        paths=paths,
        stems=stems,
        inst=inst,
        dc_th=_parse_th_token(dc_th),
        dn_th=_parse_th_token(dn_th),
        growth_th=_parse_th_token(growth_th),
        weights=weights,
        tag=tag,
        invert_dc_dn=(polarity == "inverted"),
    )
    rows.append(_row(phase, polarity, dc_th, dn_th, growth_th, metrics))
    seen.add(key)
    _write_csv(out_csv, rows)
    return max_rows is None or len(rows) < max_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", default=DEFAULT_SETUP)
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--out", default=DEFAULT_OUT, help="Output directory for CSV + summary")
    p.add_argument("--metric-weights", default=None)
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after N evaluated rows (smoke / partial runs).",
    )
    p.add_argument(
        "--skip-invert",
        action="store_true",
        help="Skip Phase 5 inverted-polarity diagnostic.",
    )
    args = p.parse_args(argv)

    weights = parse_metric_weights(args.metric_weights)
    paths = resolve_paths(args.setup, args.run, repo_root=_REPO_ROOT)
    stems = list_sample_stems(paths.per_image_dir)
    inst = instance_seg_cfg(paths.cfg)
    _assert_dc_dn_seeds(inst)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "leaderboard.csv"

    print(f"setup={paths.setup} run={paths.run}")
    print(f"out={out_dir}")
    print(f"n_samples={len(stems)} stems={stems}")
    print(f"dn_ladder={DN_LADDER}")
    print(f"dc_ladder={DC_LADDER}")
    if args.max_rows is not None:
        print(f"max_rows={args.max_rows}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def go(phase, polarity, dc, dn, growth) -> bool:
        return _run_row(
            paths=paths,
            stems=stems,
            inst=inst,
            weights=weights,
            rows=rows,
            out_csv=out_csv,
            phase=phase,
            polarity=polarity,
            dc_th=dc,
            dn_th=dn,
            growth_th=growth,
            seen=seen,
            max_rows=args.max_rows,
        )

    # Phase 0 — official baseline
    if not go("0", "normal", "auto", 0.05, "auto"):
        _write_summary(out_dir / "summary.txt", rows)
        print(f"Wrote {out_csv} ({len(rows)} rows) [max-rows stop]")
        return 0

    # Phase 1 — tune Dn, Dc=auto
    for dn in DN_LADDER:
        if not go("1", "normal", "auto", dn, "auto"):
            _write_summary(out_dir / "summary.txt", rows)
            print(f"Wrote {out_csv} ({len(rows)} rows) [max-rows stop]")
            return 0

    # Phase 2 — tune Dc, Dn=0.05
    for dc in DC_LADDER:
        if not go("2", "normal", dc, 0.05, "auto"):
            _write_summary(out_dir / "summary.txt", rows)
            print(f"Wrote {out_csv} ({len(rows)} rows) [max-rows stop]")
            return 0

    phase1 = [r for r in rows if r["phase"] == "1"]
    phase2 = [r for r in rows if r["phase"] == "2"]
    best_dn_row = _best_by_score(phase1)
    best_dc_row = _best_by_score(phase2)
    best_dn = _parse_th_token(best_dn_row["dn_th"])
    best_dc = _parse_th_token(best_dc_row["dc_th"])
    print(
        f"\nPhase 3 seeds from Phase1/2 winners: "
        f"best_dc={_fmt_th(best_dc)} best_dn={_fmt_th(best_dn)}",
        flush=True,
    )

    dc_cands = _neighbors(DC_LADDER, best_dc, k=3)
    dn_cands = _neighbors(DN_LADDER, best_dn, k=3)
    # Ensure winners themselves are included
    if not any(_th_eq(v, best_dc) for v in dc_cands):
        dc_cands = [best_dc, *dc_cands][:3]
    if not any(_th_eq(v, best_dn) for v in dn_cands):
        dn_cands = [best_dn, *dn_cands][:3]

    for dc in dc_cands:
        for dn in dn_cands:
            if not go("3", "normal", dc, dn, "auto"):
                _write_summary(out_dir / "summary.txt", rows)
                print(f"Wrote {out_csv} ({len(rows)} rows) [max-rows stop]")
                return 0

    phase3 = [r for r in rows if r["phase"] == "3"] or [
        r for r in rows if r["polarity"] == "normal"
    ]
    best_seed = _best_by_score(phase3)
    seed_dc = _parse_th_token(best_seed["dc_th"])
    seed_dn = _parse_th_token(best_seed["dn_th"])
    print(
        f"\nPhase 4 growth sweep at dc={_fmt_th(seed_dc)} dn={_fmt_th(seed_dn)}",
        flush=True,
    )

    for growth in GROWTH_LADDER:
        if not go("4", "normal", seed_dc, seed_dn, growth):
            _write_summary(out_dir / "summary.txt", rows)
            print(f"Wrote {out_csv} ({len(rows)} rows) [max-rows stop]")
            return 0

    # Phase 5 — inverted polarity diagnostic
    if not args.skip_invert:
        print("\nPhase 5: inverted Dc/Dn diagnostic (1 - channel)", flush=True)
        for dc in INVERT_DC:
            for dn in INVERT_DN:
                if not go("5", "inverted", dc, dn, "auto"):
                    _write_summary(out_dir / "summary.txt", rows)
                    print(f"Wrote {out_csv} ({len(rows)} rows) [max-rows stop]")
                    return 0

    _write_csv(out_csv, rows)
    _write_summary(out_dir / "summary.txt", rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
