#!/usr/bin/env python3
"""Diagnose PPP instance sizes + pred_affs / pred_numinst activity.

Reads existing instanced HDF + processed pred zarr; prints a short report.
Defaults target ppp_basic_long8h ckpt 30000 @ VI th=0.3 (Layer A). Override
--hdf/--zarr/--report for other exps (e.g. ppp_basic_8h).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HDF = (
    REPO
    / "metrics/ppp/ppp_basic_long8h/test/instanced/30000"
    / "patch_threshold_0_3/fc_threshold_0_3/mws_True"
    / "skeletonize_foreground_True/numinst_threshs__0_9_0_1_"
    / "R22C03-20180918_66_J2.hdf"
)
DEFAULT_ZARR = (
    REPO
    / "metrics/ppp/ppp_basic_long8h/test/processed/30000"
    / "R22C03-20180918_66_J2.zarr"
)
DEFAULT_REPORT = (
    REPO / "metrics/ppp/ppp_basic_long8h/diagnose_instances_report.txt"
)


def accumulate_label_counts(arr) -> dict[int, int]:
    """Count non-zero label sizes by streaming slabs along axis 0."""
    counts: dict[int, int] = {}
    n0 = arr.shape[0]
    slab = max(1, min(8, n0))
    for i0 in range(0, n0, slab):
        i1 = min(n0, i0 + slab)
        block = np.asarray(arr[i0:i1]).ravel()
        nonzero = block[block != 0]
        if nonzero.size == 0:
            continue
        labs, cts = np.unique(nonzero, return_counts=True)
        for lab, c in zip(labs.tolist(), cts.tolist()):
            lab_i = int(lab)
            if lab_i == 0:
                continue
            counts[lab_i] = counts.get(lab_i, 0) + int(c)
    return counts


def label_size_stats_from_counts(name: str, counts: dict[int, int]) -> dict:
    if not counts:
        return {
            "name": name,
            "n_labels": 0,
            "sizes": np.array([], dtype=np.int64),
            "fg_voxels": 0,
        }
    sizes = np.fromiter(counts.values(), dtype=np.int64)
    return {
        "name": name,
        "n_labels": int(sizes.size),
        "sizes": sizes,
        "fg_voxels": int(sizes.sum()),
        "min": int(sizes.min()),
        "median": float(np.median(sizes)),
        "max": int(sizes.max()),
        "n_lt_200": int((sizes < 200).sum()),
        "n_lt_500": int((sizes < 500).sum()),
        "n_ge_500": int((sizes >= 500).sum()),
        "n_ge_200": int((sizes >= 200).sum()),
    }


def format_label_stats(s: dict) -> str:
    lines = [f"=== HDF dataset: {s['name']} ==="]
    if s["n_labels"] == 0:
        lines.append("n_labels=0 (no non-zero voxels)")
        lines.append(f"fg_voxels={s['fg_voxels']}")
        return "\n".join(lines)
    lines.extend(
        [
            f"n_labels={s['n_labels']}",
            f"fg_voxels={s['fg_voxels']}",
            f"size min/median/max={s['min']}/{s['median']:.1f}/{s['max']}",
            f"n_size<200 (VI ignore_small_comps)={s['n_lt_200']}",
            f"n_size<500 (eval remove_small_components)={s['n_lt_500']}",
            f"n_labels remaining if filter>=200={s['n_ge_200']}",
            f"n_labels remaining if filter>=500={s['n_ge_500']}",
        ]
    )
    sizes = s["sizes"]
    edges = [1, 10, 50, 100, 200, 500, 1000, 5000, 10**9]
    lines.append("size histogram:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((sizes >= lo) & (sizes < hi)).sum())
        if n:
            lines.append(f"  [{lo}, {hi}): {n}")
    return "\n".join(lines)


def foreground_stats(arr) -> str:
    n0 = arr.shape[0]
    slab = max(1, min(16, n0))
    pos = 0
    total = 0
    vmin = None
    vmax = None
    sum_v = 0.0
    is_float = np.issubdtype(np.dtype(arr.dtype), np.floating)
    for i0 in range(0, n0, slab):
        i1 = min(n0, i0 + slab)
        block = np.asarray(arr[i0:i1])
        total += block.size
        if is_float:
            pos += int((block > 0).sum())
            bmin = float(block.min())
            bmax = float(block.max())
            sum_v += float(block.sum())
            vmin = bmin if vmin is None else min(vmin, bmin)
            vmax = bmax if vmax is None else max(vmax, bmax)
        else:
            pos += int((block != 0).sum())
    lines = [
        "=== HDF dataset: vote_foreground ===",
        f"shape={tuple(arr.shape)} dtype={arr.dtype}",
        f"nonzero_or_pos_voxels={pos}",
    ]
    if is_float and total:
        lines.append(
            f"min/max/mean={vmin:.6g}/{vmax:.6g}/{sum_v / total:.6g}"
        )
    return "\n".join(lines)


def array_chunk_stats(arr, name: str, thresh: float = 0.5) -> tuple[str, float | None]:
    """Stream stats using native zarr/h5 spatial tiles (one channel at a time)."""
    shape = tuple(arr.shape)
    dtype = arr.dtype
    total = 0
    sum_v = 0.0
    sum_sq = 0.0
    n_gt = 0
    vmin = None
    vmax = None

    def consume(block: np.ndarray) -> None:
        nonlocal total, sum_v, sum_sq, n_gt, vmin, vmax
        if block.size == 0:
            return
        b = np.asarray(block, dtype=np.float32)
        bmin = float(b.min())
        bmax = float(b.max())
        vmin = bmin if vmin is None else min(vmin, bmin)
        vmax = bmax if vmax is None else max(vmax, bmax)
        sum_v += float(b.sum())
        sum_sq += float(np.square(b, dtype=np.float64).sum())
        n_gt += int((b > thresh).sum())
        total += int(b.size)

    if len(shape) == 4:
        c_dim, z_dim, y_dim, x_dim = shape
        # Subsample channels; tile spatial to ~64^3
        c_step = max(1, c_dim // 16)
        tz = ty = tx = 64
        for c in range(0, c_dim, c_step):
            for z0 in range(0, z_dim, tz):
                for y0 in range(0, y_dim, ty):
                    for x0 in range(0, x_dim, tx):
                        consume(
                            arr[
                                c,
                                z0 : min(z_dim, z0 + tz),
                                y0 : min(y_dim, y0 + ty),
                                x0 : min(x_dim, x0 + tx),
                            ]
                        )
        note = f"channel_step={c_step} spatial_tile={tz}"
    elif len(shape) == 3:
        z_dim, y_dim, x_dim = shape
        tz = ty = tx = 64
        for z0 in range(0, z_dim, tz):
            for y0 in range(0, y_dim, ty):
                for x0 in range(0, x_dim, tx):
                    consume(
                        arr[
                            z0 : min(z_dim, z0 + tz),
                            y0 : min(y_dim, y0 + ty),
                            x0 : min(x_dim, x0 + tx),
                        ]
                    )
        note = f"spatial_tile={tz}"
    else:
        consume(arr[...])
        note = "full load"

    if total == 0:
        return (
            f"=== pred: {name} ===\nempty array shape={shape} dtype={dtype}",
            None,
        )

    mean = sum_v / total
    var = max(0.0, sum_sq / total - mean * mean)
    std = var**0.5
    frac = n_gt / total
    text = "\n".join(
        [
            f"=== pred: {name} ===",
            f"shape={shape} dtype={dtype}",
            note,
            f"min/max/mean/std={vmin:.6g}/{vmax:.6g}/{mean:.6g}/{std:.6g}",
            f"frac>{thresh}={frac:.6g} ({n_gt}/{total})",
        ]
    )
    return text, frac


def decide_verdict(
    vi_stats: dict, aff_frac: float | None, numinst_frac: float | None
) -> str:
    n = vi_stats["n_labels"]
    weak_aff = False
    if aff_frac is not None and numinst_frac is not None:
        # inactive heads: almost nothing above 0.5
        weak_aff = aff_frac < 1e-4 and numinst_frac < 1e-3
    elif aff_frac is not None:
        weak_aff = aff_frac < 1e-4

    if n == 0:
        if weak_aff:
            return "WEAK_PREDICTIONS"
        return "EMPTY_AFTER_VI"

    n_ge_500 = vi_stats.get("n_ge_500", 0)
    if n_ge_500 == 0:
        return "FILTERED_BY_RM500"
    if n_ge_500 > 0 and weak_aff:
        return "MIXED"
    if n_ge_500 > 0:
        # labels large enough to survive rm500 — Num Pred=0 would be unexpected
        return "MIXED"
    return "FILTERED_BY_RM500"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hdf", type=Path, default=DEFAULT_HDF)
    ap.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument(
        "--skip-zarr",
        action="store_true",
        help="Only analyze HDF (skip pred_affs / pred_numinst)",
    )
    args = ap.parse_args()

    lines: list[str] = []
    lines.append("PPP instance / prediction diagnosis")
    lines.append(f"hdf={args.hdf}")
    lines.append(f"zarr={args.zarr}")
    lines.append("")

    if not args.hdf.is_file():
        raise SystemExit(f"HDF not found: {args.hdf}")

    with h5py.File(args.hdf, "r") as f:
        vi_counts = accumulate_label_counts(f["vote_instances"])
        vi_stats = label_size_stats_from_counts("vote_instances", vi_counts)
        lines.append(format_label_stats(vi_stats))
        lines.append("")

        if "vote_instances_masked" in f:
            vim_counts = accumulate_label_counts(f["vote_instances_masked"])
            vim_stats = label_size_stats_from_counts(
                "vote_instances_masked", vim_counts
            )
            lines.append(format_label_stats(vim_stats))
            lines.append("")

        if "vote_foreground" in f:
            lines.append(foreground_stats(f["vote_foreground"]))
            lines.append("")

    aff_frac = None
    numinst_frac = None
    if args.skip_zarr:
        lines.append("=== pred: SKIPPED (--skip-zarr) ===")
        lines.append("")
    elif not args.zarr.exists():
        lines.append(f"WARNING: pred zarr missing: {args.zarr}")
    else:
        root = zarr.open(str(args.zarr), mode="r")
        for key, short in [
            ("volumes/pred_affs", "volumes/pred_affs"),
            ("volumes/pred_numinst", "volumes/pred_numinst"),
        ]:
            try:
                arr = root[key]
            except Exception as e:
                lines.append(f"WARNING: cannot open {key}: {e}")
                continue
            block, frac = array_chunk_stats(arr, short)
            lines.append(block)
            lines.append("")
            if short.endswith("pred_affs"):
                aff_frac = frac
            elif short.endswith("pred_numinst"):
                numinst_frac = frac

    verdict = decide_verdict(vi_stats, aff_frac, numinst_frac)
    lines.append(f"VERDICT: {verdict}")
    lines.append("")
    lines.append(
        "Legend: FILTERED_BY_RM500=labels exist but all size<500; "
        "EMPTY_AFTER_VI=no labels in HDF; "
        "WEAK_PREDICTIONS=empty labels and inactive aff/numinst; "
        "MIXED=needs follow-up."
    )

    text = "\n".join(lines) + "\n"
    print(text, end="")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
