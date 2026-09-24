#!/usr/bin/env python3
"""Fast FG (mid-aff) + numinst F1 for one processed prediction zarr vs GT.

Avoids loading full pred_affs. Writes a small CSV next to the zarr.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]


def f1_from_masks(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, int, int, int]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = int(np.logical_and(gt_b, pred_b).sum())
    num_gt = int(gt_b.sum())
    num_pred = int(pred_b.sum())
    if num_pred > 0 and tp > 0:
        precision = tp / float(num_pred)
        recall = tp / float(num_gt)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = f1 = 0.0
    return f1, precision, recall, tp, num_gt, num_pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred-zarr",
        type=Path,
        required=True,
        help="Path to sample.zarr under test/processed/<ckpt>/",
    )
    ap.add_argument(
        "--gt-zarr",
        type=Path,
        default=REPO
        / "fisbe/completely/val/R22C03-20180918_66_J2.zarr",
    )
    ap.add_argument("--gt-key", default="volumes/gt_instances_rm_5")
    ap.add_argument("--aff-key", default="volumes/pred_affs")
    ap.add_argument("--numinst-key", default="volumes/pred_numinst")
    ap.add_argument("--patchshape", type=int, nargs=3, default=[7, 7, 7])
    ap.add_argument("--numinst-threshs", type=float, nargs=2, default=[0.9, 0.1])
    ap.add_argument(
        "--fg-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
    )
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    pred = zarr.open(str(args.pred_zarr), mode="r")
    gt_root = zarr.open(str(args.gt_zarr), mode="r")
    labels = np.asarray(gt_root[args.gt_key])
    gt_fg = np.max(labels > 0, axis=0).astype(np.uint8)
    gt_numinst = np.sum(labels > 0, axis=0).astype(np.uint8)
    gt_numinst = np.clip(gt_numinst, 0, 2)

    mid = int(np.prod(args.patchshape) // 2)
    aff = pred[args.aff_key]
    # mid channel only — one volume
    mid_aff = np.asarray(aff[mid], dtype=np.float32)
    numinst_prob = np.asarray(pred[args.numinst_key], dtype=np.float32)
    pred_numinst = np.zeros(numinst_prob.shape[1:], dtype=np.uint8)
    for i, th in enumerate(args.numinst_threshs):
        pred_numinst[numinst_prob[i + 1] > th] = i + 1

    rows: list[dict] = []
    for th in args.fg_thresholds:
        f1, prec, rec, tp, ng, np_ = f1_from_masks(gt_fg, mid_aff > th)
        rows.append(
            {
                "metric": "fg_mid_aff",
                "thresh": th,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "tp": tp,
                "num_gt": ng,
                "num_pred": np_,
            }
        )
        print(
            f"fg_mid_aff th={th}: f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} "
            f"tp={tp} gt={ng} pred={np_}"
        )

    for i in (0, 1, 2):
        f1, prec, rec, tp, ng, np_ = f1_from_masks(gt_numinst == i, pred_numinst == i)
        rows.append(
            {
                "metric": f"numinst_{i}",
                "thresh": "",
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "tp": tp,
                "num_gt": ng,
                "num_pred": np_,
            }
        )
        print(
            f"numinst class {i}: f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} "
            f"tp={tp} gt={ng} pred={np_}"
        )

    out = args.out_csv
    if out is None:
        out = args.pred_zarr.parent / f"{args.pred_zarr.stem}_pred_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "thresh",
                "f1",
                "precision",
                "recall",
                "tp",
                "num_gt",
                "num_pred",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
