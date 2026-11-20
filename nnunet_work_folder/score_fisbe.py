#!/usr/bin/env python
"""Score a folder of instance label volumes against FISBe GT with the same evalinstseg call that
PatchPerPix's `-d evaluate` makes (PatchPerPix/experiments/run_ppp.py evaluate_file call + setup01
[evaluation] TOML: cldice localization, greedy assignment, evaluate_false_labels, general metrics
avg_gt_skel_coverage / avg_f1_cov_score / false_merge / false_split, fm/fs thresholds 0.1/0.05), and
report both aggregates used in this repo:
  MEAN   = mean of per-volume metrics (what md_guides/experiments/*_status.md report for PPP)
  POOLED = FISBe official instance-pooled aggregate (evalinstseg.summarize, 'avg_inst' row)

Predictions: <pred-dir>/<sample>.zarr | .hdf | .tif/.tiff, integer label map [Z,Y,X] (0 = background),
same Z,Y,X as GT (evalinstseg silently centre-crops on a full mismatch, so shapes are asserted here).
Run inside env/evaluate.ext3 (conda env evalinstseg):
  python nnunet_work_folder/score_fisbe.py --pred-dir <dir> --gt-dir fisbe/completely/test \
      --rm 1500 --out-dir <dir>/eval_rm1500 --workers 7
Writes <out-dir>/fisbe_summary_rm<rm>.csv (per sample + MEAN + POOLED rows).
"""
import argparse
import csv
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from evalinstseg import evaluate_file
from evalinstseg.summarize import average_flylight_score_over_instances
from natsort import natsorted

COLS = [("S", "general.avg_f1_cov_score"), ("avF1", "confusion_matrix.avFscore19"),
        ("C", "general.avg_gt_skel_coverage"), ("clDiceTP", "general.avg_TP_05_cldice"),
        ("tp", "general.TP_05_rel"), ("FS", "general.FS"), ("FM", "general.FM"),
        ("nPred", "general.Num Pred"), ("nGT", "general.Num GT")]


def get(d, key):
    for k in key.split("."):
        d = d[k]
    return float(d)


def zarr_shape(path, key):
    with open(os.path.join(path, key, ".zarray")) as fh:
        return tuple(json.load(fh)["shape"])


def score_one(job):
    f, a = job
    s = os.path.basename(f).split(".")[0]
    gt = os.path.join(a.gt_dir, s + ".zarr")
    assert os.path.isdir(gt), "no GT for %s: %s" % (s, gt)
    if f.endswith(".zarr"):
        ps, gs = zarr_shape(f, a.pred_key), zarr_shape(gt, a.gt_key)
        assert ps[-3:] == gs[-3:], "%s: pred %s vs GT %s" % (s, ps, gs)
    extra = ["avg_gt_cov_dim", "avg_gt_cov_overlap"] if a.eval_dim else []
    m = evaluate_file(
        f, gt, 3, a.out_dir, res_key=a.pred_key, gt_key=a.gt_key,
        localization_criterion="cldice", assignment_strategy="greedy",
        remove_small_components=a.rm, evaluate_false_labels=True,
        add_general_metrics=["avg_gt_skel_coverage", "avg_f1_cov_score", "false_merge", "false_split"] + extra,
        fm_thresh=0.1, fs_thresh=0.05, partly=False, eval_dim=a.eval_dim,
        from_scratch=not a.resume)
    return s, m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--pred-key", default="volumes/pred_instance", help="zarr/hdf key (ignored for tif)")
    p.add_argument("--gt-dir", default="fisbe/completely/test")
    p.add_argument("--gt-key", default="volumes/gt_instances")  # PPP uses gt_instances_rm_5 (identical values)
    p.add_argument("--rm", type=int, default=1500, help="remove_small_components: labels <= rm voxels dropped "
                                                        "(PPP best run: 1500; BiaPy: 500; --app flylight: 800)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-dim", action="store_true", help="also dim/overlap subsets")
    p.add_argument("--resume", action="store_true", help="reuse per-volume TOMLs in out-dir (same settings only)")
    p.add_argument("--workers", type=int, default=1, help="volumes scored in parallel")
    a = p.parse_args()

    preds = natsorted(f for ext in ("zarr", "hdf", "tif", "tiff")
                      for f in glob.glob(os.path.join(a.pred_dir, "*." + ext)))
    assert preds, "no predictions in %s" % a.pred_dir
    os.makedirs(a.out_dir, exist_ok=True)
    res = {}
    with Pool(min(a.workers, len(preds))) as pool:
        for s, m in pool.imap_unordered(score_one, [(f, a) for f in preds]):
            res[s] = m
            print(s, " ".join("%s=%.4f" % (n, get(m, k)) for n, k in COLS), flush=True)
    names = natsorted(res)
    rows = [[s] + [get(res[s], k) for _, k in COLS] for s in names]
    mean = ["MEAN"] + list(np.mean([r[1:] for r in rows], axis=0))
    avS, inst = average_flylight_score_over_instances(names, res)
    g = inst["general"]
    pooled = ["POOLED", avS, g["avFscore"], g["avg_gt_skel_coverage"], g["avg_TP_05_cldice"],
              g["TP_05_rel"], g["FS"], g["FM"], g["Num Pred"], g["Num GT"]]
    out = os.path.join(a.out_dir, "fisbe_summary_rm%d.csv" % a.rm)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample"] + [n for n, _ in COLS])
        w.writerows(rows + [mean, pooled])
    for r in (mean, pooled):
        print(r[0], " ".join("%s=%.4f" % (n, float(v)) for (n, _), v in zip(COLS, r[1:])))
    print("wrote", out)


if __name__ == "__main__":
    main()
