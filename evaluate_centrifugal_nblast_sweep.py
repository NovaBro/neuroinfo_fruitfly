"""Hemilineage-classification comparison across the 4 centrifugal-order
pruning severities, NBLAST only (no Geodesic/Euclidean GW, no UGW -- this
branch-order definition is being swept for NBLAST alone per an explicit
request, after the Strahler-order sweep already covered all four methods).

Mirrors evaluate_pruning_sweep.py's structure, but single-method: reuses
evaluate_metrics.evaluate_method unchanged, and the existing baseline NBLAST
matrix (data/MANC/T[1]_[LR]_nblast_dist.csv, from subset_nblast.py) rather
than recomputing a baseline.

Expects prune_skeletons.py and compute_nblast.py to have already been run
for each of the 4 centrifugal_<severity> conditions, and the existing
baseline get_common_ids.py/evaluate_metrics.py outputs (common_ids.npy,
id_to_class.pkl, results_comparison.csv) to already exist.
"""

import functools
import pickle
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

import evaluate_metrics as em
import ugw_common as common

MANC_DIR = common.MANC_DIR
RESULTS_DIR = MANC_DIR / "results"
INSTANCE_FILE_NAME = common.INSTANCE_FILE_NAME

SEVERITIES = ["mild", "moderate", "tertiary", "aggressive"]
SEVERITY_RANK = {"mild": 1, "moderate": 2, "tertiary": 3, "aggressive": 4}


def nblast_csv_for_tag(tag):
    return MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_nblast_dist.csv"


def main():
    common_ids_baseline = np.load(RESULTS_DIR / "common_ids.npy")
    with open(RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        id_to_class = pickle.load(f)
    print(f"Baseline common IDs: {len(common_ids_baseline)}")

    rows = []

    baseline_csv = RESULTS_DIR / "results_comparison.csv"
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv, index_col=0)
        if "NBLAST" in baseline.index:
            r = baseline.loc["NBLAST"]
            rows.append({
                "severity": "none", "severity_rank": 0, "tag": "baseline",
                "mcc": r["mcc"], "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
            })
            print(f"Loaded baseline NBLAST row from {baseline_csv}")
    else:
        print(f"No baseline {baseline_csv} found, skipping baseline reference row")

    for severity in SEVERITIES:
        tag = f"centrifugal_{severity}"
        csv_path = nblast_csv_for_tag(tag)
        if not csv_path.exists():
            print(f"Skipping {tag}: {csv_path} not found")
            continue

        dist_matrix = pd.read_csv(csv_path, index_col=0)
        dist_matrix.columns = dist_matrix.columns.astype(int)

        surviving_ids = set(int(x) for x in common_ids_baseline) & set(dist_matrix.index)
        surviving_ids = np.array(sorted(surviving_ids))
        labels = np.array([id_to_class[i] for i in surviving_ids])
        print(f"\n{tag}: {len(surviving_ids)}/{len(common_ids_baseline)} baseline IDs survived")

        sweep = em.evaluate_method(dist_matrix, surviving_ids, labels)
        headline = sweep[sweep["k"] == em.HEADLINE_K].iloc[0].to_dict()
        print(f"  NBLAST: mcc={headline['mcc']:.3f} (k={em.HEADLINE_K}, n={len(surviving_ids)})")
        rows.append({
            "severity": severity, "severity_rank": SEVERITY_RANK[severity], "tag": tag,
            "mcc": headline["mcc"], "accuracy": headline["accuracy"],
            "balanced_accuracy": headline["balanced_accuracy"], "macro_f1": headline["macro_f1"],
        })

    result_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "centrifugal_nblast_performance.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved centrifugal NBLAST performance (k={em.HEADLINE_K}) to {out_csv}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
