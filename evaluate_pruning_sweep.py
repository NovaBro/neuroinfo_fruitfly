"""Hemilineage-classification comparison across the 8 pruning-sweep
conditions (2 branch-order definitions x 4 severities) plus the unpruned
baseline, for all four distance methods (NBLAST, Geodesic GW, Euclidean GW,
UGW). Mirrors evaluate_eps_sweep.py's structure, generalized from an eps
manifest to a (definition, severity) tag.

Reuses evaluate_metrics.evaluate_method / create_symmetric_matrix unchanged
-- only the matrix-loading/path logic is new, since pruning uses a different
filename convention (T[1]_[LR]_pruned_<tag>_*) than the eps sweep's.

Since pruning can remove some neurons entirely (see prune_skeletons.py's
per-condition survival counts), each condition is evaluated on whichever
subset of the baseline's common_ids actually survived pruning and appears in
all four of that condition's distance matrices, not necessarily the full
baseline common_ids set.

Expects prune_skeletons.py, compute_icdm_and_gw.py, and compute_nblast.py to
have already been run for each condition, and the existing baseline
get_common_ids.py/evaluate_metrics.py outputs (common_ids.npy,
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

# Centrifugal-order conditions were dropped from this sweep per an explicit
# "only run Strahler" instruction mid-experiment -- prune_skeletons.py still
# supports them (CENTRIFUGAL_CONDITIONS, prune_centrifugal), just unused here.
STRAHLER_SEVERITIES = ["mild", "moderate", "tertiary", "aggressive"]
SEVERITY_RANK = {"mild": 1, "moderate": 2, "tertiary": 3, "aggressive": 4}

CONDITIONS = [
    {"definition": "strahler", "severity": s, "tag": f"strahler_{s}"} for s in STRAHLER_SEVERITIES
]


def ugw_csv_for_tag(tag):
    return MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_ugw_dist_mass80_eps{common.eps_suffix(common.EPS)}.csv"


def load_condition_matrices(tag):
    paths = {
        "NBLAST": MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_nblast_dist.csv",
        "Geodesic GW": MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_gw_dist_geodesic.csv",
        "Euclidean GW": MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_gw_dist_euclidean.csv",
        "UGW": ugw_csv_for_tag(tag),
    }
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        return None, missing

    matrices = {}
    for name in ["Geodesic GW", "Euclidean GW"]:
        raw = pd.read_csv(paths[name], index_col=0)
        matrices[name] = em.create_symmetric_matrix(raw)
    for name in ["NBLAST", "UGW"]:
        m = pd.read_csv(paths[name], index_col=0)
        m.columns = m.columns.astype(int)
        matrices[name] = m
    return matrices, []


def main():
    common_ids_baseline = np.load(RESULTS_DIR / "common_ids.npy")
    with open(RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        id_to_class = pickle.load(f)
    print(f"Baseline common IDs: {len(common_ids_baseline)}")

    rows = []

    baseline_csv = RESULTS_DIR / "results_comparison.csv"
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv, index_col=0)
        for method, r in baseline.iterrows():
            rows.append({
                "definition": "baseline", "severity": "none", "severity_rank": 0,
                "tag": "baseline", "method": method,
                "mcc": r["mcc"], "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
            })
        print(f"Loaded {len(baseline)} baseline rows from {baseline_csv}")
    else:
        print(f"No baseline {baseline_csv} found, skipping baseline reference rows")

    for cond in CONDITIONS:
        tag = cond["tag"]
        matrices, missing = load_condition_matrices(tag)
        if matrices is None:
            print(f"Skipping {tag}: missing {missing}")
            continue

        surviving_ids = set(int(x) for x in common_ids_baseline)
        for m in matrices.values():
            surviving_ids &= set(m.index)
        surviving_ids = np.array(sorted(surviving_ids))
        labels = np.array([id_to_class[i] for i in surviving_ids])
        print(f"\n{tag}: {len(surviving_ids)}/{len(common_ids_baseline)} baseline IDs survived")

        for method, dist_matrix in matrices.items():
            sweep = em.evaluate_method(dist_matrix, surviving_ids, labels)
            headline = sweep[sweep["k"] == em.HEADLINE_K].iloc[0].to_dict()
            print(f"  {method}: mcc={headline['mcc']:.3f} (k={em.HEADLINE_K}, n={len(surviving_ids)})")
            rows.append({
                "definition": cond["definition"], "severity": cond["severity"],
                "severity_rank": SEVERITY_RANK[cond["severity"]],
                "tag": tag, "method": method,
                "mcc": headline["mcc"], "accuracy": headline["accuracy"],
                "balanced_accuracy": headline["balanced_accuracy"], "macro_f1": headline["macro_f1"],
            })

    result_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "pruning_sweep_performance.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved pruning-sweep performance (k={em.HEADLINE_K}) to {out_csv}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
