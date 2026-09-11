"""Evaluate hemilineage-classification performance for every eps value that
completed a full pipeline run (per data/MANC/results/eps_sweep_jobs.json),
using the same leave-one-out KNN evaluation as the main 4-method comparison.
Collects headline (k=10) metrics per eps into eps_sweep_performance.csv.
"""

import functools
import json
import pickle

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

import evaluate_metrics as em
import ugw_common as common

JOBS_JSON = em.RESULTS_DIR / "eps_sweep_jobs.json"
OUT_CSV = em.RESULTS_DIR / "eps_sweep_performance.csv"


def main():
    with open(JOBS_JSON) as f:
        jobs = json.load(f)
    eps_values = sorted(float(e) for e in jobs)
    print(f"Evaluating {len(eps_values)} eps values: {eps_values}")

    common_ids = np.load(em.RESULTS_DIR / "common_ids.npy")
    with open(em.RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        id_to_class = pickle.load(f)
    labels = np.array([id_to_class[i] for i in common_ids])
    print(f"Evaluating on {len(common_ids)} common body IDs")

    headline_rows = []
    for eps in eps_values:
        csv_path = common.final_ugw_csv_for_eps(eps)
        if not csv_path.exists():
            print(f"\nSkipping eps={eps}: {csv_path} not found (pipeline not finished?)")
            continue

        print(f"\nEvaluating UGW eps={eps}...")
        dist_matrix = pd.read_csv(csv_path, index_col=0)
        dist_matrix.columns = dist_matrix.columns.astype(int)

        sweep = em.evaluate_method(dist_matrix, common_ids, labels)
        print(sweep.to_string(index=False))
        sweep.to_csv(em.RESULTS_DIR / f"k_sweep_eps{common.eps_suffix(eps)}.csv", index=False)

        headline = sweep[sweep["k"] == em.HEADLINE_K].iloc[0].to_dict()
        headline["eps"] = eps
        headline_rows.append(headline)

    results = pd.DataFrame(headline_rows).set_index("eps")
    results = results[["mcc", "accuracy", "balanced_accuracy", "macro_f1"]]
    results.to_csv(OUT_CSV)
    print(f"\nSaved eps-sweep performance (k={em.HEADLINE_K}) to {OUT_CSV}")
    print(results)


if __name__ == "__main__":
    main()
