"""Unified hemilineage-classification comparison across NBLAST, Geodesic GW,
Euclidean GW, and Unbalanced GW distance matrices on the MANC T1_[LR] subset.

Reuses the existing evaluation templates rather than inventing new ones:
- Leave-one-out KNN with a precomputed distance metric + Matthews correlation
  (gw_only.ipynb).
- A k=1-10 sweep of accuracy / balanced accuracy / macro-F1
  (manc.ipynb's evaluate_and_plot_knn pattern).

Expects get_common_ids.py, subset_nblast.py, and ugw.py to have already been
run so that data/MANC/results/{common_ids.npy,id_to_class.pkl} and the four
distance-matrix CSVs exist.
"""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

import ugw_common as common

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"
RESULTS_DIR = MANC_DIR / "results"

INSTANCE_FILE_NAME = "T[1]_[LR]"
K_VALUES = range(1, 11)
HEADLINE_K = 10


def create_symmetric_matrix(df):
    """Pivot a flat CAJAL pairwise-distance dataframe into a symmetric square matrix."""
    flat_df = df.reset_index()
    flat_df.columns = ["query", "target", "gw_distance"]
    dist_matrix = flat_df.pivot(index="query", columns="target", values="gw_distance")
    dist_matrix = dist_matrix.combine_first(dist_matrix.T).fillna(0.0)
    return dist_matrix


def load_distance_matrices():
    gw_distances_geo = pd.read_csv(
        MANC_DIR / f"{INSTANCE_FILE_NAME}_gw_dist_geodesic.csv", index_col=0
    )
    gw_distances_euc = pd.read_csv(
        MANC_DIR / f"{INSTANCE_FILE_NAME}_gw_dist_euclidean.csv", index_col=0
    )
    dist_matrix_geo = create_symmetric_matrix(gw_distances_geo)
    dist_matrix_euc = create_symmetric_matrix(gw_distances_euc)

    dist_matrix_nblast = pd.read_csv(
        MANC_DIR / f"{INSTANCE_FILE_NAME}_nblast_dist.csv", index_col=0
    )
    dist_matrix_nblast.columns = dist_matrix_nblast.columns.astype(int)

    dist_matrix_ugw = pd.read_csv(common.FINAL_UGW_CSV, index_col=0)
    dist_matrix_ugw.columns = dist_matrix_ugw.columns.astype(int)

    return {
        "NBLAST": dist_matrix_nblast,
        "Geodesic GW": dist_matrix_geo,
        "Euclidean GW": dist_matrix_euc,
        "UGW": dist_matrix_ugw,
    }


def evaluate_method(dist_matrix, common_ids, labels):
    aligned = dist_matrix.loc[common_ids, common_ids].values

    rows = []
    for k in K_VALUES:
        knn = KNeighborsClassifier(
            n_neighbors=k, metric="precomputed", weights="distance"
        )
        predicted = cross_val_predict(knn, aligned, labels, cv=LeaveOneOut(), n_jobs=-1)
        rows.append(
            {
                "k": k,
                "mcc": matthews_corrcoef(labels, predicted),
                "accuracy": accuracy_score(labels, predicted),
                "balanced_accuracy": balanced_accuracy_score(labels, predicted),
                "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def main():
    common_ids = np.load(RESULTS_DIR / "common_ids.npy")
    with open(RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        id_to_class = pickle.load(f)
    labels = np.array([id_to_class[i] for i in common_ids])
    print(f"Evaluating on {len(common_ids)} common body IDs")

    matrices = load_distance_matrices()

    headline_rows = []
    all_sweeps = {}
    for method_name, dist_matrix in matrices.items():
        print(f"\nEvaluating {method_name}...")
        sweep = evaluate_method(dist_matrix, common_ids, labels)
        all_sweeps[method_name] = sweep
        headline = sweep[sweep["k"] == HEADLINE_K].iloc[0].to_dict()
        headline["method"] = method_name
        headline_rows.append(headline)
        print(sweep.to_string(index=False))

    results_comparison = pd.DataFrame(headline_rows).set_index("method")
    results_comparison = results_comparison[["mcc", "accuracy", "balanced_accuracy", "macro_f1"]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_comparison.to_csv(RESULTS_DIR / "results_comparison.csv")
    print(f"\nSaved headline (k={HEADLINE_K}) comparison to results_comparison.csv")
    print(results_comparison)

    for method_name, sweep in all_sweeps.items():
        sweep.to_csv(RESULTS_DIR / f"k_sweep_{method_name.replace(' ', '_')}.csv", index=False)

    # Bar chart of the headline metrics across methods.
    fig, ax = plt.subplots(figsize=(9, 5))
    results_comparison.plot(kind="bar", ax=ax)
    ax.set_title(f"Hemilineage Classification Performance (k={HEADLINE_K}, LOO)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Distance/Similarity Method")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "results_comparison.png", dpi=150)
    print(f"Saved comparison plot to {RESULTS_DIR / 'results_comparison.png'}")


if __name__ == "__main__":
    main()
