"""Check whether the Strahler pruning sweep's headline result is a
survivorship artifact.

Aggressive pruning removes neurons that prune down to nothing (702 of 4105
for strahler_aggressive), so evaluate_pruning_sweep.py scored that condition
on only 3403 neurons while the baseline row was scored on all 4105. If the
dropped neurons were disproportionately hard to classify, part of the
apparent "pruning improves NBLAST" effect is just an easier test set.

This re-scores the UNPRUNED baseline matrices restricted to exactly the
neurons that survived each pruned condition, so the comparison becomes
apples-to-apples. Only conditions that dropped more than ~1% of neurons are
checked -- mild (0 dropped) and moderate (6 dropped) can't move the numbers.

Reuses evaluate_metrics.load_distance_matrices/evaluate_method unchanged.
"""

import functools
import pickle

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

import evaluate_metrics as em
import evaluate_pruning_sweep as sweep

CONDITIONS = ["strahler_tertiary", "strahler_aggressive"]


def surviving_ids_for(tag, common_ids):
    """All four of a condition's matrices are built from the same pruned
    skeleton directory, so the NBLAST matrix's index alone gives the
    surviving set (its count matches the 4-way intersection that
    evaluate_pruning_sweep.py reported for each condition).
    """
    csv_path = sweep.MANC_DIR / f"{sweep.INSTANCE_FILE_NAME}_pruned_{tag}_nblast_dist.csv"
    pruned_matrix = pd.read_csv(csv_path, index_col=0)
    return np.array(sorted(set(int(x) for x in common_ids) & set(pruned_matrix.index)))


def main():
    common_ids = np.load(em.RESULTS_DIR / "common_ids.npy")
    with open(em.RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        id_to_class = pickle.load(f)

    print("Loading unpruned baseline distance matrices...")
    baseline_matrices = em.load_distance_matrices()
    prior = pd.read_csv(em.RESULTS_DIR / "pruning_sweep_performance.csv")

    rows = []
    for tag in CONDITIONS:
        ids = surviving_ids_for(tag, common_ids)
        labels = np.array([id_to_class[i] for i in ids])
        print(f"\n{tag}: scoring unpruned baseline on the {len(ids)}/{len(common_ids)} "
              f"neurons that survived this condition")

        for method, matrix in baseline_matrices.items():
            result = em.evaluate_method(matrix, ids, labels)
            subset_mcc = result[result["k"] == em.HEADLINE_K].iloc[0]["mcc"]

            full_mcc = prior[(prior["tag"] == "baseline") & (prior["method"] == method)]["mcc"].iloc[0]
            pruned_mcc = prior[(prior["tag"] == tag) & (prior["method"] == method)]["mcc"].iloc[0]

            print(f"  {method}: baseline_full={full_mcc:.3f} "
                  f"baseline_on_subset={subset_mcc:.3f} pruned={pruned_mcc:.3f}")
            rows.append({
                "tag": tag,
                "n": len(ids),
                "method": method,
                "baseline_full_mcc": full_mcc,
                "baseline_subset_mcc": subset_mcc,
                "pruned_mcc": pruned_mcc,
                # How much of the pruned-vs-baseline gap survives once the
                # test set is held fixed:
                "naive_delta": pruned_mcc - full_mcc,
                "true_delta": pruned_mcc - subset_mcc,
            })

    out = pd.DataFrame(rows)
    out_csv = em.RESULTS_DIR / "survivorship_check.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
