"""Evaluate the "cheap wins" variants against the unpruned NBLAST baseline.

Four levers, measured together so each one's contribution is isolable:
- L/R mirroring (mirror_skeletons.py) -- contralateral homologs of a
  hemilineage are mirror images, which raw-coordinate NBLAST reads as far
  apart.
- Pruning, already validated, stacked with mirroring.
- NBLAST's untried use_alpha / resample knobs.
- An ensemble over pruning severities, which needs no new NBLAST compute
  since every constituent matrix is already on disk.

Also fixes the reporting bug that motivated this sweep: evaluate_metrics
hardcodes HEADLINE_K=10, but k=10 is the *worst* k in the NBLAST sweep
(MCC 0.869 at k=1 vs 0.817 at k=10). Every variant here records its full
k=1..10 curve and its best-k row, not just k=10.

Comparisons respect the fixed-test-set rule established by
evaluate_survivorship_check.py: a variant covering fewer neurons is scored
against the baseline restricted to that same neuron set, never against the
full-set baseline.
"""

import functools
import pickle
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

import evaluate_metrics as em

MANC_DIR = Path.cwd() / "data" / "MANC"
RESULTS_DIR = MANC_DIR / "results"
INSTANCE_FILE_NAME = "T[1]_[LR]"

BASELINE_CSV = MANC_DIR / f"{INSTANCE_FILE_NAME}_nblast_dist.csv"

# compute_nblast.py writes every variant under a legacy "_pruned_" infix,
# including variants that involve no pruning (mirrored/alpha/resampled).
VARIANT_TAGS = [
    "strahler_tertiary",
    "strahler_aggressive",
    "mirrored",
    "strahler_tertiary_mirrored",
    "alpha",
    "resampled",
]
ENSEMBLE_TAGS = ["strahler_mild", "strahler_moderate", "strahler_tertiary"]


def variant_csv(tag):
    return MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_nblast_dist.csv"


def load_matrix(path):
    matrix = pd.read_csv(path, index_col=0)
    matrix.columns = matrix.columns.astype(int)
    matrix.index = matrix.index.astype(int)
    return matrix


def build_ensemble():
    """Elementwise mean of the baseline plus each pruning severity, over the
    neurons all of them share. All NBLAST matrices are on the same
    1 - normalized_score scale, so a plain mean is meaningful.
    """
    paths = [BASELINE_CSV] + [variant_csv(t) for t in ENSEMBLE_TAGS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"Skipping ensemble: missing {[p.name for p in missing]}")
        return None

    matrices = [load_matrix(p) for p in paths]
    shared = set(matrices[0].index)
    for m in matrices[1:]:
        shared &= set(m.index)
    shared = sorted(shared)
    print(f"Ensemble over {len(matrices)} matrices, {len(shared)} shared neurons")

    aligned = [m.loc[shared, shared] for m in matrices]
    for m in aligned[1:]:
        # Align by label, then assert it actually worked rather than trusting
        # row order across independently-written CSVs.
        assert list(m.index) == list(aligned[0].index)
        assert list(m.columns) == list(aligned[0].columns)

    stacked = np.mean([m.to_numpy() for m in aligned], axis=0)
    return pd.DataFrame(stacked, index=aligned[0].index, columns=aligned[0].columns)


def summarize(sweep):
    best = sweep.loc[sweep["mcc"].idxmax()]
    k10 = sweep[sweep["k"] == 10].iloc[0]
    return best, k10


def main():
    common_ids = np.load(RESULTS_DIR / "common_ids.npy")
    with open(RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        id_to_class = pickle.load(f)
    labelled = set(int(i) for i in common_ids)
    print(f"Baseline labelled neurons: {len(labelled)}")

    baseline_matrix = load_matrix(BASELINE_CSV)

    variants = [("baseline", baseline_matrix)]
    for tag in VARIANT_TAGS:
        path = variant_csv(tag)
        if not path.exists():
            print(f"Skipping {tag}: {path.name} not found")
            continue
        variants.append((tag, load_matrix(path)))

    ensemble = build_ensemble()
    if ensemble is not None:
        variants.append(("ensemble_severities", ensemble))

    baseline_cache = {}
    rows, curves = [], []

    for name, matrix in variants:
        ids = np.array(sorted(labelled & set(matrix.index)))
        labels = np.array([id_to_class[i] for i in ids])
        print(f"\n{name}: {len(ids)} neurons")

        sweep = em.evaluate_method(matrix, ids, labels)
        best, k10 = summarize(sweep)

        sweep_out = sweep.copy()
        sweep_out.insert(0, "variant", name)
        curves.append(sweep_out)

        # Fixed-test-set reference: the baseline scored on this variant's own
        # neuron set, cached since several variants share an id set.
        key = (len(ids), int(ids[0]), int(ids[-1]))
        if key not in baseline_cache:
            if len(ids) == len(labelled):
                baseline_cache[key] = summarize(
                    sweep if name == "baseline"
                    else em.evaluate_method(baseline_matrix, ids, labels)
                )[0]["mcc"]
            else:
                print(f"  scoring baseline restricted to these {len(ids)} neurons...")
                baseline_cache[key] = summarize(
                    em.evaluate_method(baseline_matrix, ids, labels)
                )[0]["mcc"]
        baseline_ref = baseline_cache[key]

        print(f"  best k={int(best['k'])}: mcc={best['mcc']:.3f} "
              f"(k=10: {k10['mcc']:.3f}) | baseline on same neurons: {baseline_ref:.3f} "
              f"| delta {best['mcc'] - baseline_ref:+.3f}")

        rows.append({
            "variant": name,
            "n": len(ids),
            "best_k": int(best["k"]),
            "mcc_best_k": best["mcc"],
            "mcc_k10": k10["mcc"],
            "accuracy": best["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"],
            "macro_f1": best["macro_f1"],
            "baseline_same_ids_mcc": baseline_ref,
            "delta_vs_baseline": best["mcc"] - baseline_ref,
        })

    out = pd.DataFrame(rows).sort_values("mcc_best_k", ascending=False)
    out_csv = RESULTS_DIR / "improvement_sweep_performance.csv"
    out.to_csv(out_csv, index=False)

    curves_csv = RESULTS_DIR / "improvement_sweep_kcurves.csv"
    pd.concat(curves, ignore_index=True).to_csv(curves_csv, index=False)

    print(f"\nSaved {out_csv} and {curves_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
