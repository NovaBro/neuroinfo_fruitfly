"""Two visualizations for the centrifugal-order pruning sweep (NBLAST only),
saved to data/MANC/results/:
- centrifugal_nblast_distances.png: pairwise NBLAST distance distributions,
  baseline + 4 severities overlaid.
- centrifugal_nblast_performance.png: leave-one-out KNN classification
  performance (evaluate_centrifugal_nblast_sweep.py's output) vs severity,
  with the unpruned-baseline NBLAST MCC as a reference line.

Single-panel figures (unlike plot_pruning_sweep.py's 2x2 method grid), since
this sweep only covers one method.
"""

import functools

print = functools.partial(print, flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_centrifugal_nblast_sweep as sweep
import evaluate_metrics as em

RESULTS_DIR = em.RESULTS_DIR
PERFORMANCE_CSV = RESULTS_DIR / "centrifugal_nblast_performance.csv"
BASELINE_NBLAST_CSV = sweep.MANC_DIR / f"{sweep.INSTANCE_FILE_NAME}_nblast_dist.csv"


def plot_distance_distributions():
    fig, ax = plt.subplots(figsize=(9, 5))
    found_any = False

    if BASELINE_NBLAST_CSV.exists():
        baseline = pd.read_csv(BASELINE_NBLAST_CSV, index_col=0).values
        upper = baseline[np.triu_indices_from(baseline, k=1)]
        ax.hist(upper, bins=100, histtype="step", linewidth=1.5, label="baseline", density=True)
        found_any = True

    for severity in sweep.SEVERITIES:
        tag = f"centrifugal_{severity}"
        csv_path = sweep.nblast_csv_for_tag(tag)
        if not csv_path.exists():
            continue
        m = pd.read_csv(csv_path, index_col=0).values
        upper = m[np.triu_indices_from(m, k=1)]
        ax.hist(upper, bins=100, histtype="step", linewidth=1.5, label=severity, density=True)
        found_any = True

    if not found_any:
        print("Skipping distance-distribution plot: no NBLAST matrices found")
        plt.close(fig)
        return

    ax.set_xlabel("Pairwise NBLAST distance")
    ax.set_ylabel("Density")
    ax.set_title("NBLAST Pairwise Distance Distributions vs Centrifugal-Order Pruning Severity")
    ax.legend()
    plt.tight_layout()
    out_path = RESULTS_DIR / "centrifugal_nblast_distances.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_performance():
    if not PERFORMANCE_CSV.exists():
        print(f"Skipping performance plot: {PERFORMANCE_CSV} not found "
              f"(run evaluate_centrifugal_nblast_sweep.py first)")
        return

    perf = pd.read_csv(PERFORMANCE_CSV).sort_values("severity_rank")
    baseline_rows = perf[perf["tag"] == "baseline"]
    sweep_rows = perf[perf["tag"] != "baseline"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["mcc", "accuracy", "balanced_accuracy", "macro_f1"]:
        ax.plot(sweep_rows["severity_rank"], sweep_rows[metric], marker="o", label=metric)
        if not baseline_rows.empty:
            ax.axhline(baseline_rows.iloc[0][metric], linestyle="--", alpha=0.4)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(sweep.SEVERITIES, rotation=30, ha="right")
    ax.set_xlabel("severity")
    ax.set_ylabel(f"Score (k={em.HEADLINE_K}, LOO)")
    ax.set_title("NBLAST Hemilineage Classification Performance vs Centrifugal-Order Pruning Severity\n"
                 "(dashed lines = unpruned baseline)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = RESULTS_DIR / "centrifugal_nblast_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_distance_distributions()
    plot_performance()


if __name__ == "__main__":
    main()
