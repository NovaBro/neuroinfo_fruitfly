"""Three visualizations for the UGW eps sweep, saved to data/MANC/results/:
- eps_sweep_runtimes.png: per-block wall-clock time vs eps (from SLURM sacct,
  collected by collect_eps_runtimes.py).
- eps_sweep_distances.png: pairwise UGW-distance distributions vs eps, to see
  how eps compresses/spreads the cost scale.
- eps_sweep_performance.png: leave-one-out KNN classification performance
  (evaluate_eps_sweep.py's output) vs eps, with NBLAST/Geodesic-GW/
  Euclidean-GW reference lines from the main 4-method comparison for context.
"""

import functools
import json

print = functools.partial(print, flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_metrics as em
import ugw_common as common

JOBS_JSON = em.RESULTS_DIR / "eps_sweep_jobs.json"
RUNTIMES_CSV = em.RESULTS_DIR / "eps_sweep_runtimes.csv"
PERFORMANCE_CSV = em.RESULTS_DIR / "eps_sweep_performance.csv"
MAIN_RESULTS_CSV = em.RESULTS_DIR / "results_comparison.csv"


def plot_runtimes():
    if not RUNTIMES_CSV.exists():
        print(f"Skipping runtime plot: {RUNTIMES_CSV} not found "
              f"(run collect_eps_runtimes.py on the bare login node first)")
        return

    df = pd.read_csv(RUNTIMES_CSV)
    eps_values = sorted(df["eps"].unique())
    data = [df.loc[df["eps"] == eps, "elapsed_seconds"].values for eps in eps_values]
    positions = np.log10(eps_values)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, positions=positions, widths=0.15)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{eps:,.0f}" for eps in eps_values], rotation=30, ha="right")
    ax.set_xlabel("eps")
    ax.set_ylabel("Per-block wall-clock time (s)")
    ax.set_title("UGW Block Runtime vs eps")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = em.RESULTS_DIR / "eps_sweep_runtimes.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_distance_distributions(eps_values):
    fig, ax = plt.subplots(figsize=(9, 5))
    found_any = False
    for eps in eps_values:
        csv_path = common.final_ugw_csv_for_eps(eps)
        if not csv_path.exists():
            continue
        found_any = True
        dist_matrix = pd.read_csv(csv_path, index_col=0).values
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        ax.hist(upper, bins=100, histtype="step", linewidth=1.5, label=f"eps={eps:,.0f}", density=True)

    if not found_any:
        print("Skipping distance-distribution plot: no eps sweep FINAL_UGW_CSV files found")
        plt.close(fig)
        return

    ax.set_xlabel("Pairwise UGW distance")
    ax.set_ylabel("Density")
    ax.set_title("UGW Pairwise Distance Distributions vs eps")
    ax.legend()
    plt.tight_layout()
    out_path = em.RESULTS_DIR / "eps_sweep_distances.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_performance():
    if not PERFORMANCE_CSV.exists():
        print(f"Skipping performance plot: {PERFORMANCE_CSV} not found "
              f"(run evaluate_eps_sweep.py first)")
        return

    perf = pd.read_csv(PERFORMANCE_CSV, index_col=0).sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["mcc", "accuracy", "balanced_accuracy", "macro_f1"]:
        ax.plot(perf.index, perf[metric], marker="o", label=metric)

    if MAIN_RESULTS_CSV.exists():
        main_results = pd.read_csv(MAIN_RESULTS_CSV, index_col=0)
        for method in ["NBLAST", "Geodesic GW", "Euclidean GW"]:
            if method in main_results.index:
                ax.axhline(
                    main_results.loc[method, "mcc"], linestyle="--", alpha=0.5,
                    label=f"{method} MCC (baseline)",
                )

    ax.set_xscale("log")
    ax.set_xlabel("eps")
    ax.set_ylabel("Score")
    ax.set_title(f"UGW Hemilineage Classification Performance vs eps (k={em.HEADLINE_K}, LOO)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = em.RESULTS_DIR / "eps_sweep_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    with open(JOBS_JSON) as f:
        jobs = json.load(f)
    eps_values = sorted(float(e) for e in jobs)

    plot_runtimes()
    plot_distance_distributions(eps_values)
    plot_performance()


if __name__ == "__main__":
    main()
