"""Three visualizations for the tertiary-branch pruning sweep, saved to
data/MANC/results/, each split into a Strahler-order panel and a
centrifugal-order panel for direct side-by-side comparison of the two
branch-order definitions:
- pruning_sweep_runtimes.png: per-UGW-block wall-clock time vs severity
  (from SLURM sacct, collected by collect_pruning_runtimes.py).
- pruning_sweep_distances.png: pairwise distance/similarity distributions
  per method, overlaid across severities (+ baseline), to see how pruning
  compresses/spreads each method's distance scale.
- pruning_sweep_performance.png: leave-one-out KNN classification
  performance (evaluate_pruning_sweep.py's output) vs severity, one line per
  method, with the unpruned baseline as a reference line.
"""

import functools

print = functools.partial(print, flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_metrics as em
import evaluate_pruning_sweep as eps_sweep

RESULTS_DIR = em.RESULTS_DIR
RUNTIMES_CSV = RESULTS_DIR / "pruning_sweep_runtimes.csv"
PERFORMANCE_CSV = RESULTS_DIR / "pruning_sweep_performance.csv"

# Centrifugal-order conditions were dropped from this sweep per an explicit
# "only run Strahler" instruction mid-experiment.
DEFINITIONS = ["strahler"]
SEVERITY_ORDER = ["mild", "moderate", "tertiary", "aggressive"]
METHODS = ["NBLAST", "Geodesic GW", "Euclidean GW", "UGW"]


def plot_runtimes():
    if not RUNTIMES_CSV.exists():
        print(f"Skipping runtime plot: {RUNTIMES_CSV} not found "
              f"(run collect_pruning_runtimes.py on the bare login node first)")
        return

    df = pd.read_csv(RUNTIMES_CSV)
    block_df = df[df["stage"] == "ugw_block"]
    if block_df.empty:
        print("Skipping runtime plot: no ugw_block rows in runtimes CSV yet")
        return

    fig, axes = plt.subplots(1, len(DEFINITIONS), figsize=(6.5 * len(DEFINITIONS), 5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, definition in zip(axes, DEFINITIONS):
        sub = block_df[block_df["definition"] == definition]
        data = [sub.loc[sub["severity"] == sev, "elapsed_seconds"].values for sev in SEVERITY_ORDER]
        positions = np.arange(1, len(SEVERITY_ORDER) + 1)
        ax.boxplot(data, positions=positions, widths=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(SEVERITY_ORDER, rotation=30, ha="right")
        ax.set_xlabel("severity")
        ax.set_title(f"{definition} order")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Per-block wall-clock time (s)")
    fig.suptitle("UGW Block Runtime vs Pruning Severity")
    plt.tight_layout()
    out_path = RESULTS_DIR / "pruning_sweep_runtimes.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_distance_distributions():
    baseline_matrices = em.load_distance_matrices() if (RESULTS_DIR / "results_comparison.csv").exists() else {}

    for definition in DEFINITIONS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, method in zip(axes.flat, METHODS):
            if method in baseline_matrices:
                upper = baseline_matrices[method].values[np.triu_indices_from(baseline_matrices[method].values, k=1)]
                ax.hist(upper, bins=100, histtype="step", linewidth=1.5, label="baseline", density=True)

            for severity in SEVERITY_ORDER:
                tag = f"{definition}_{severity}"
                matrices, missing = eps_sweep.load_condition_matrices(tag)
                if matrices is None:
                    continue
                m = matrices[method].values
                upper = m[np.triu_indices_from(m, k=1)]
                ax.hist(upper, bins=100, histtype="step", linewidth=1.5, label=severity, density=True)

            ax.set_title(method)
            ax.set_xlabel("Pairwise distance")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        fig.suptitle(f"Pairwise Distance Distributions vs Pruning Severity ({definition} order)")
        plt.tight_layout()
        out_path = RESULTS_DIR / f"pruning_sweep_distances_{definition}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_performance():
    if not PERFORMANCE_CSV.exists():
        print(f"Skipping performance plot: {PERFORMANCE_CSV} not found "
              f"(run evaluate_pruning_sweep.py first)")
        return

    perf = pd.read_csv(PERFORMANCE_CSV)
    baseline = perf[perf["definition"] == "baseline"].set_index("method")["mcc"]

    fig, axes = plt.subplots(1, len(DEFINITIONS), figsize=(6.5 * len(DEFINITIONS), 5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, definition in zip(axes, DEFINITIONS):
        sub = perf[perf["definition"] == definition]
        for method in METHODS:
            method_rows = sub[sub["method"] == method].sort_values("severity_rank")
            if method_rows.empty:
                continue
            ax.plot(method_rows["severity_rank"], method_rows["mcc"], marker="o", label=method)
            if method in baseline.index:
                ax.axhline(baseline[method], linestyle="--", alpha=0.4)

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(SEVERITY_ORDER, rotation=30, ha="right")
        ax.set_xlabel("severity")
        ax.set_title(f"{definition} order")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel(f"MCC (k={em.HEADLINE_K}, LOO)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Hemilineage Classification Performance vs Pruning Severity\n"
                 "(dashed lines = unpruned baseline)")
    plt.tight_layout()
    out_path = RESULTS_DIR / "pruning_sweep_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_runtimes()
    plot_distance_distributions()
    plot_performance()


if __name__ == "__main__":
    main()
