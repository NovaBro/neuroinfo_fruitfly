"""Two figures for the cheap-wins sweep, saved to data/MANC/results/:
- improvement_sweep_performance.png: MCC at each variant's best k, as a bar
  chart against the fixed-test-set baseline, so each lever's contribution is
  directly readable.
- improvement_sweep_kcurves.png: MCC vs k for every variant, which is what
  makes the "k=10 was the worst choice" point visible rather than asserted.
"""

import functools
from pathlib import Path

print = functools.partial(print, flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path.cwd() / "data" / "MANC" / "results"
PERFORMANCE_CSV = RESULTS_DIR / "improvement_sweep_performance.csv"
CURVES_CSV = RESULTS_DIR / "improvement_sweep_kcurves.csv"


def plot_performance():
    if not PERFORMANCE_CSV.exists():
        print(f"Skipping performance plot: {PERFORMANCE_CSV} not found")
        return

    perf = pd.read_csv(PERFORMANCE_CSV).sort_values("mcc_best_k")
    colors = ["tab:grey" if v == "baseline" else "tab:blue" for v in perf["variant"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(perf["variant"], perf["mcc_best_k"], color=colors)
    for bar, row in zip(bars, perf.itertuples()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row.mcc_best_k:.3f} (k={row.best_k}, n={row.n}, {row.delta_vs_baseline:+.3f})",
                va="center", fontsize=8)

    baseline = perf[perf["variant"] == "baseline"]
    if not baseline.empty:
        ax.axvline(baseline.iloc[0]["mcc_best_k"], linestyle="--", color="k", alpha=0.5,
                   label="unpruned baseline (best k)")
        ax.legend(fontsize=8, loc="lower right")

    ax.set_xlabel("MCC at best k (LOO)")
    ax.set_title("Hemilineage Classification: effect of each improvement lever\n"
                 "(delta is vs. the baseline scored on that variant's own neuron set)")
    ax.set_xlim(0, max(1.0, perf["mcc_best_k"].max() * 1.25))
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = RESULTS_DIR / "improvement_sweep_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_k_curves():
    if not CURVES_CSV.exists():
        print(f"Skipping k-curve plot: {CURVES_CSV} not found")
        return

    curves = pd.read_csv(CURVES_CSV)
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, group in curves.groupby("variant"):
        group = group.sort_values("k")
        style = "--" if variant == "baseline" else "-"
        ax.plot(group["k"], group["mcc"], marker="o", linestyle=style, label=variant)

    ax.set_xlabel("k (KNN neighbours)")
    ax.set_ylabel("MCC (LOO)")
    ax.set_title("MCC vs k for every variant\n"
                 "(the previously-reported headline used k=10, the worst end of the curve)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = RESULTS_DIR / "improvement_sweep_kcurves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    plot_performance()
    plot_k_curves()


if __name__ == "__main__":
    main()
