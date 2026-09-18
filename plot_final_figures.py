"""Final static figure set for the pruning study, from the tables written by
build_figure_data.py. Reads nothing else -- no distance matrices, no SWCs --
so figures can be revised without recomputing anything.

The argument the figures make, in order:
  F1  NBLAST starts ahead and Strahler pruning moves it further up while
      moving GW/UGW down; centrifugal pruning moves NBLAST down.
  F2  that conclusion is not an artifact of choosing MCC.
  F3  mechanism, in the distance distributions.
  F4  what pruning actually removes from a neuron, and how much.
  F5  the effect survives the fixed-test-set control.
"""

import functools
from pathlib import Path

print = functools.partial(print, flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path.cwd() / "data" / "MANC" / "results"
DATA_DIR = RESULTS_DIR / "figure_data"
FIG_DIR = RESULTS_DIR / "figures"

METHODS = ["NBLAST", "Geodesic GW", "Euclidean GW", "UGW"]
SEVERITY_ORDER = ["none", "mild", "moderate", "tertiary", "aggressive"]

sns.set_theme(style="whitegrid", context="talk")
# One colour per method, reused in every figure so NBLAST reads the same
# throughout; sequential ramp for the ordered severities.
METHOD_COLORS = dict(zip(METHODS, sns.color_palette("colorblind", len(METHODS))))
SEVERITY_COLORS = dict(zip(SEVERITY_ORDER, sns.color_palette("rocket_r", 5)))


def load(name):
    path = DATA_DIR / name
    if not path.exists():
        print(f"missing {path.name}, skipping figures that need it")
        return None
    return pd.read_csv(path)


def best_k(tidy):
    """Each method at its own best k -- best k is method-dependent here
    (NBLAST peaks at k=1, UGW nearer k=6-7), so a single fixed k would
    understate some methods and flatter others."""
    idx = tidy.groupby(["definition", "severity", "method"])["mcc"].idxmax()
    return tidy.loc[idx].copy()


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png / .pdf")


def f1_headline(tidy):
    peak = best_k(tidy)
    base = peak[peak["definition"] == "baseline"].set_index("method")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, definition in zip(axes, ["strahler", "centrifugal"]):
        sub = peak[peak["definition"] == definition]
        for method in METHODS:
            rows = sub[sub["method"] == method].set_index("severity")
            if rows.empty:
                # Only NBLAST was run for centrifugal; show where the other
                # methods started rather than a misleading unconnected point.
                if method in base.index:
                    ax.axhline(base.loc[method, "mcc"], color=METHOD_COLORS[method],
                               linestyle=":", linewidth=2, alpha=0.8)
                continue
            severities = [s for s in SEVERITY_ORDER if s == "none" or s in rows.index]
            y = [base.loc[method, "mcc"] if s == "none" else rows.loc[s, "mcc"]
                 for s in severities]
            ax.plot(severities, y, marker="o", markersize=9, linewidth=2.5,
                    color=METHOD_COLORS[method], label=method)
        ax.set_title(f"{definition.capitalize()}-order pruning")
        ax.set_xlabel("pruning severity")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("MCC at each method's best k")
    axes[1].legend(title="method", fontsize=12, title_fontsize=12, loc="center right")
    axes[1].text(0.02, 0.02, "dotted = unpruned baseline\n(GW/UGW not run for centrifugal)",
                 transform=axes[1].transAxes, fontsize=11, color="#555")
    fig.suptitle("Strahler pruning raises NBLAST and collapses GW/UGW;\n"
                 "centrifugal pruning lowers NBLAST instead", y=1.04)
    save(fig, "F1_headline_severity_by_method")


def f2_metric_robustness(tidy):
    peak = best_k(tidy)
    sel = peak[((peak["definition"] == "baseline")
                | ((peak["definition"] == "strahler") & (peak["severity"] == "tertiary")))]
    metrics = ["mcc", "accuracy", "balanced_accuracy", "macro_f1"]
    long = sel.melt(id_vars=["method", "definition"], value_vars=metrics,
                    var_name="metric", value_name="score")
    long["condition"] = long["definition"].map(
        {"baseline": "unpruned", "strahler": "Strahler tertiary"})
    # Raw column names overlapped each other on the shared x axis.
    long["metric"] = long["metric"].map({"mcc": "MCC", "accuracy": "Acc",
                                         "balanced_accuracy": "BAcc",
                                         "macro_f1": "F1"})

    g = sns.catplot(data=long, x="metric", y="score", hue="condition", col="method",
                    kind="bar", col_order=METHODS, height=4.5, aspect=1.15,
                    palette="colorblind", sharey=True)
    g.set_axis_labels("", "score")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=0)
    g.figure.suptitle("The same conclusion holds on every metric, not just MCC", y=1.05)
    save(g.figure, "F2_metric_robustness")


def pair_auc(rows):
    """P(a random between-hemilineage pair is further apart than a random
    within-hemilineage pair). 0.5 is chance; higher means the method separates
    the classes. This is the quantity classification actually depends on --
    the overall spread of distances does not measure it, which is why the
    coefficient of variation turned out to be uninformative here."""
    within = rows.loc[rows["same_class"], "distance"].to_numpy()
    between = rows.loc[~rows["same_class"], "distance"].to_numpy()
    if len(within) == 0 or len(between) == 0:
        return np.nan
    order = np.argsort(np.concatenate([within, between]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_sum = ranks[: len(within)].sum()
    u = rank_sum - len(within) * (len(within) + 1) / 2
    return 1.0 - u / (len(within) * len(between))


def f3_distance_distributions(samples):
    sub = samples[samples["definition"].isin(["baseline", "strahler"])].copy()
    sub["severity"] = pd.Categorical(sub["severity"], SEVERITY_ORDER, ordered=True)
    plot_sub = sub.groupby(["method", "severity"], observed=True).sample(
        frac=0.2, random_state=0)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10),
                             gridspec_kw={"height_ratios": [2, 1]})
    for col, method in enumerate(METHODS):
        rows = plot_sub[plot_sub["method"] == method]
        if rows.empty:
            continue
        ax = axes[0][col]
        sns.violinplot(data=rows, x="severity", y="distance", ax=ax,
                       palette=SEVERITY_COLORS, order=SEVERITY_ORDER,
                       cut=0, linewidth=0.8, hue="severity", legend=False)
        ax.set_title(method, color=METHOD_COLORS[method])
        ax.set_xlabel("")
        ax.set_ylabel("pairwise distance" if col == 0 else "")
        ax.set_xticklabels([])

        auc = (sub[sub["method"] == method]
               .groupby("severity", observed=True)[["distance", "same_class"]]
               .apply(pair_auc))
        present = [s for s in SEVERITY_ORDER if s in auc.index]
        ax2 = axes[1][col]
        ax2.plot(present, [auc[s] for s in present], marker="o",
                 color=METHOD_COLORS[method], linewidth=2.5)
        ax2.axhline(0.5, color="#999", linestyle="--", linewidth=1)
        ax2.set_ylim(0.45, 1.0)
        ax2.set_ylabel("pair AUC\n(within vs between)" if col == 0 else "")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=40)
    # Top-row axes are deliberately not shared: NBLAST distances live on 0-2
    # and UGW near 1e7, so a common scale would flatten three of the four.
    # The bottom row IS shared, being dimensionless and the point of comparison.
    fig.suptitle("Mechanism: absolute distances shrink for every method (top), but only "
                 "NBLAST keeps separating\nsame-hemilineage pairs from different ones "
                 "(bottom; 0.5 = chance)", y=1.02)
    save(fig, "F3_distance_distributions")


def _draw_overlay(ax, neuron_rows, conditions, labels):
    for condition, label in zip(conditions, labels):
        rows = neuron_rows[neuron_rows["condition"] == condition]
        if rows.empty:
            continue
        ax.scatter(rows["x"], rows["y"], s=1.6, linewidths=0,
                   color=SEVERITY_COLORS[label], label=label)


def f4_before_after(skel):
    body_ids = sorted(skel["body_id"].unique())
    conditions = ["none"] + [f"strahler_{s}" for s in SEVERITY_ORDER[1:]]

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    for ax, body_id in zip(axes.flat, body_ids[:9]):
        rows = skel[skel["body_id"] == body_id]
        _draw_overlay(ax, rows, conditions, SEVERITY_ORDER)
        kept = rows[rows["condition"] == conditions[-1]].shape[0]
        total = rows[rows["condition"] == "none"].shape[0]
        ax.set_title(f"{body_id}  ({total} nodes -> {kept})", fontsize=13)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                          color=SEVERITY_COLORS[s], label=s) for s in SEVERITY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, title="Strahler severity",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Strahler pruning is strictly nested: each successive level is drawn "
                 "over the last,\nso the arbor visibly melts away from the tips inward", y=1.01)
    save(fig, "F4_pruning_before_after")


def f4b_retention(retention):
    sub = retention[retention["definition"].isin(["baseline", "strahler"])].copy()
    sub["severity"] = pd.Categorical(sub["severity"], SEVERITY_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=sub, x="severity", y="frac_retained", order=SEVERITY_ORDER,
                hue="severity", palette=SEVERITY_COLORS, legend=False,
                showfliers=False, ax=ax)
    medians = sub.groupby("severity", observed=True)["frac_retained"].median()
    for i, severity in enumerate(SEVERITY_ORDER):
        if severity in medians:
            ax.text(i + 0.34, medians[severity], f"{medians[severity]:.0%}",
                    ha="left", va="center", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.03, 1.12)
    ax.set_ylabel("fraction of nodes retained")
    ax.set_xlabel("Strahler severity")
    ax.set_title("How much of each neuron survives pruning\n(all 4105 neurons)")
    save(fig, "F4b_pruning_retention")


def f4c_strahler_vs_centrifugal(skel):
    counts = skel[skel["condition"] == "none"].groupby("body_id").size()
    body_id = counts.sort_values().index[len(counts) // 2]
    rows = skel[skel["body_id"] == body_id]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    for ax, definition in zip(axes, ["strahler", "centrifugal"]):
        conditions = ["none"] + [f"{definition}_{s}" for s in SEVERITY_ORDER[1:]]
        _draw_overlay(ax, rows, conditions, SEVERITY_ORDER)
        ax.set_title(f"{definition.capitalize()} order")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                          color=SEVERITY_COLORS[s], label=s) for s in SEVERITY_ORDER]
    axes[1].legend(handles=handles, title="severity", fontsize=11, loc="upper right")
    fig.suptitle(f"The two definitions remove different structure (neuron {body_id}):\n"
                 "Strahler strips twigs everywhere, centrifugal truncates past a fixed "
                 "depth from the soma", y=1.03)
    save(fig, "F4c_strahler_vs_centrifugal")


def f5_survivorship(check):
    long = check.melt(id_vars=["method", "tag", "n"],
                      value_vars=["naive_delta", "true_delta"],
                      var_name="comparison", value_name="delta")
    long["comparison"] = long["comparison"].map(
        {"naive_delta": "vs full-set baseline (naive)",
         "true_delta": "vs same-neuron baseline (correct)"})

    g = sns.catplot(data=long, x="method", y="delta", hue="comparison", col="tag",
                    kind="bar", height=5, aspect=1.1, palette="colorblind", sharey=True)
    for ax in g.axes.flat:
        ax.axhline(0, color="k", linewidth=1)
        ax.tick_params(axis="x", rotation=30)
    g.set_axis_labels("", "change in MCC vs baseline")
    g.set_titles("{col_name}")
    g.figure.suptitle("Holding the test set fixed makes NBLAST's gain larger, "
                      "not smaller", y=1.05)
    save(g.figure, "F5_survivorship_correction")


def main():
    tidy = load("performance_tidy.csv")
    samples = load("distance_samples.csv.gz")
    skel = load("skeleton_examples.csv.gz")
    retention = load("retention.csv")
    check = pd.read_csv(RESULTS_DIR / "survivorship_check.csv")

    if tidy is not None:
        f1_headline(tidy)
        f2_metric_robustness(tidy)
    if samples is not None:
        f3_distance_distributions(samples)
    if skel is not None:
        f4_before_after(skel)
        f4c_strahler_vs_centrifugal(skel)
    if retention is not None:
        f4b_retention(retention)
    f5_survivorship(check)
    print("Done.")


if __name__ == "__main__":
    main()
