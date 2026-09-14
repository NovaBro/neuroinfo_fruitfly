"""Verify the mirroring transform in mirror_skeletons.py was legitimate.

Two independent checks, because eyeballing a plot cannot prove a transform
is exact and a numeric assert cannot show whether homologs actually align:

1. NUMERIC (exactness). For every neuron, compare the original and mirrored
   coordinate *sets* -- navis.write_swc renumbers nodes on write, so a
   row-by-row comparison silently compares different anatomical points and
   reports enormous bogus deviations. Both sides are sorted lexicographically
   first. Left-side neurons must be untouched; right-side neurons must match
   the original reflected as (2*midline - x, y, z).

2. VISUAL (legitimacy). Population soma scatter before/after, example
   neurons overlaid on their own reflection, and -- the informative one --
   a mirrored right neuron drawn on top of a left neuron of the SAME
   hemilineage, which is the alignment the whole idea depended on.
"""

import functools
import os
import pickle
from pathlib import Path

print = functools.partial(print, flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MANC_DIR = Path.cwd() / "data" / "MANC"
RESULTS_DIR = MANC_DIR / "results"
INSTANCE = "T[1]_[LR]"
ORIG_DIR = MANC_DIR / f"Skeletons_{INSTANCE}"
MIRR_DIR = MANC_DIR / f"Skeletons_{INSTANCE}_mirrored_raw"
OUT_DIR = RESULTS_DIR / "mirroring_verification"


def read_nodes(path):
    """(node_id, x, y, z, parent) as a float array, straight from SWC text."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 7:
                rows.append((float(p[0]), float(p[2]), float(p[3]), float(p[4]), float(p[6])))
    return np.array(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(ORIG_DIR) if f.endswith(".swc"))

    # Recompute the midline exactly as mirror_skeletons.py did, rather than
    # hardcoding it, so this check is independent of that run's logging.
    soma_x = {}
    for f in files:
        arr = read_nodes(ORIG_DIR / f)
        root = arr[arr[:, 4] == -1]
        if len(root):
            soma_x[f] = root[0, 1]
    midline = (min(soma_x.values()) + max(soma_x.values())) / 2
    print(f"Recomputed midline x = {midline}")

    # ---------- 1. numeric exactness ----------
    worst = {"left_moved": 0.0, "right_yz_moved": 0.0, "right_x_err": 0.0}
    n_left = n_right = n_missing = n_count_mismatch = 0

    for f in files:
        mpath = MIRR_DIR / f
        if not mpath.exists():
            n_missing += 1
            continue
        a, b = read_nodes(ORIG_DIR / f), read_nodes(mpath)
        if a.shape != b.shape:
            n_count_mismatch += 1
            continue

        def as_sorted_xyz(arr):
            xyz = arr[:, 1:4]
            return xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]

        got = as_sorted_xyz(b)
        if soma_x.get(f, midline) > midline:
            n_right += 1
            reflected = a.copy()
            reflected[:, 1] = 2 * midline - reflected[:, 1]
            expected = as_sorted_xyz(reflected)
            worst["right_yz_moved"] = max(worst["right_yz_moved"],
                                          np.abs(expected[:, 1:] - got[:, 1:]).max())
            worst["right_x_err"] = max(worst["right_x_err"],
                                       np.abs(expected[:, 0] - got[:, 0]).max())
        else:
            n_left += 1
            worst["left_moved"] = max(worst["left_moved"],
                                      np.abs(as_sorted_xyz(a) - got).max())

    print(f"\nNumeric check over {n_left + n_right} neurons "
          f"({n_left} left, {n_right} right, {n_missing} missing, "
          f"{n_count_mismatch} node-table mismatches)")
    print(f"  max coordinate change on LEFT neurons (must be 0):        {worst['left_moved']}")
    print(f"  max y/z change on RIGHT neurons (must be 0):              {worst['right_yz_moved']}")
    print(f"  max |x' - (2*midline - x)| on RIGHT neurons (must be 0):  {worst['right_x_err']}")
    verdict = (worst["left_moved"] == 0 and worst["right_yz_moved"] == 0
               and worst["right_x_err"] == 0 and n_count_mismatch == 0 and n_missing == 0)
    print(f"  => transform is an exact pure x-reflection: {verdict}")

    pd.DataFrame([{**worst, "n_left": n_left, "n_right": n_right,
                   "n_missing": n_missing, "n_node_mismatch": n_count_mismatch,
                   "midline": midline, "exact_x_reflection": verdict}]
                 ).to_csv(OUT_DIR / "numeric_check.csv", index=False)

    # ---------- 2. visual ----------
    sx = pd.Series(soma_x)
    side = (sx > midline).map({True: "R", False: "L"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # Read the ACTUAL mirrored somata rather than recomputing the transform,
    # so this plot verifies the output files instead of restating the intent.
    orig_soma, mirr_soma = {}, {}
    for f in files:
        arr = read_nodes(ORIG_DIR / f)
        root = arr[arr[:, 4] == -1]
        if len(root):
            orig_soma[f] = (root[0, 1], root[0, 2])
        if (MIRR_DIR / f).exists():
            marr = read_nodes(MIRR_DIR / f)
            mroot = marr[marr[:, 4] == -1]
            if len(mroot):
                mirr_soma[f] = (mroot[0, 1], mroot[0, 2])
    orig_df = pd.DataFrame(orig_soma, index=["x", "y"]).T
    mirr_df = pd.DataFrame(mirr_soma, index=["x", "y"]).T

    for ax, (df, mirrored) in zip(axes, [(orig_df, False), (mirr_df, True)]):
        for s, c in (("L", "tab:blue"), ("R", "tab:red")):
            sel = df.index[side.reindex(df.index) == s]
            ax.scatter(df.loc[sel, "x"], df.loc[sel, "y"], s=4, alpha=0.5, c=c,
                       label=f"{s} soma (original side)")
        ax.axvline(midline, color="k", linestyle="--", alpha=0.6, label="midline")
        ax.set_title("after mirroring right onto left" if mirrored else "original")
        ax.set_xlabel("x")
        ax.legend(fontsize=8, markerscale=2)
    axes[0].set_ylabel("y")
    fig.suptitle("Soma positions before and after mirroring\n"
                 "(right-side somata should fold onto the left half)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "soma_positions.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUT_DIR / 'soma_positions.png'}")

    # Example neurons: a right neuron with its own reflection, and a
    # same-hemilineage left neuron for the alignment test.
    with open(RESULTS_DIR / "id_to_class.pkl", "rb") as fh:
        id_to_class = pickle.load(fh)
    by_id = {int(f[:-4]): f for f in files}
    cls = pd.Series({i: id_to_class[i] for i in by_id if i in id_to_class})
    side_by_id = pd.Series({i: side[by_id[i]] for i in cls.index})

    examples = []
    for hemilineage in cls.value_counts().index:
        members = cls[cls == hemilineage].index
        rights = [i for i in members if side_by_id[i] == "R"]
        lefts = [i for i in members if side_by_id[i] == "L"]
        if rights and lefts:
            examples.append((hemilineage, rights[0], lefts[0]))
        if len(examples) == 3:
            break

    # Shared axes across every panel: these are meant to be compared directly,
    # so autoscaling each one independently would hide the real offsets.
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 4.2 * len(examples)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for row, (hemilineage, rid, lid) in enumerate(examples):
        ro = read_nodes(ORIG_DIR / by_id[rid])
        rm = read_nodes(MIRR_DIR / by_id[rid])
        lo = read_nodes(ORIG_DIR / by_id[lid])

        ax = axes[row][0]
        ax.scatter(ro[:, 1], ro[:, 2], s=1, c="tab:red", label=f"right neuron {rid} (original)")
        ax.scatter(rm[:, 1], rm[:, 2], s=1, c="tab:orange", label=f"same neuron, mirrored")
        ax.axvline(midline, color="k", linestyle="--", alpha=0.6)
        ax.set_title(f"{hemilineage}: reflection of one neuron")
        ax.legend(fontsize=7, markerscale=6)

        ax = axes[row][1]
        ax.scatter(lo[:, 1], lo[:, 2], s=1, c="tab:blue", label=f"left neuron {lid} (original)")
        ax.scatter(rm[:, 1], rm[:, 2], s=1, c="tab:orange", label=f"right neuron {rid}, mirrored")
        ax.axvline(midline, color="k", linestyle="--", alpha=0.6)
        ax.set_title(f"{hemilineage}: does the mirrored right neuron land on its left homolog?")
        ax.legend(fontsize=7, markerscale=6)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for row in axes:
        row[0].set_ylabel("y")
    fig.suptitle("Mirroring verification: single neurons (xy projection)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "example_neurons.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'example_neurons.png'}")


if __name__ == "__main__":
    main()
