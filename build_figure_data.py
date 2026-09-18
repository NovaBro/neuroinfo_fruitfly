"""Extract everything the final figures need, once, into small plot-ready
tables under data/MANC/results/figure_data/.

Drawing a chart takes seconds; its inputs here are twenty-odd ~300MB
pairwise-distance matrices and thousands of SWC skeletons. Keeping the
extraction in a separate stage means revising a figure never costs another
cluster job -- plot_final_figures.py and plot_interactive_report.py read only
the outputs of this script.

Outputs:
  performance_tidy.csv    method x condition x k x metric, from every sweep
  distance_samples.csv.gz subsampled pairwise distances per method/condition
  skeleton_examples.csv.gz node coords for exemplar neurons, every condition
  retention.csv           node counts per neuron per condition
"""

import functools
import os
import pickle
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

MANC_DIR = Path.cwd() / "data" / "MANC"
RESULTS_DIR = MANC_DIR / "results"
OUT_DIR = RESULTS_DIR / "figure_data"
INSTANCE = "T[1]_[LR]"

STRAHLER = ["mild", "moderate", "tertiary", "aggressive"]
SEVERITY_RANK = {"none": 0, "mild": 1, "moderate": 2, "tertiary": 3, "aggressive": 4}
METHODS = ["NBLAST", "Geodesic GW", "Euclidean GW", "UGW"]
N_SAMPLES = 100_000
N_EXEMPLARS = 9
UGW_EPS = "3000000"


def skeleton_dir(condition):
    if condition == "none":
        return MANC_DIR / f"Skeletons_{INSTANCE}"
    return MANC_DIR / f"Skeletons_{INSTANCE}_pruned_{condition}"


def matrix_path(method, condition):
    """Distance-matrix path. Note compute_nblast.py writes every variant under
    a legacy '_pruned_' infix, including ones involving no pruning."""
    stem = f"{INSTANCE}" if condition == "none" else f"{INSTANCE}_pruned_{condition}"
    return {
        "NBLAST": MANC_DIR / f"{stem}_nblast_dist.csv",
        "Geodesic GW": MANC_DIR / f"{stem}_gw_dist_geodesic.csv",
        "Euclidean GW": MANC_DIR / f"{stem}_gw_dist_euclidean.csv",
        "UGW": MANC_DIR / f"{stem}_ugw_dist_mass80_eps{UGW_EPS}.csv",
    }[method]


def sample_distances(method, path, rng, labels):
    """Return sampled pairwise distances tagged by whether the two neurons share
    a hemilineage. GW matrices are stored flat (one row per pair) so their body
    ids are already columns; NBLAST/UGW are square, so take the upper triangle.

    Sampling is stratified: same-hemilineage pairs are only a few percent of all
    pairs, so a uniform sample would leave too few of them to estimate the
    within-class distribution stably."""
    if "GW" in method and "UGW" not in method:
        flat = pd.read_csv(path)
        flat.columns = ["first", "second", "distance"][: len(flat.columns)]
        a = flat["first"].to_numpy()
        b = flat["second"].to_numpy()
        d = flat["distance"].to_numpy()
    else:
        square = pd.read_csv(path, index_col=0)
        ids = square.index.astype(int).to_numpy()
        arr = square.to_numpy()
        iu = np.triu_indices_from(arr, k=1)
        a, b, d = ids[iu[0]], ids[iu[1]], arr[iu]

    lab = pd.Series(labels)
    la = lab.reindex(a).to_numpy()
    lb = lab.reindex(b).to_numpy()
    keep = np.isfinite(d) & pd.notna(la) & pd.notna(lb)
    a, b, d, la, lb = a[keep], b[keep], d[keep], la[keep], lb[keep]
    same = la == lb

    parts = []
    for flag in (True, False):
        idx = np.flatnonzero(same == flag)
        if len(idx) == 0:
            continue
        take = min(len(idx), N_SAMPLES // 2)
        pick = rng.choice(idx, take, replace=False)
        parts.append(pd.DataFrame({"distance": d[pick], "same_class": flag}))
    return pd.concat(parts, ignore_index=True)


def build_performance_tidy():
    """One tidy table of every k-curve we have, across all sweeps."""
    frames = []

    pruning = RESULTS_DIR / "pruning_sweep_kcurves.csv"
    if pruning.exists():
        df = pd.read_csv(pruning)
        df["definition"] = df["tag"].str.split("_").str[0]
        df["severity"] = df["tag"].str.split("_", n=1).str[1].fillna("none")
        frames.append(df)
    else:
        print(f"WARNING: {pruning.name} missing - rerun evaluate_pruning_sweep.py")

    centrifugal = RESULTS_DIR / "centrifugal_nblast_kcurves.csv"
    if centrifugal.exists():
        df = pd.read_csv(centrifugal)
        df["method"] = "NBLAST"
        df["definition"] = "centrifugal"
        df["severity"] = df["tag"].str.replace("centrifugal_", "", regex=False)
        frames.append(df)
    else:
        print(f"WARNING: {centrifugal.name} missing")

    # Baseline curves live in the original per-method k_sweep_*.csv files.
    for method in METHODS:
        path = RESULTS_DIR / f"k_sweep_{method.replace(' ', '_')}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["method"] = method
        df["tag"] = "baseline"
        df["definition"] = "baseline"
        df["severity"] = "none"
        frames.append(df)

    tidy = pd.concat(frames, ignore_index=True)
    tidy["severity_rank"] = tidy["severity"].map(SEVERITY_RANK)
    out = OUT_DIR / "performance_tidy.csv"
    tidy.to_csv(out, index=False)
    print(f"Saved {out} ({len(tidy)} rows, "
          f"{tidy['method'].nunique()} methods, {tidy['tag'].nunique()} conditions)")


def build_distance_samples():
    rng = np.random.default_rng(0)
    with open(RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        labels = pickle.load(f)
    rows = []
    conditions = ["none"] + [f"strahler_{s}" for s in STRAHLER] + [f"centrifugal_{s}" for s in STRAHLER]
    for condition in conditions:
        for method in METHODS:
            # GW/UGW were only ever run for the Strahler arm.
            if condition.startswith("centrifugal") and method != "NBLAST":
                continue
            path = matrix_path(method, condition)
            if not path.exists():
                print(f"  skip {method} / {condition}: {path.name} not found")
                continue
            vals = sample_distances(method, path, rng, labels)
            vals["method"] = method
            vals["condition"] = condition
            vals["definition"] = "baseline" if condition == "none" else condition.split("_")[0]
            vals["severity"] = "none" if condition == "none" else condition.split("_", 1)[1]
            rows.append(vals)
            print(f"  {method} / {condition}: {len(vals)} sampled "
                  f"({int(vals['same_class'].sum())} within-hemilineage)")

    samples = pd.concat(rows, ignore_index=True)
    out = OUT_DIR / "distance_samples.csv.gz"
    samples.to_csv(out, index=False, compression="gzip")
    print(f"Saved {out} ({len(samples)} rows)")


def node_count(path):
    with open(path) as f:
        return sum(1 for line in f if not line.startswith("#") and line.strip())


def build_skeletons_and_retention():
    conditions = ["none"] + [f"strahler_{s}" for s in STRAHLER] + [f"centrifugal_{s}" for s in STRAHLER]
    available = {}
    for condition in conditions:
        d = skeleton_dir(condition)
        if not d.exists():
            print(f"  skip {condition}: {d.name} not found")
            continue
        available[condition] = set(f for f in os.listdir(d) if f.endswith(".swc"))
        print(f"  {condition}: {len(available[condition])} skeletons")

    # Retention for every neuron, not just the exemplars -- it is only line
    # counting, and it lets the retention figure show population spread.
    retention = []
    for condition, files in available.items():
        d = skeleton_dir(condition)
        for fname in files:
            retention.append({
                "body_id": int(fname[:-4]),
                "condition": condition,
                "definition": "baseline" if condition == "none" else condition.split("_")[0],
                "severity": "none" if condition == "none" else condition.split("_", 1)[1],
                "n_nodes": node_count(d / fname),
            })
    ret = pd.DataFrame(retention)
    original = ret[ret["condition"] == "none"].set_index("body_id")["n_nodes"]
    ret["frac_retained"] = ret["n_nodes"] / ret["body_id"].map(original)
    ret["severity_rank"] = ret["severity"].map(SEVERITY_RANK)
    ret.to_csv(OUT_DIR / "retention.csv", index=False)
    print(f"Saved {OUT_DIR / 'retention.csv'} ({len(ret)} rows)")

    # Exemplars must survive every condition (strahler_aggressive drops 702),
    # and should span the size range so the effect is shown on simple and
    # highly-arborized cells alike rather than one flattering case.
    common = set.intersection(*available.values())
    sizes = original[[int(f[:-4]) for f in common]].sort_values()
    picks = sizes.iloc[np.linspace(0, len(sizes) - 1, N_EXEMPLARS).astype(int)]
    print(f"{len(common)} neurons present in all conditions; "
          f"exemplars span {picks.min()}-{picks.max()} nodes")

    coords = []
    for body_id in picks.index:
        for condition in available:
            path = skeleton_dir(condition) / f"{body_id}.swc"
            arr = np.loadtxt(path, comments="#", usecols=(2, 3, 4), ndmin=2)
            coords.append(pd.DataFrame({
                "body_id": body_id,
                "condition": condition,
                "definition": "baseline" if condition == "none" else condition.split("_")[0],
                "severity": "none" if condition == "none" else condition.split("_", 1)[1],
                "x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2],
            }))
    skel = pd.concat(coords, ignore_index=True)
    out = OUT_DIR / "skeleton_examples.csv.gz"
    skel.to_csv(out, index=False, compression="gzip")
    print(f"Saved {out} ({len(skel)} nodes, {skel['body_id'].nunique()} neurons)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== performance ==")
    build_performance_tidy()
    print("== skeletons + retention ==")
    build_skeletons_and_retention()
    print("== distance samples ==")
    build_distance_samples()
    print("Done.")


if __name__ == "__main__":
    main()
