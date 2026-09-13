"""Mirroring mechanism check: does reflecting the right side onto the left
actually let NBLAST match contralateral homologs?

On the unpruned baseline matrix, 79.8% of each neuron's top-10 NBLAST
neighbours sit on its own body side (chance ~50%) and 83.4% of its
correct-hemilineage neighbours do. If the single-plane x-reflection in
mirror_skeletons.py works, those numbers should fall toward ~50% / lower.
If they don't move, the reflection is too crude and no MCC gain should be
expected either -- so this runs before drawing conclusions from MCC alone.
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
INSTANCE_FILE_NAME = "T[1]_[LR]"
RAW_SKEL_DIR = MANC_DIR / f"Skeletons_{INSTANCE_FILE_NAME}"

# compute_nblast.py writes every variant under a legacy "_pruned_" infix,
# including variants that involve no pruning at all (mirrored/alpha/...).
MATRICES = {
    "baseline": MANC_DIR / f"{INSTANCE_FILE_NAME}_nblast_dist.csv",
    "mirrored": MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_mirrored_nblast_dist.csv",
}


def soma_side():
    """Body side per neuron, from the raw (un-mirrored) soma x coordinate, so
    the same ground-truth side labels apply to both matrices."""
    soma_x = {}
    for fname in os.listdir(RAW_SKEL_DIR):
        if not fname.endswith(".swc"):
            continue
        with open(RAW_SKEL_DIR / fname) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7 and parts[6] == "-1":
                    soma_x[int(fname[:-4])] = float(parts[2])
                    break
    sx = pd.Series(soma_x)
    midline = (sx.min() + sx.max()) / 2
    return (sx > midline).map({True: "R", False: "L"})


def main():
    side = soma_side()
    common_ids = np.load(RESULTS_DIR / "common_ids.npy")
    with open(RESULTS_DIR / "id_to_class.pkl", "rb") as f:
        id_to_class = pickle.load(f)

    rows = []
    for name, path in MATRICES.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue

        matrix = pd.read_csv(path, index_col=0)
        matrix.columns = matrix.columns.astype(int)
        ids = np.array(sorted(set(int(i) for i in common_ids) & set(matrix.index)))
        matrix = matrix.loc[ids, ids]

        arr = matrix.to_numpy()
        np.fill_diagonal(arr, np.inf)
        neighbours = np.argsort(arr, axis=1)[:, :10]

        sides = side.reindex(ids).to_numpy()
        labels = np.array([id_to_class[i] for i in ids])
        same_side = (sides[neighbours] == sides[:, None])
        same_class = (labels[neighbours] == labels[:, None])

        row = {
            "variant": name,
            "n": len(ids),
            "frac_same_side": same_side.mean(),
            "frac_same_hemilineage": same_class.mean(),
            "frac_correct_that_are_same_side": (same_class & same_side).sum() / max(same_class.sum(), 1),
        }
        print(f"{name}: same-side={row['frac_same_side']:.3f} (chance ~0.5), "
              f"same-hemilineage={row['frac_same_hemilineage']:.3f}, "
              f"correct-and-same-side={row['frac_correct_that_are_same_side']:.3f}")
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "mirroring_check.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
