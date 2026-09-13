"""Compute NBLAST all-by-all distances from scratch for an arbitrary skeleton
directory + output tag. Unlike subset_nblast.py (which subsets the cached
full-MANC parquet, valid only for the unpruned baseline geometry), a pruned
skeleton's geometry has actually changed, so NBLAST must be recomputed from
scratch here -- there's no cached matrix to subset from for a pruned
condition.

Reuses the exact self-normalize / symmetrize / 1-minus conversion already
used in subset_nblast.py (duplicated here rather than refactored into a
shared module, so subset_nblast.py itself stays untouched).

Usage: python compute_nblast.py --skeleton-dir <dir> --tag <tag>
"""

import argparse
import functools
import os
from pathlib import Path

print = functools.partial(print, flush=True)

import navis
import numpy as np
import pandas as pd

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"

INSTANCE_FILE_NAME = "T[1]_[LR]"
# nm -> um, matching manc.ipynb's `skels_um = skels / (1000 / 8)` (MANC
# skeleton coordinates are stored in 8nm-voxel units; SWC header confirms
# "units": "8 nanometer").
NM_PER_VOXEL_OVER_UM = 1000 / 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--use-alpha", action="store_true",
                        help="Weight NBLAST scores by local neighbourhood linearity.")
    parser.add_argument("--resample", type=float, default=None,
                        help="Uniformly resample skeletons to this spacing (um) before dotprops.")
    args = parser.parse_args()

    skeleton_dir = Path(args.skeleton_dir)
    tag = args.tag
    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    out_csv = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_nblast_dist.csv"

    if out_csv.exists():
        print(f"{out_csv} already exists, skipping")
        return

    swc_paths = [
        str(skeleton_dir / f)
        for f in sorted(os.listdir(skeleton_dir))
        if f.endswith(".swc")
    ]
    print(f"Reading {len(swc_paths)} skeletons from {skeleton_dir}")
    skels = navis.read_swc(swc_paths)
    skels_um = skels / NM_PER_VOXEL_OVER_UM

    print(f"Building dotprops (resample={args.resample})...")
    dps = navis.make_dotprops(skels_um, resample=args.resample or False)

    print(f"Running all-by-all NBLAST on {num_cores} cores (use_alpha={args.use_alpha})...")
    raw_score_matrix = navis.nblast_allbyall(
        dps, n_cores=num_cores, normalized=False, use_alpha=args.use_alpha
    )

    score_matrix_self_normalized = raw_score_matrix.div(
        np.diag(raw_score_matrix), axis=0
    )
    score_matrix_self_normalized_mean = (
        score_matrix_self_normalized + score_matrix_self_normalized.T
    ) / 2

    nblast_dist = 1 - score_matrix_self_normalized_mean

    arr = nblast_dist.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    nblast_dist = pd.DataFrame(arr, index=nblast_dist.index, columns=nblast_dist.columns)

    nblast_dist.index = nblast_dist.index.astype(int)
    nblast_dist.columns = nblast_dist.columns.astype(int)

    print(f"Final NBLAST distance matrix shape: {nblast_dist.shape}")
    nblast_dist.to_csv(out_csv)
    print(f"Saved NBLAST distance matrix to {out_csv}")


if __name__ == "__main__":
    main()
