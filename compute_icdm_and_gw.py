"""Parameterized SWC -> ICDM -> GW-distance pipeline for an arbitrary skeleton
directory + output tag. Generalizes manc.ipynb's icdm/GW cells (and
calc_GW.py's FANC-only, geodesic-only pattern) so each pruning-sweep
condition can get its own geodesic + euclidean icdm and GW distance matrices
without duplicating that logic 8 times or touching manc.ipynb / calc_GW.py.

Usage: python compute_icdm_and_gw.py --skeleton-dir <dir> --tag <tag>

Safe to re-run: each of the four output files (icdm x {geodesic,euclidean},
gw_dist x {geodesic,euclidean}) is skipped if it already exists.
"""

import argparse
import functools
import os
import shutil
from pathlib import Path

print = functools.partial(print, flush=True)

import pandas as pd
from cajal.run_gw import compute_gw_distance_matrix
from cajal.sample_swc import (
    compute_icdm_all_euclidean,
    compute_icdm_all_geodesic,
    icdm_geodesic,
    read_swc,
)

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"

INSTANCE_FILE_NAME = "T[1]_[LR]"
N_SAMPLE = 50


def gw_dist_csv_is_complete(path, n_cells):
    """A killed/OOM'd job can leave a truncated gw_dist CSV behind (observed
    in practice: a job died mid-write with the file already present but only
    ~40% of rows written). Checking mere existence would silently treat that
    truncated file as done and skip regenerating it -- so verify the row
    count matches the full upper-triangle pair count before trusting it.
    """
    if not path.exists():
        return False
    expected_rows = n_cells * (n_cells - 1) // 2
    actual_rows = sum(1 for _ in open(path)) - 1  # minus header
    if actual_rows != expected_rows:
        print(f"  {path} exists but has {actual_rows}/{expected_rows} rows -- "
              f"treating as incomplete (likely a killed prior job) and recomputing")
        return False
    return True


def quarantine_bad_files(skeleton_dir):
    """Pre-flight check mirroring calc_GW.py's test_swc_files: find SWCs that
    crash CAJAL's geodesic sampler (pruned trees are more likely to hit this
    than the originals) and move them aside so the real, multiprocessed
    compute_icdm_all_* pass below never touches them.
    """
    bad_dir = skeleton_dir.parent / f"Bad_Skeletons_{skeleton_dir.name}"
    swc_files = sorted(f for f in os.listdir(skeleton_dir) if f.endswith(".swc"))
    bad_files = []
    for i, fname in enumerate(swc_files):
        if i % 500 == 0:
            print(f"  preflight check {i}/{len(swc_files)}...")
        try:
            forest, _ = read_swc(str(skeleton_dir / fname))
            icdm_geodesic(forest[0], N_SAMPLE)
        except Exception as e:
            print(f"  preflight failed: {fname} - {e}")
            bad_files.append(fname)

    if bad_files:
        bad_dir.mkdir(exist_ok=True)
        for fname in bad_files:
            shutil.move(str(skeleton_dir / fname), str(bad_dir / fname))
    print(
        f"Preflight: {len(bad_files)}/{len(swc_files)} skeletons quarantined"
        + (f" to {bad_dir}" if bad_files else "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    skeleton_dir = Path(args.skeleton_dir)
    tag = args.tag
    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Using {num_cores} cores, skeleton_dir={skeleton_dir}, tag={tag}")

    out_csv_geo = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_icdm_geodesic.csv"
    out_node_types_geo = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_node_types_geo.npy"
    gw_dist_csv_geo = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_gw_dist_geodesic.csv"
    gw_coupling_npz_geo = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_gw_coupling_geodesic.npz"

    out_csv_euc = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_icdm_euclidean.csv"
    out_node_types_euc = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_node_types_euc.npy"
    gw_dist_csv_euc = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_gw_dist_euclidean.csv"
    gw_coupling_npz_euc = MANC_DIR / f"{INSTANCE_FILE_NAME}_pruned_{tag}_gw_coupling_euclidean.npz"

    if not (out_csv_geo.exists() and out_csv_euc.exists()):
        print(f"Preflight-checking {skeleton_dir} for CAJAL-incompatible skeletons...")
        quarantine_bad_files(skeleton_dir)

    n_cells = len([f for f in os.listdir(skeleton_dir) if f.endswith(".swc")])
    print(f"n_cells in {skeleton_dir} (post-preflight): {n_cells}")

    if out_csv_geo.exists():
        print(f"{out_csv_geo} already exists, skipping geodesic icdm computation")
    else:
        print("Computing geodesic ICDMs...")
        compute_icdm_all_geodesic(
            infolder=str(skeleton_dir),
            out_csv=str(out_csv_geo),
            out_node_types=str(out_node_types_geo),
            n_sample=N_SAMPLE,
            num_processes=num_cores,
        )

    if gw_dist_csv_is_complete(gw_dist_csv_geo, n_cells):
        print(f"{gw_dist_csv_geo} already complete, skipping geodesic GW computation")
    else:
        print("Computing geodesic GW distances...")
        compute_gw_distance_matrix(
            intracell_csv_loc=str(out_csv_geo),
            gw_dist_csv_loc=str(gw_dist_csv_geo),
            gw_coupling_mat_npz_loc=str(gw_coupling_npz_geo),
            num_processes=num_cores,
        )

    if out_csv_euc.exists():
        print(f"{out_csv_euc} already exists, skipping euclidean icdm computation")
    else:
        print("Computing euclidean ICDMs...")
        compute_icdm_all_euclidean(
            infolder=str(skeleton_dir),
            out_csv=str(out_csv_euc),
            out_node_types=str(out_node_types_euc),
            n_sample=N_SAMPLE,
            num_processes=num_cores,
        )

    if gw_dist_csv_is_complete(gw_dist_csv_euc, n_cells):
        print(f"{gw_dist_csv_euc} already complete, skipping euclidean GW computation")
    else:
        print("Computing euclidean GW distances...")
        compute_gw_distance_matrix(
            intracell_csv_loc=str(out_csv_euc),
            gw_dist_csv_loc=str(gw_dist_csv_euc),
            gw_coupling_mat_npz_loc=str(gw_coupling_npz_euc),
            num_processes=num_cores,
        )

    print("Done.")


if __name__ == "__main__":
    main()
