"""Subset the cached full-MANC NBLAST all-by-all scores down to the T1_[LR]
body IDs already used for the geodesic/euclidean GW comparison, and convert
the NBLAST similarity scores into a distance matrix.

Reuses the exact normalize/symmetrize/1-minus conversion already prototyped
in manc.ipynb (cells ~44, 51, 87-88) rather than recomputing NBLAST from
scratch -- data/MANC/nblast_allbyall_scores_raw.parquet already covers the
full MANC dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"


def main():
    neuropil_regex = "T[1]"
    side_regex = "[LR]"
    instance_file_name = f"{neuropil_regex}_{side_regex}"

    icdm_geo_csv = MANC_DIR / f"{instance_file_name}_icdm_geodesic.csv"
    nblast_parquet = MANC_DIR / "nblast_allbyall_scores_raw.parquet"
    out_csv = MANC_DIR / f"{instance_file_name}_nblast_dist.csv"

    print(f"Loading target body IDs from {icdm_geo_csv}")
    target_ids = pd.read_csv(icdm_geo_csv)["cell_id"].unique()
    print(f"Target body IDs: {len(target_ids)}")

    print(f"Loading raw NBLAST scores from {nblast_parquet}")
    raw_score_matrix = pd.read_parquet(nblast_parquet)
    print(f"Raw NBLAST matrix shape: {raw_score_matrix.shape}")


    # Self-normalize each row by its self-match score, then symmetrize
    # (NBLAST is directional query->target, so forward/reverse scores differ).
    score_matrix_self_normalized = raw_score_matrix.div(
        np.diag(raw_score_matrix), axis=0
    )
    score_matrix_self_normalized_mean = (
        score_matrix_self_normalized + score_matrix_self_normalized.T
    ) / 2



    # The NBLAST parquet is indexed by body ID as strings; intersect with the
    # target body IDs (as strings) rather than assuming full coverage.
    target_ids_str = [str(x) for x in target_ids]
    good_indices = score_matrix_self_normalized_mean.index.intersection(target_ids_str)
    print(f"Body IDs present in both NBLAST matrix and target set: {len(good_indices)}")

    nblast_matrix = score_matrix_self_normalized_mean.loc[good_indices, good_indices]

    # Convert similarity -> distance (1 = identical becomes 0).
    nblast_dist = (1 - nblast_matrix)

    try:
        assert (np.diag(nblast_dist) == 0).all()
    except:
        arr = nblast_dist.to_numpy(copy=True)
        np.fill_diagonal(arr, 0.0)
        nblast_dist = pd.DataFrame(arr, index=nblast_dist.index, columns=nblast_dist.columns)   


    # Standardize on an integer body-ID index/columns, matching the
    # geodesic/euclidean/UGW distance matrices.
    nblast_dist.index = nblast_dist.index.astype(int)
    nblast_dist.columns = nblast_dist.columns.astype(int)

    print(f"Final NBLAST distance matrix shape: {nblast_dist.shape}")
    nblast_dist.to_csv(out_csv)
    print(f"Saved NBLAST distance matrix to {out_csv}")


if __name__ == "__main__":
    main()
