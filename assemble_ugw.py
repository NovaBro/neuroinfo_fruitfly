"""Stitch all completed block-pair checkpoint CSVs (data/MANC/ugw_chunks/) into
the single final square UGW distance matrix, in the exact format the existing
get_common_ids.py / evaluate_metrics.py already expect:
data/MANC/T[1]_[LR]_ugw_dist_mass80_eps100.csv

Refuses to run (prints the missing block indices and exits non-zero) unless
every expected block-pair checkpoint exists.
"""

import functools
import sys

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

import ugw_common as common


def main():
    print(f"Loading cell order from {common.ICDM_GEO_CSV}")
    cell_ids, dmats = common.load_full_dmats()
    n_cells = len(cell_ids)

    bounds = common.get_group_bounds(n_cells)
    pairs = common.all_pairs(len(bounds))
    print(f"{n_cells} cells, {len(bounds)} groups, {len(pairs)} expected block-pairs")

    missing = [
        (i, j) for (i, j) in pairs if not common.block_checkpoint_path(i, j).exists()
    ]
    if missing:
        print(f"Missing {len(missing)}/{len(pairs)} block-pairs:")
        for i, j in missing:
            pair_index = pairs.index((i, j))
            print(f"  pair-index {pair_index}: block ({i},{j})")
        print(
            "Resubmit these before assembling, e.g.: "
            f"sbatch --array={','.join(str(pairs.index(p)) for p in missing)} run_ugw_array.slurm"
        )
        sys.exit(1)

    print("All block-pairs present. Assembling final matrix...")
    full = pd.DataFrame(0.0, index=cell_ids, columns=cell_ids)
    for i, j in pairs:
        block = pd.read_csv(common.block_checkpoint_path(i, j), index_col=0)
        block.columns = block.columns.astype(int)
        if i == j:
            full.loc[block.index, block.columns] = block.values
        else:
            full.loc[block.index, block.columns] = block.values
            full.loc[block.columns, block.index] = block.values.T

    # pandas copy-on-write leaves .values read-only even after .copy(); force
    # an independent, genuinely-writable ndarray instead.
    arr = full.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    full = pd.DataFrame(arr, index=full.index, columns=full.columns)

    n_nan = int(np.isnan(full.values).sum())
    print(f"Final matrix shape: {full.shape} ({n_nan} NaN entries, from non-converged pairs)")
    full.to_csv(common.FINAL_UGW_CSV)
    print(f"Saved {common.FINAL_UGW_CSV}")


if __name__ == "__main__":
    main()
