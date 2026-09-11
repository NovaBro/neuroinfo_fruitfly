"""Compute UGW distances for a single block-pair of the chunked T1_[LR]
comparison (see ugw_common.py for the chunk-boundary scheme). Designed to be
invoked once per SLURM array task with --pair-index set to
$SLURM_ARRAY_TASK_ID (0..230 for the current CHUNK_SIZE=200).

Safe to re-run: if this block's checkpoint CSV already exists, the script
prints a message and exits immediately without recomputing anything, so a
resubmitted array (or a resubmission of just the missing indices) never
redoes finished work.
"""

import argparse
import functools
import glob
import time

print = functools.partial(print, flush=True)

from cajal.ugw import UGW, _multicore
import numpy as np
import pandas as pd

import ugw_common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-index", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading full icdm array from {common.ICDM_GEO_CSV}")
    cell_ids, dmats = common.load_full_dmats()
    n_cells = len(cell_ids)

    i, j, (i_start, i_end), (j_start, j_end) = common.pair_index_to_groups(
        args.pair_index, n_cells
    )
    n_pairs = len(common.all_pairs(len(common.get_group_bounds(n_cells))))
    print(f"pair-index {args.pair_index}/{n_pairs - 1} -> groups ({i}, {j})")

    out_path = common.block_checkpoint_path(i, j)
    if out_path.exists():
        print(f"Block ({i},{j}) already done, skipping: {out_path}")
        return

    rho = common.load_rho()
    print(f"Loaded global rho={rho}")

    i_ids = cell_ids[i_start:i_end]
    if i == j:
        block_ids = list(i_ids)
        sub_dmats = dmats[i_start:i_end]
    else:
        j_ids = cell_ids[j_start:j_end]
        block_ids = list(i_ids) + list(j_ids)
        sub_dmats = np.concatenate([dmats[i_start:i_end], dmats[j_start:j_end]], axis=0)

    print(f"Computing UGW for block ({i},{j}): {len(block_ids)} cells "
          f"({'diagonal' if i == j else 'off-diagonal'})")
    t0 = time.time()
    UGW_multicore = UGW(_multicore)
    full_block_matrix = UGW_multicore.ugw_armijo_pairwise(
        mass_kept=common.MASS_KEPT,
        eps=common.EPS,
        dmats=sub_dmats,
        rho=rho,
        increasing_ratio=None,  # disable cajal's uncapped NaN-retry loop (see ugw_common.py note)
        as_matrix=True,
    )
    elapsed = time.time() - t0
    n_nan = int(np.isnan(full_block_matrix).sum())
    print(f"UGW computation for block ({i},{j}) finished in {elapsed:.1f}s "
          f"({n_nan} NaN entries out of {full_block_matrix.size})")

    if i == j:
        block_result = full_block_matrix
        row_ids, col_ids = block_ids, block_ids
    else:
        n_i = i_end - i_start
        block_result = full_block_matrix[:n_i, n_i:]
        row_ids, col_ids = list(i_ids), list(j_ids)

    common.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(block_result, index=row_ids, columns=col_ids).to_csv(out_path)

    n_done = len(glob.glob(str(common.CHUNKS_DIR / "block_*_*.csv")))
    print(f"Completed block ({i},{j}): {n_done}/{n_pairs} done, elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
