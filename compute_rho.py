"""Precompute a single global UGW `rho` value from the full T1_[LR] icdm
array, so every chunked ugw_block.py call can pass the SAME fixed rho
instead of each block independently (and inconsistently) re-estimating it
from whatever subset of cells it happens to see.

Run once, before any ugw_block.py array tasks. Safe to re-run: skips work if
data/MANC/ugw_rho.json already exists.
"""

import functools
import time

print = functools.partial(print, flush=True)

from cajal.ugw import estimate_distr, rho_of

import ugw_common as common


def main():
    if common.RHO_JSON.exists():
        print(f"{common.RHO_JSON} already exists, skipping. rho={common.load_rho()}")
        return

    print(f"Loading full icdm array from {common.ICDM_GEO_CSV}")
    t0 = time.time()
    cell_ids, dmats = common.load_full_dmats()
    print(f"Loaded {len(cell_ids)} cells, dmats shape {dmats.shape} ({time.time() - t0:.1f}s)")

    print("Estimating global gw_cost quantile (sample_size=200, quantile=0.15)...")
    t0 = time.time()
    gw_cost = estimate_distr(dmats, sample_size=200, quantile=0.15)
    rho = rho_of(gw_cost, common.MASS_KEPT)
    print(f"gw_cost={gw_cost}, mass_kept={common.MASS_KEPT} -> rho={rho} ({time.time() - t0:.1f}s)")

    common.MANC_DIR.mkdir(parents=True, exist_ok=True)
    common.save_rho(rho)
    print(f"Saved rho to {common.RHO_JSON}")


if __name__ == "__main__":
    main()
