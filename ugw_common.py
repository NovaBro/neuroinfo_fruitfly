"""Shared constants/helpers for the chunked, resumable UGW pipeline
(compute_rho.py, ugw_block.py, assemble_ugw.py).

All three scripts must agree on identical chunk boundaries, so the group
partitioning logic lives here once rather than being duplicated three times.
"""

import itertools
import json
import os
from pathlib import Path

import numpy as np
from cajal.utilities import cell_iterator_csv

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"

INSTANCE_FILE_NAME = "T[1]_[LR]"

# Overridable via the PRUNE_TAG env var (e.g. `sbatch
# --export=ALL,PRUNE_TAG=strahler_tertiary ...`) to point the whole pipeline
# at a pruned-skeleton condition's icdm/rho/UGW files instead of the baseline
# (unpruned) ones. Default "" reproduces every path below exactly as it was
# before pruning existed -- the baseline results are never at risk from this.
PRUNE_TAG = os.environ.get("PRUNE_TAG", "")
_prune_suffix = f"_pruned_{PRUNE_TAG}" if PRUNE_TAG else ""

ICDM_GEO_CSV = MANC_DIR / f"{INSTANCE_FILE_NAME}{_prune_suffix}_icdm_geodesic.csv"
# rho depends on the icdm matrix's own GW-cost quantile (see compute_rho.py),
# which changes with the underlying tree geometry -- unlike EPS, it is NOT
# reusable across pruning conditions and must be recomputed per PRUNE_TAG.
RHO_JSON = MANC_DIR / (f"ugw_rho{_prune_suffix}.json" if PRUNE_TAG else "ugw_rho.json")

MASS_KEPT = 0.80
# Overridable via the UGW_EPS env var (e.g. `sbatch --export=ALL,UGW_EPS=300000
# ...`) so an eps sweep doesn't require editing this file per value. Default
# is the production value validated by the eps sweep below.
EPS = float(os.environ.get("UGW_EPS", 3_000_000.0))
CHUNK_SIZE = 200

# Derive the eps-tagged filenames from EPS itself so the chunk directory and
# final CSV name can never drift out of sync with the actual value used
# (this happened once already when EPS moved from 100 -> 1,000,000 but the
# hardcoded "_eps100" filename suffix didn't). Also folds in PRUNE_TAG so a
# pruning-condition sweep never collides with the baseline or with each other.
def eps_suffix(eps):
    return str(int(eps)) if float(eps).is_integer() else str(eps)


def chunks_dir_for_eps(eps):
    return MANC_DIR / f"ugw_chunks_eps{eps_suffix(eps)}{_prune_suffix}"


def final_ugw_csv_for_eps(eps):
    return MANC_DIR / f"{INSTANCE_FILE_NAME}{_prune_suffix}_ugw_dist_mass80_eps{eps_suffix(eps)}.csv"


CHUNKS_DIR = chunks_dir_for_eps(EPS)
FINAL_UGW_CSV = final_ugw_csv_for_eps(EPS)

# EPS was originally 100.0, but the icdm distances here are on a raw scale up
# to ~30000+. With EPS that much smaller than the cost scale, the core
# Sinkhorn/Armijo solver (_ugw_armijo_pairwise_unif) is numerically unstable:
# it leaks memory internally until the node runs out and the Futhark kernel
# crashes with "Assertion is false: M.((gradient) M.< zero)". Raising EPS
# fixes this, but the stability boundary is close: eps=300,000 passed a
# 2-block smoke test but still failed on 10/231 blocks (4.3%) in the full
# run (same deterministic crash signature each time), while eps=1,000,000
# ran cleanly (0/231 failures). A full eps sweep (1e6/3e6/1e7/3e7, all clean
# at 231/231) then compared leave-one-out KNN hemilineage-classification MCC:
# 1e6 -> 0.356, 3e6 -> 0.372 (best), 1e7 -> 0.308, 3e7 -> 0.095 (collapsed --
# the entropic regularization over-smooths the transport plan at that scale,
# visibly compressing the pairwise-distance distribution). 3,000,000 is both
# more accurate and ~2x cheaper to compute than the original 1,000,000
# choice, hence the new default. See data/MANC/results/eps_sweep_*.{csv,png}
# for the full sweep artifacts.
#
# ugw_block.py also calls ugw_armijo_pairwise with increasing_ratio=None:
# cajal's default increasing_ratio=1.1 NaN-retry loop has no iteration cap
# exposed by the API, so disabling it bounds runtime/memory at the cost of
# some pairs coming back as NaN when they don't converge at the chosen EPS.


def load_full_dmats():
    """Load the full ordered (cell_id, icdm) list once, matching the file order
    cell_iterator_csv yields -- the same order used to define chunk boundaries.
    """
    cell_ids, icdms = zip(*cell_iterator_csv(intracell_csv_loc=str(ICDM_GEO_CSV)))
    # cell_iterator_csv yields cell_id as raw strings; cast to int so every
    # downstream index/column built from this list matches the int64 index
    # pd.read_csv infers when re-loading a saved block CSV.
    return [int(c) for c in cell_ids], np.stack(icdms, axis=0)


def get_group_bounds(n_cells, chunk_size=CHUNK_SIZE):
    """Return a list of (start, end) index ranges partitioning range(n_cells)
    into contiguous groups of size chunk_size (last group may be smaller).
    """
    return [
        (start, min(start + chunk_size, n_cells))
        for start in range(0, n_cells, chunk_size)
    ]

def all_pairs(n_groups):
    """All (i, j) group-index pairs with i <= j, in a fixed, deterministic order."""
    return list(itertools.combinations_with_replacement(range(n_groups), 2))


def pair_index_to_groups(pair_index, n_cells, chunk_size=CHUNK_SIZE):
    """Map a flat --pair-index to its (i, j) group indices and their bounds."""
    bounds = get_group_bounds(n_cells, chunk_size)
    pairs = all_pairs(len(bounds))
    i, j = pairs[pair_index]
    return i, j, bounds[i], bounds[j]


def block_checkpoint_path(i, j):
    return CHUNKS_DIR / f"block_{i}_{j}.csv"


def save_rho(rho):
    RHO_JSON.write_text(json.dumps({"rho": rho}))


def load_rho():
    return json.loads(RHO_JSON.read_text())["rho"]
