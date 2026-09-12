"""Generate pruned-skeleton variants of the MANC T1_[LR] SWC set under two
different "branch order" definitions, each swept across four severities
(mild/moderate/tertiary/aggressive), for the tertiary-branch-pruning
experiment (see the plan file for the full writeup of why two definitions
and these particular thresholds).

- Strahler order (centripetal, from the tips inward): built into navis via
  navis.prune_by_strahler(neuron, to_prune=[...]).
- Centrifugal order (from the soma outward, incrementing every time a path
  crosses a branch point -- the standard L-Measure-style "centrifugal branch
  order"): not built into navis, so a small BFS helper computes it here, then
  navis.subset_neuron keeps only nodes at or below a chosen order.

Writes surviving pruned neurons to
data/MANC/Skeletons_T[1]_[LR]_pruned_<tag>/<bodyId>.swc, one directory per
condition. Skips any neuron that prunes down to an empty/degenerate (<2-node)
tree, logging how many were dropped per condition. Resumable: skips any SWC
that's already been written for a given condition.
"""

import argparse
import functools
import os
from collections import deque
from pathlib import Path

print = functools.partial(print, flush=True)

import navis
import pandas as pd

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"

INSTANCE_FILE_NAME = "T[1]_[LR]"
SRC_DIR = MANC_DIR / f"Skeletons_{INSTANCE_FILE_NAME}"

# to_prune lists for navis.prune_by_strahler: order 1 = terminal twigs.
STRAHLER_CONDITIONS = {
    "strahler_mild": [1],
    "strahler_moderate": [1, 2],
    "strahler_tertiary": [1, 2, 3],
    "strahler_aggressive": [1, 2, 3, 4],
}

# Keep-order-<=K thresholds for the custom centrifugal order below. Chosen
# empirically (not the literal 4/3/2/1) -- see the plan file: MANC neurons'
# centrifugal order commonly reaches 30-70+, so literal order<=4 would strip
# 90%+ of every neuron at every severity. These four values were picked to
# give a comparable node-fraction-kept gradient to the Strahler conditions.
CENTRIFUGAL_CONDITIONS = {
    "centrifugal_mild": 25,
    "centrifugal_moderate": 15,
    "centrifugal_tertiary": 10,
    "centrifugal_aggressive": 6,
}

MIN_NODES = 2


def centrifugal_order(neuron):
    """Branch order counted outward from the root: the root is order 1, and
    order increments by 1 every time a path crosses a node with >1 child
    (a branch point) -- the standard L-Measure "centrifugal order" rule,
    applied to every child at a branch point (not just one continuing path).
    """
    nodes = neuron.nodes
    children = nodes.groupby("parent_id")["node_id"].apply(list).to_dict()
    root_ids = nodes.loc[nodes.parent_id < 0, "node_id"].tolist()

    order = {}
    dq = deque()
    for r in root_ids:
        order[r] = 1
        dq.append(r)
    while dq:
        nid = dq.popleft()
        kids = children.get(nid, [])
        child_order = order[nid] + (1 if len(kids) > 1 else 0)
        for c in kids:
            order[c] = child_order
            dq.append(c)
    return pd.Series(order, name="centrifugal_order")


def prune_strahler(neuron, to_prune):
    return navis.prune_by_strahler(neuron, to_prune=to_prune, inplace=False)


def prune_centrifugal(neuron, keep_le):
    order = centrifugal_order(neuron)
    keep_ids = order[order <= keep_le].index.tolist()
    return navis.subset_neuron(neuron, subset=keep_ids, inplace=False)


def run_condition(tag, prune_fn, swc_files):
    out_dir = MANC_DIR / f"Skeletons_{INSTANCE_FILE_NAME}_pruned_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_skipped, n_survived, n_removed_degenerate, n_failed = 0, 0, 0, 0
    for fname in swc_files:
        out_path = out_dir / fname
        if out_path.exists():
            n_skipped += 1
            continue

        try:
            neuron = navis.read_swc(str(SRC_DIR / fname))
            pruned = prune_fn(neuron)
        except Exception as e:
            print(f"  [{tag}] {fname}: pruning failed ({e}), skipping")
            n_failed += 1
            continue

        if pruned.n_nodes < MIN_NODES:
            n_removed_degenerate += 1
            continue

        navis.write_swc(pruned, str(out_path))
        n_survived += 1

    n_total = len(swc_files)
    print(
        f"{tag}: {n_survived}/{n_total} survived, "
        f"{n_removed_degenerate} removed (too small), {n_failed} failed, "
        f"{n_skipped} already done"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", default=None,
        help="Run only this condition tag (default: all 8 conditions).",
    )
    args = parser.parse_args()

    swc_files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".swc"))
    print(f"Found {len(swc_files)} source skeletons in {SRC_DIR}")

    conditions = {}
    for tag, to_prune in STRAHLER_CONDITIONS.items():
        conditions[tag] = functools.partial(prune_strahler, to_prune=to_prune)
    for tag, keep_le in CENTRIFUGAL_CONDITIONS.items():
        conditions[tag] = functools.partial(prune_centrifugal, keep_le=keep_le)

    if args.condition is not None:
        if args.condition not in conditions:
            raise SystemExit(
                f"Unknown condition {args.condition!r}; choices: {sorted(conditions)}"
            )
        conditions = {args.condition: conditions[args.condition]}

    for tag, prune_fn in conditions.items():
        run_condition(tag, prune_fn, swc_files)


if __name__ == "__main__":
    main()
