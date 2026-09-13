"""Mirror right-side neurons across the VNC midline so that left/right
homologs of the same hemilineage become directly comparable to NBLAST.

Hemilineage labels carry no side suffix (`19A`, `12B`, ...), so each class
spans both body sides, but left and right homologs are approximate mirror
images -- which NBLAST, comparing raw coordinates, reads as "far apart".
Measured on the unpruned baseline matrix, 79.8% of each neuron's top-10
NBLAST neighbours are on its own side (chance ~50%), so each hemilineage is
effectively split into two half-sized clusters. Reflecting the right side
onto the left should roughly double usable within-class neighbour density.

Mirroring is geometric and pruning is topological, so they commute: a
pruned skeleton directory can be mirrored directly rather than re-pruned.

Usage: python mirror_skeletons.py --skeleton-dir <dir> --tag <tag>
"""

import argparse
import functools
import os
from pathlib import Path

print = functools.partial(print, flush=True)

import navis

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"

INSTANCE_FILE_NAME = "T[1]_[LR]"


def soma_x(swc_path):
    """x coordinate of the root node, read straight from the SWC text --
    much cheaper than a full navis parse just to decide which side a neuron
    is on.
    """
    with open(swc_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7 and parts[6] == "-1":
                return float(parts[2])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    skeleton_dir = Path(args.skeleton_dir)
    out_dir = MANC_DIR / f"Skeletons_{INSTANCE_FILE_NAME}_mirrored_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    swc_files = sorted(f for f in os.listdir(skeleton_dir) if f.endswith(".swc"))
    print(f"Found {len(swc_files)} skeletons in {skeleton_dir}")

    soma_xs = {f: soma_x(skeleton_dir / f) for f in swc_files}
    valid = [x for x in soma_xs.values() if x is not None]
    # These are MANC EM coordinates (8nm voxels), not JRC2018 template space,
    # so there is no canonical midline to look up -- take the midpoint of the
    # soma x-extent. A single-plane reflection is an approximation if the true
    # mirror plane is tilted; the same-side-neighbour diagnostic measures
    # whether it is good enough.
    midline = (min(valid) + max(valid)) / 2
    print(f"Midline x = {midline:.1f} (soma x range {min(valid):.1f}-{max(valid):.1f})")

    n_mirrored, n_copied, n_skipped, n_failed = 0, 0, 0, 0
    for fname in swc_files:
        out_path = out_dir / fname
        if out_path.exists():
            n_skipped += 1
            continue

        try:
            neuron = navis.read_swc(str(skeleton_dir / fname))
            if soma_xs[fname] is not None and soma_xs[fname] > midline:
                neuron.nodes["x"] = 2 * midline - neuron.nodes["x"]
                n_mirrored += 1
            else:
                n_copied += 1
            navis.write_swc(neuron, str(out_path))
        except Exception as e:
            print(f"  {fname}: failed ({e}), skipping")
            n_failed += 1

    print(
        f"{args.tag}: {n_mirrored} mirrored (right side), {n_copied} copied as-is "
        f"(left side), {n_failed} failed, {n_skipped} already done -> {out_dir}"
    )


if __name__ == "__main__":
    main()
