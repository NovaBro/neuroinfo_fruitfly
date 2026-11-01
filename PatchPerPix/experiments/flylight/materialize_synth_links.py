"""Replace raw/gt_instances symlinks in fisbe/<dataset>/<split>/*.zarr/volumes with real copies.

build_synth_vh.py links the big arrays into vh2308's loose synth_MCFO tree. This copies them
from the read-only SquashFS image instead (mounted as an overlay: /synth_MCFO/v1_u8), so the
dataset no longer depends on the loose copy. Each copy is byte-compared against the current
symlink target (when it still resolves) before the symlink is swapped.
"""
import argparse
import filecmp
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

LINK_ARRAYS = ("raw", "gt_instances")


def tree_files(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)


def same_tree(a, b):
    fa, fb = tree_files(a), tree_files(b)
    if fa != fb:
        return False
    return all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)
               for f in fa)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, help="e.g. fisbe/synth_vh_u8")
    p.add_argument("--src-root", required=True, help="e.g. /synth_MCFO/v1_u8/samples")
    p.add_argument("--splits", default="val,test")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    n_done = n_skip = 0
    for split in [s for s in args.splits.split(",") if s]:
        sdir = os.path.join(args.dataset_root, split)
        for zname in sorted(os.listdir(sdir)):
            if not zname.endswith(".zarr"):
                continue
            for arr in LINK_ARRAYS:
                dst = os.path.join(sdir, zname, "volumes", arr)
                if not os.path.islink(dst):
                    print(f"skip (not a symlink): {dst}")
                    n_skip += 1
                    continue
                src = os.path.join(args.src_root, zname, "volumes", arr)
                if not os.path.isdir(src):
                    raise SystemExit(f"missing in source: {src}")
                if args.dry_run:
                    print(f"would copy {src} -> {dst} (was -> {os.readlink(dst)})")
                    continue
                tmp = dst + ".tmp_copy"
                if os.path.lexists(tmp):
                    shutil.rmtree(tmp)
                shutil.copytree(src, tmp)
                if os.path.isdir(dst):  # old target still resolves: must match byte for byte
                    if not same_tree(tmp, dst):
                        shutil.rmtree(tmp)
                        raise SystemExit(f"MISMATCH sqf vs symlink target: {dst}")
                os.unlink(dst)
                os.rename(tmp, dst)
                print(f"copied {dst}  files={len(tree_files(dst))}")
                n_done += 1
    print(f"done: copied={n_done} skipped={n_skip}")


if __name__ == "__main__":
    main()
