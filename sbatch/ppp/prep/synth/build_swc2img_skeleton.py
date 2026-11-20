#!/usr/bin/env python3
"""Build fisbe/swc2img_1k_v1/{train,val,test}/<brain>.zarr skeletons for PatchPerPix (experiment_v2 EXP 2).

SWC2IMG renders the same synth_MCFO 1k_v1 brains as 1k_v1.1, so each brain gets the split it has in
fisbe/1k_v1.1 (no brain is in train for one dataset and val/test for the other). volumes/raw and
volumes/gt_instances are symlinks into the 4 SquashFS parts in /scratch/gw2467/data/swc2img_1k_v1_sqf, which
mount together at /swc2img_1k_v1/{zarr,meta} (<brain>_rgb.zarr). The parts are listed in
fisbe/swc2img_1k_v1/source_sqf.txt, which pack_ppp_sqf_sbatch.sh (PLAIN_SRC=1) and ppp_job.sh mount.
(The unpacked copy in /scratch/gw2467/data/swc2img_1k_v1/zarr was deleted on 2026-11-09.)
The prepared arrays are added by sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh (SUBTRACT_MEDIAN=1 RAW_DTYPE=float16).

Run inside the container with the 4 parts mounted (the links only resolve there; idempotent; --dry-run counts):
  D=/scratch/gw2467/data/swc2img_1k_v1_sqf
  singularity exec $(for i in 0 1 2 3; do echo --overlay $D/swc2img_1k_v1_part$i.sqf:ro; done) <cuda .sif> \
    python3 sbatch/ppp/prep/synth/build_swc2img_skeleton.py [--dry-run]
"""
import argparse
import os
import shutil
import sys

SRC = "/swc2img_1k_v1/zarr"
SQF_PARTS = [f"/scratch/gw2467/data/swc2img_1k_v1_sqf/swc2img_1k_v1_part{i}.sqf" for i in range(4)]
SPLIT_REF = "fisbe/1k_v1.1"
OUT = "fisbe/swc2img_1k_v1"
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split_of = {}
    for sp in SPLITS:
        for fn in os.listdir(os.path.join(SPLIT_REF, sp)):
            if fn.endswith(".zarr"):
                split_of[fn[:-len(".zarr")]] = sp
    srcs = sorted(f for f in os.listdir(SRC) if f.endswith("_rgb.zarr"))
    counts = dict.fromkeys(SPLITS, 0)
    unmatched = []
    for f in srcs:
        brain = f[:-len("_rgb.zarr")]
        sp = split_of.get(brain)
        if sp is None:
            unmatched.append(brain)
            continue
        counts[sp] += 1
        if args.dry_run:
            continue
        src = os.path.join(SRC, f)
        dst = os.path.join(OUT, sp, brain + ".zarr")
        os.makedirs(os.path.join(dst, "volumes"), exist_ok=True)
        for meta in (".zgroup", ".zattrs"):
            for sub in ("", "volumes"):
                s = os.path.join(src, sub, meta)
                if os.path.isfile(s):
                    shutil.copy2(s, os.path.join(dst, sub, meta))
        for arr in ("raw", "gt_instances"):
            link = os.path.join(dst, "volumes", arr)
            target = os.path.join(src, "volumes", arr)
            if not os.path.isdir(target):
                sys.exit(f"missing {target}")
            if os.path.islink(link):
                if os.readlink(link) == target:
                    continue
                os.remove(link)
            os.symlink(target, link)
    if not args.dry_run:
        with open(os.path.join(OUT, "source_sqf.txt"), "w") as fh:
            fh.write("# SquashFS images holding volumes/raw + gt_instances (mount at /swc2img_1k_v1)\n")
            fh.write("\n".join(SQF_PARTS) + "\n")
    missing = sorted(set(split_of) - {f[:-len("_rgb.zarr")] for f in srcs})
    print(f"{len(srcs)} renders -> {counts}; not in {SPLIT_REF}: {unmatched or 'none'}; "
          f"{SPLIT_REF} brains without a render: {len(missing)} {missing}")
    if unmatched:
        sys.exit(1)


if __name__ == "__main__":
    main()
