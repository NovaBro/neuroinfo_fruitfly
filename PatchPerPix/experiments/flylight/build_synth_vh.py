"""Build fisbe/synth_vh/{train,val,test}/<sample>.zarr from vh2308's synth_MCFO samples.

The source is read-only for us, and prepare_fisbe_for_ppp.py writes new arrays into
<sample>.zarr/volumes/. So each sample is a real directory tree whose big arrays
(``raw``, ``gt_instances``) are symlinks into the source (or copies with --copy).
"""
import argparse
import os
import random
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

LINK_ARRAYS = ("raw", "gt_instances")
META_NAMES = (".zgroup", ".zattrs")


def is_complete(src_zarr):
    return os.path.isfile(os.path.join(src_zarr, "volumes", ".zgroup")) and all(
        os.path.isfile(os.path.join(src_zarr, "volumes", a, ".zarray"))
        for a in LINK_ARRAYS)


def build_sample(src_zarr, dst_zarr, copy=False):
    os.makedirs(os.path.join(dst_zarr, "volumes"), exist_ok=True)
    for base in ("", "volumes"):
        for meta in META_NAMES:
            s = os.path.join(src_zarr, base, meta)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst_zarr, base, meta))
    for arr in LINK_ARRAYS:
        s = os.path.join(src_zarr, "volumes", arr)
        d = os.path.join(dst_zarr, "volumes", arr)
        if not os.path.isdir(s):
            raise FileNotFoundError(s)
        if os.path.lexists(d):
            continue
        if copy:
            shutil.copytree(s, d)
        else:
            os.symlink(s, d)


def split_names(names, ratios, seed):
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n = len(names)
    n_val = round(n * ratios[1])
    n_test = round(n * ratios[2])
    n_train = n - n_val - n_test
    return {
        "train": sorted(names[:n_train]),
        "val": sorted(names[n_train:n_train + n_val]),
        "test": sorted(names[n_train + n_val:]),
    }


def parse_sample_list(path):
    """Parse v1_u8-style sample_list_per_split.txt (train:/val:/test: + stems)."""
    splits = {s: [] for s in ("train", "val", "test")}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("("):
                continue
            if line.endswith(":") and line[:-1] in splits:
                cur = line[:-1]
                continue
            if cur is None:
                continue
            stem = line.replace(".zarr", "")
            if stem.startswith("brain_"):
                splits[cur].append(stem)
    for s in splits:
        splits[s] = sorted(dict.fromkeys(splits[s]))
    if not any(splits.values()):
        raise SystemExit(f"no brain_* stems in {path}")
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/scratch/vh2308/data/synth_MCFO/v1/samples")
    p.add_argument("--out", default="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/synth_vh")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-list", default=None,
                   help="official sample_list_per_split.txt; skip reshuffle when set")
    p.add_argument("--copy", action="store_true",
                   help="copy raw/gt_instances instead of symlinking")
    args = p.parse_args()

    if args.sample_list:
        splits = parse_sample_list(args.sample_list)
        for split, names in splits.items():
            kept = []
            for stem in names:
                src_zarr = os.path.join(args.src, stem + ".zarr")
                if is_complete(src_zarr):
                    kept.append(stem)
                else:
                    print(f"skip (incomplete): {stem}.zarr", flush=True)
            splits[split] = kept
        if not any(splits.values()):
            raise SystemExit(f"no complete brain_*.zarr under {args.src}")
    else:
        stems = []
        for f in os.listdir(args.src):
            if not (f.startswith("brain_") and f.endswith(".zarr")):
                continue
            if is_complete(os.path.join(args.src, f)):
                stems.append(f[:-5])
            else:
                print(f"skip (incomplete): {f}", flush=True)
        if not stems:
            raise SystemExit(f"no brain_*.zarr under {args.src}")
        splits = split_names(stems, args.ratios, args.seed)

    out_label = os.path.basename(os.path.normpath(args.out))
    meta_dir = os.path.join(args.out, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    for split, names in splits.items():
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        for stem in names:
            build_sample(os.path.join(args.src, stem + ".zarr"),
                         os.path.join(args.out, split, stem + ".zarr"),
                         copy=args.copy)
            for ext in (".json", "_skeletons.npz"):
                s = os.path.join(args.src, stem + ext)
                d = os.path.join(meta_dir, stem + ext)
                if os.path.exists(s) and not os.path.lexists(d):
                    os.symlink(s, d)
        print(f"{split}: {len(names)}", flush=True)

    with open(os.path.join(args.out, "sample_list_per_split.txt"), "w") as f:
        f.write(f"(1) samples for {out_label}:\n")
        for split in ("train", "val", "test"):
            f.write(f"\n{split}:\n")
            f.writelines(f"{n}\n" for n in splits[split])


if __name__ == "__main__":
    main()
