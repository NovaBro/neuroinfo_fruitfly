"""Add new vh2308 synth_MCFO samples to an existing fisbe/synth_vh without touching old ones.

build_synth_vh.py reshuffles the whole set, which would move already-prepared samples between
splits. This script freezes every sample already under <out>/{train,val,test} and assigns only
new, complete source samples, picking splits so overall totals approach --ratios.
"""
import argparse
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_synth_vh import build_sample, is_complete  # noqa: E402

SPLITS = ("train", "val", "test")


def list_zarr_stems(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".zarr"))


def assign_new(new_stems, existing_counts, ratios, seed):
    """Deterministically assign new stems so overall split sizes approach ratios."""
    total = sum(existing_counts.values()) + len(new_stems)
    n_val = round(total * ratios[1])
    n_test = round(total * ratios[2])
    target = {"val": n_val, "test": n_test, "train": total - n_val - n_test}
    need = {s: max(0, target[s] - existing_counts[s]) for s in SPLITS}
    stems = sorted(new_stems)
    random.Random(seed).shuffle(stems)
    out = {s: [] for s in SPLITS}
    for split in ("val", "test"):
        take = min(need[split], len(stems))
        out[split] = stems[:take]
        stems = stems[take:]
    out["train"] = stems
    return {s: sorted(v) for s, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/scratch/vh2308/data/synth_MCFO/v1/samples")
    p.add_argument("--out", default="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/synth_vh")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy", action="store_true",
                   help="copy raw/gt_instances instead of symlinking")
    p.add_argument("--dry-run", action="store_true",
                   help="print planned additions and write nothing")
    args = p.parse_args()

    src_stems = set()
    for f in sorted(os.listdir(args.src)):
        if not (f.startswith("brain_") and f.endswith(".zarr")):
            continue
        if is_complete(os.path.join(args.src, f)):
            src_stems.add(f[:-5])
        else:
            print(f"skip (incomplete, retry next run): {f}", flush=True)

    existing = {s: list_zarr_stems(os.path.join(args.out, s)) for s in SPLITS}
    existing_all = {n for v in existing.values() for n in v}
    dupes = [n for n in existing_all if sum(n in v for v in existing.values()) > 1]
    if dupes:
        raise SystemExit(f"sample(s) in more than one split under {args.out}: {sorted(dupes)}")

    missing = sorted(existing_all - src_stems)
    if missing:
        print(f"warning: {len(missing)} existing sample(s) not complete in source "
              f"(left untouched): {missing[:5]}{' ...' if len(missing) > 5 else ''}", flush=True)

    new = sorted(src_stems - existing_all)
    print(f"source complete: {len(src_stems)}  existing: {len(existing_all)}  new: {len(new)}",
          flush=True)
    if not new:
        print("nothing to add", flush=True)
        return

    plan = assign_new(new, {s: len(existing[s]) for s in SPLITS}, args.ratios, args.seed)
    for s in SPLITS:
        print(f"{s}: {len(existing[s])} existing + {len(plan[s])} new = "
              f"{len(existing[s]) + len(plan[s])}", flush=True)
    if args.dry_run:
        print("dry run: nothing written", flush=True)
        return

    meta_dir = os.path.join(args.out, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    for split in SPLITS:
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        for stem in plan[split]:
            build_sample(os.path.join(args.src, stem + ".zarr"),
                         os.path.join(args.out, split, stem + ".zarr"),
                         copy=args.copy)
            for ext in (".json", "_skeletons.npz"):
                s = os.path.join(args.src, stem + ext)
                d = os.path.join(meta_dir, stem + ext)
                if os.path.exists(s) and not os.path.lexists(d):
                    os.symlink(s, d)

    with open(os.path.join(args.out, "sample_list_per_split.txt"), "w") as f:
        f.write("(1) samples for synth_vh:\n")
        for split in SPLITS:
            f.write(f"\n{split}:\n")
            f.writelines(f"{n}\n" for n in sorted(existing[split] + plan[split]))

    print(
        "\nNext (new samples lack PPP arrays; train_ppp.tar/val_ppp.tar are now stale):\n"
        "  OVERWRITE=0 DATASETS=synth_vh sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh\n"
        "  SPLITS=synth_vh/train,synth_vh/val,synth_vh/test sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh\n"
        "  DATASETS=synth_vh sbatch sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh",
        flush=True)


if __name__ == "__main__":
    main()
