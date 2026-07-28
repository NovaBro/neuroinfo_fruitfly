#!/usr/bin/env python3
"""Exit 0 if a prediction zarr is fully written, 1 otherwise.

run_ppp.check_file() only does zarr.open() + a key lookup, but
predict_no_gp.create_zarr_outputs() writes .zarray before any block is
predicted. An interrupted volume therefore still looks "already exists" to
run_ppp and is silently skipped, leaving zeros in the unwritten region. This
counts materialised chunks instead.

Tile size does not matter: the UNet is valid-padded, so every output voxel sees
the same context no matter how the volume was tiled. A complete prediction is
valid even if it was written with a different test_input_shape_valid.

    python3 check_prediction_complete.py <sample>.zarr [--aff-key volumes/pred_affs]
"""
import argparse
import json
import math
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("zarr")
p.add_argument("--aff-key", default="volumes/pred_affs")
p.add_argument("-q", "--quiet", action="store_true")
args = p.parse_args()


def report(msg):
    if not args.quiet:
        print(f"check_prediction_complete: {args.zarr}: {msg}")


arr = os.path.join(args.zarr, args.aff_key)
zarray = os.path.join(arr, ".zarray")
if not os.path.exists(zarray):
    report("missing .zarray -> incomplete")
    sys.exit(1)

meta = json.load(open(zarray))
expected = 1
for size, chunk in zip(meta["shape"], meta["chunks"]):
    expected *= math.ceil(size / chunk)
present = sum(1 for _, _, files in os.walk(arr)
              for f in files if not f.startswith("."))

if present < expected:
    report(f"{present}/{expected} chunks ({100.0 * present / expected:.1f}%) -> incomplete")
    sys.exit(1)

report(f"{present}/{expected} chunks -> complete")
sys.exit(0)
