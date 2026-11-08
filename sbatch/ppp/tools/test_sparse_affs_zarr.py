#!/usr/bin/env python3
"""CPU check of the sparse pred_affs storage used by predict_no_gp.py.

Creates outputs with predict_no_gp.create_zarr_outputs (sparse and dense), writes
tiles the way predict does (one all-background tile skipped, one tile with a few
foreground voxels masked), and checks:
  * missing chunks read back as 0 and kept values round-trip exactly;
  * all-zero chunks inside a written tile are not stored;
  * check_prediction_complete.py: sparse without marker -> incomplete,
    with marker -> complete; dense legacy (no marker) still uses chunk counting.

  python sbatch/ppp/tools/test_sparse_affs_zarr.py   (inside the ppp container)
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PRED = os.path.join(REPO, "PatchPerPix", "experiments", "flylight", "setups", "setup01", "predict_no_gp.py")
CHECK = os.path.join(REPO, "sbatch", "ppp", "check_prediction_complete.py")


def load_create():
    # predict_no_gp.py uses a relative import (from . import torch_model); only
    # create_zarr_outputs is needed, so exec that function's source alone.
    src = open(PRED).read()
    start = src.index("def create_zarr_outputs(")
    ns = {}
    exec("import os, logging, numpy as np, zarr\nfrom numcodecs import Blosc\n"
         "logger = logging.getLogger('t')\n" + src[start:], ns)
    return ns["create_zarr_outputs"]


def complete(path):
    return subprocess.run([sys.executable, CHECK, "-q", path]).returncode == 0


def n_chunk_files(arr_dir):
    return sum(1 for _, _, fs in os.walk(arr_dir) for f in fs if not f.startswith("."))


def main():
    create = load_create()
    shape, chunk, tile, ch = (64, 128, 64), [16, 16, 16], 32, 343
    keys = {"volumes/pred_affs": ch, "volumes/pred_numinst": 3}
    rng = np.random.default_rng(0)
    ok = True
    with tempfile.TemporaryDirectory() as d:
        cfg = {"output_folder": d, "voxel_size": [1, 1, 1]}
        zf = create(cfg, "s_sparse", shape, chunk, keys, sparse_keys=["volumes/pred_affs"])
        zf["volumes/pred_affs"].attrs["sparse_fg_thresh"] = 0.05
        affs = zf["volumes/pred_affs"]
        # tile A (z 0:32, y 0:32, x 0:32): all background -> skipped (not written)
        # tile B (z 0:32, y 32:64, x 0:32): 3 foreground voxels, rest masked to 0
        t = rng.random((ch, tile, tile, tile)).astype(np.float16)
        keep = np.zeros((tile,) * 3, bool)
        keep[1, 2, 3] = keep[20, 5, 30] = keep[31, 31, 31] = True
        affs[:, 0:32, 32:64, 0:32] = t * keep
        path = os.path.join(d, "s_sparse.zarr")
        arr_dir = os.path.join(path, "volumes", "pred_affs")
        stored = n_chunk_files(arr_dir)
        # tile B covers 8 chunks of 16^3; the 3 fg voxels fall in 3 distinct chunks
        print(f"sparse chunk files stored: {stored} (expect 3 of 8 in tile B, 0 elsewhere)")
        ok &= stored == 3
        back = affs[:, 0:32, 32:64, 0:32]
        ok &= np.array_equal(back, t * keep)
        ok &= not np.any(affs[:, 0:32, 0:32, 0:32])  # skipped tile reads as 0
        ok &= not np.any(affs[:, 32:64, :, :])      # never-touched region reads as 0
        r1 = complete(path)
        affs.attrs["predict_complete"] = True
        r2 = complete(path)
        print(f"check_prediction_complete sparse: no marker -> {r1} (expect False), marker -> {r2} (expect True)")
        ok &= (r1 is False) and (r2 is True)

        # dense legacy: no marker, full chunk count -> complete; partial -> incomplete
        zd = create(cfg, "s_dense", shape, chunk, keys)
        pd = os.path.join(d, "s_dense.zarr")
        zd["volumes/pred_affs"][:, 0:32] = 1
        r3 = complete(pd)
        zd["volumes/pred_affs"][:, 32:64] = 1
        r4 = complete(pd)
        print(f"check_prediction_complete dense legacy: half -> {r3} (expect False), full -> {r4} (expect True)")
        ok &= (r3 is False) and (r4 is True)
    print("SPARSE AFFS ZARR OK" if ok else "SPARSE AFFS ZARR FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
