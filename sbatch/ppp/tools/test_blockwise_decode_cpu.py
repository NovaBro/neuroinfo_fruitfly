#!/usr/bin/env python3
"""CPU unit test for blockwise decode gather/scatter/resume (mock decoder)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import zarr

# Import decode module from setups path without full package init.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SETUP = os.path.join(
    ROOT, "PatchPerPix/experiments/flylight/setups/setup01")
sys.path.insert(0, os.path.join(ROOT, "PatchPerPix/experiments"))
sys.path.insert(0, SETUP)

# Stub relative import dependency before loading decode.
import types
torch_model = types.ModuleType("torch_model")
sys.modules["torch_model"] = torch_model
sys.modules["PatchPerPix.visualize"] = types.ModuleType("PatchPerPix.visualize")
sys.modules["PatchPerPix.visualize"].visualize_patches = lambda *a, **k: None

import importlib.util
spec = importlib.util.spec_from_file_location(
    "decode_mod", os.path.join(SETUP, "decode.py"))
decode_mod = importlib.util.module_from_spec(spec)
# Provide package-like attrs for relative imports used at top of decode.py
decode_mod.__package__ = ""
# Re-exec with patched imports: load source and exec after binding stubs
src = open(os.path.join(SETUP, "decode.py")).read()
src = src.replace("from . import torch_model\n", "")
src = src.replace(
    "from PatchPerPix.visualize import visualize_patches\n",
    "visualize_patches = None\n")
ns = {
    "__name__": "decode_mod",
    "__file__": os.path.join(SETUP, "decode.py"),
}
exec(compile(src, os.path.join(SETUP, "decode.py"), "exec"), ns)


class MockModel:
    def __init__(self, patch_vol, code_units):
        self.patch_vol = patch_vol
        self.code_units = code_units

    def decoder(self, codes):
        # Deterministic: tile mean of code into patch_vol channels.
        m = codes.mean(dim=-1, keepdim=True)
        return m.expand(-1, self.patch_vol).contiguous()


def main():
    rng = np.random.default_rng(0)
    shape = (40, 70, 70)  # not divisible by 64 → edge blocks
    code_units = 8
    patch_vol = 27
    block = (16, 16, 16)

    tmp = tempfile.mkdtemp(prefix="blockwise_decode_")
    try:
        zpath = os.path.join(tmp, "sample.zarr")
        zf = zarr.open(zpath, "w")
        code = rng.standard_normal((code_units,) + shape).astype(np.float16)
        # Multi-channel numinst: FG where ch0 < 0.1
        numinst = rng.random((3,) + shape).astype(np.float16)
        numinst[0] = 0.5
        numinst[0, 5:30, 10:50, 10:50] = 0.01  # FG box
        zc = zf.create(
            "volumes/pred_code", shape=code.shape,
            chunks=(code_units, 8, 8, 8), dtype=np.float16)
        zc[:] = code
        zn = zf.create(
            "volumes/pred_numinst", shape=numinst.shape,
            chunks=(3, 8, 8, 8), dtype=np.float16)
        zn[:] = numinst
        aff = zf.create(
            "volumes/pred_affs",
            shape=(patch_vol,) + shape,
            chunks=(patch_vol,) + block,
            dtype=np.float16,
        )

        config = {
            "decode_batch_size": 64,
            "code_units": code_units,
            "patchshape": [3, 3, 3],
            "output_format": "zarr",
            "code_key": "volumes/pred_code",
            "numinst_key": "volumes/pred_numinst",
            "fg_thresh": 0.5,
        }
        model = MockModel(patch_vol, code_units)
        device = torch.device("cpu")

        decode_sample = ns["decode_sample"]
        decode_sample(config, model, zpath, device, aff_out=aff)

        # Completeness: all chunks present
        arr = os.path.join(zpath, "volumes/pred_affs")
        meta = json_load = __import__("json").load(open(os.path.join(arr, ".zarray")))
        import math
        expected = 1
        for size, ch in zip(meta["shape"], meta["chunks"]):
            expected *= math.ceil(size / ch)
        present = sum(
            1 for _, _, files in os.walk(arr)
            for f in files if not f.startswith("."))
        assert present == expected, f"{present}/{expected}"

        # Reference: FG voxels decode to tile(mean(code))
        fg = numinst[0] < 0.1
        mz, my, mx = np.nonzero(fg)
        codes = code[:, mz, my, mx].astype(np.float32).T
        ref = np.repeat(codes.mean(axis=-1, keepdims=True), patch_vol, axis=1)
        aff_np = np.asarray(aff[:], dtype=np.float32)
        got = aff_np[:, mz, my, mx].T
        max_abs = float(np.abs(got - ref).max())
        print(f"FG voxels={len(mz)} max|got-ref|={max_abs}")
        assert max_abs < 1e-3

        # Resume: delete nothing; re-run should skip all blocks
        n_before = present
        decode_sample(config, model, zpath, device, aff_out=aff)
        present2 = sum(
            1 for _, _, files in os.walk(arr)
            for f in files if not f.startswith("."))
        assert present2 == n_before

        # Resume after deleting one chunk file
        chunk_file = os.path.join(arr, "0.0.0.0")
        assert os.path.exists(chunk_file)
        os.remove(chunk_file)
        decode_sample(config, model, zpath, device, aff_out=aff)
        assert os.path.exists(chunk_file)
        present3 = sum(
            1 for _, _, files in os.walk(arr)
            for f in files if not f.startswith("."))
        assert present3 == expected

        # HDF / in-memory path
        out = decode_sample(config, model, zpath, device, aff_out=None)
        assert out.shape == (patch_vol,) + shape
        got2 = out[(slice(None), mz, my, mx)].T
        assert float(np.abs(got2 - ref).max()) < 1e-3

        print("CPU_UNIT_TEST_OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
