"""Check the GPU confetti used in training against the generator's numpy confetti.

  python -m synthetic.simple_tubes.check_gpu_confetti [--crop 140] [--repeats 12]
      [--sample /simple_v1/train/tube_00000.zarr --png out.png]

1. Parity: on constant crops (all channels at level 5), apply numpy `noise.confetti`
   and torch `GpuConfetti` with the same fixed densities, and compare the fraction of
   voxels raised by bright confetti, the fraction clamped by dark confetti, and the
   histogram of bright levels. Prints relative differences (expect a few %).
2. Speed: GpuConfetti on a (18, 3, crop, crop, crop) fp16 batch on the GPU at typical
   and maximum density; median ms per batch.
3. Optional picture: a crop of a stored train sample, as stored and with GPU confetti.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import replace

import numpy as np
import torch

from . import noise
from .generate import REPO_ROOT


def load_gpu_confetti():
    path = (REPO_ROOT / "PatchPerPix" / "experiments" / "flylight" / "setups" / "setup01"
            / "gpu_confetti.py")
    spec = importlib.util.spec_from_file_location("_gpu_confetti", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stats(levels, base):
    """levels: (3, Z, Y, X) integer levels after confetti on a constant `base` crop."""
    raised = (levels > base).any(0)
    dark = (levels[0] <= 1)
    hist = np.bincount(levels.max(0)[raised].ravel(), minlength=16)[8:16]
    return raised.mean(), dark.mean(), hist / max(hist.sum(), 1)


def parity(mod, crop, repeats, density, device):
    base = 5
    P = replace(noise.NoiseParams(), bright_density=(density, density),
                dark_density=(density, density))
    np_stats, gpu_stats = [], []
    for r in range(repeats):
        raw = np.full((3, crop, crop, crop), base, dtype=np.uint8)
        noise.confetti(raw, np.random.default_rng(1000 + r), P)
        np_stats.append(_stats(raw, base))
    gc = mod.GpuConfetti(bright_density=(density, density), dark_density=(density, density),
                         seed=7)
    for r in range(repeats):
        t = torch.full((1, 3, crop, crop, crop), base / 15.0, device=device)
        gc(t, step=r)
        lv = torch.round(t[0] * 15).to(torch.uint8).cpu().numpy()
        gpu_stats.append(_stats(lv, base))
    for name, k in (("bright voxel fraction", 0), ("dark voxel fraction", 1)):
        a = np.mean([s[k] for s in np_stats])
        b = np.mean([s[k] for s in gpu_stats])
        print(f"  {name:22s} numpy {a:.5f}  gpu {b:.5f}  rel diff {abs(b - a) / a:+.1%}")
    ha = np.mean([s[2] for s in np_stats], axis=0)
    hb = np.mean([s[2] for s in gpu_stats], axis=0)
    print("  bright level histogram 8..15 numpy", np.round(ha, 3))
    print("  bright level histogram 8..15 gpu  ", np.round(hb, 3))


def speed(mod, crop, device, **params):
    gc = mod.GpuConfetti(**params)
    raw = torch.rand(18, 3, crop, crop, crop, device=device).half()
    for i in range(3):
        gc(raw, step=i)
    torch.cuda.synchronize()
    ts = []
    for i in range(10):
        t0 = time.perf_counter()
        gc(raw, step=100 + i)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def picture(mod, sample, crop, png, device):
    import zarr
    from PIL import Image

    raw = zarr.open(sample, "r")["volumes/raw"]
    z0, y0, x0 = (np.asarray(raw.shape[1:]) - crop) // 2
    c = raw[:, z0:z0 + crop, y0:y0 + crop, x0:x0 + crop].astype(np.float32) / 15.0
    t = torch.as_tensor(c[None], device=device)
    mod.GpuConfetti(seed=3)(t, step=0)
    after = t[0].cpu().numpy()
    tiles = [(np.clip(a.max(axis=1), 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
             for a in (c, after)]
    sep = np.full((crop, 4, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([tiles[0], sep, tiles[1]], axis=1)).save(png)
    print(f"wrote {png} (stored crop | with GPU confetti)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crop", type=int, default=140)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--png", default="gpu_confetti_crop.png")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = load_gpu_confetti()
    print(f"device {device}, crop {args.crop}^3")
    for d in (1e-4, 2e-3):
        print(f"parity at {d:g} spots/voxel (bright and dark):")
        parity(mod, args.crop, args.repeats, d, device)
    if device == "cuda":
        print(f"speed, batch 18 fp16: typical densities {speed(mod, args.crop, device):.1f} ms; "
              f"max density {speed(mod, args.crop, device, bright_density=(2e-3, 2e-3), dark_density=(2e-3, 2e-3)):.1f} ms")
    if args.sample:
        picture(mod, args.sample, args.crop, args.png, device)


if __name__ == "__main__":
    main()
