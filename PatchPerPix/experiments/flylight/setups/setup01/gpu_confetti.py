"""Bright and dark confetti added to training batches on the GPU (simple_v1).

simple_v1 train samples are stored with neurons, haze and distractor blobs but without
confetti; val/test samples have it baked in. This module adds the confetti to the raw
tensor of every training batch inside the training step (TorchTrainExt hook in
train.py), so each step sees fresh confetti, and no gunpowder node is added
(validation crops are unchanged). Torch port of the confetti part of
synthetic/simple_tubes/noise.py, with the same parameters:

  * density field: 2-octave Perlin noise at 1/`coarse` resolution, normalised to [0, 1]
    by its 1st/99th percentiles, raised to gamma ~ U(density_gamma)
  * per crop and kind, density ~ log-uniform(bright_density / dark_density) spots per
    voxel, count ~ Poisson(density * voxels), centres drawn proportional to the field
  * spot = voxels within radius ~ U(spot_radius) of the centre (1-3 voxels across)
  * bright: level ~ U{bright_level}, one of the 7 MCFO colours, per-channel max
  * dark: level ~ U{dark_level}, all channels, min, applied last

Bright is a max and dark is applied last in the generator too, so adding both to a baked
sample gives exactly what full pre-rendering would. Values are level / `levels` (raw
clipmax), mapped to [-1, 1] when shift_intensity is on. Densities are drawn per crop,
not per volume (more variety than the baked val/test samples).

Everything is batched over the crops (one density-field computation, one multinomial and
one scatter_reduce_ per kind) and written in place in the batch dtype (fp16 with
raw_transport_fp16), so the cost does not grow with the batch size in Python.
synthetic/simple_tubes/check_gpu_confetti.py checks parity with the numpy version and speed.
"""
from __future__ import annotations

import itertools
import math

import torch

DEFAULTS = {
    "levels": 15,                      # raw clipmax (4-bit raw): value = level / levels
    "coarse": 4,
    "density_period": (60.0, 200.0),
    "density_gamma": (1.0, 3.0),
    "bright_density": (1e-5, 2e-3),
    "dark_density": (1e-5, 2e-3),
    "spot_radius": (0.5, 1.5),
    "bright_level": (8, 15),
    "dark_level": (0, 1),
    "seed": 0,
}

COLOURS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
SPOT_CHUNK = 32768                     # spots per scatter, bounds the (spots x offsets) temporaries


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin3d_batch(shape, periods, gen):
    """Gradient (Perlin) noise for B crops at once; periods: (B,) lattice spacings in grid
    cells. Returns (B, *shape) float32, about [-1, 1]."""
    B, dev = periods.shape[0], periods.device
    pmin = float(periods.min())
    n_lat = [int(math.ceil(n / pmin)) + 2 for n in shape]
    g = torch.randn(B, *n_lat, 3, generator=gen, device=dev)
    g = g / (g.norm(dim=-1, keepdim=True) + 1e-12)
    off = torch.rand(B, 3, generator=gen, device=dev)
    t = [torch.arange(n, device=dev, dtype=torch.float32)[None, :] / periods[:, None]
         + off[:, k:k + 1] for k, n in enumerate(shape)]          # (B, n) per axis
    i = [torch.floor(a).long() for a in t]
    f = [a - b.float() for a, b in zip(t, i)]
    w = [_fade(a) for a in f]
    bi = torch.arange(B, device=dev)[:, None, None, None]
    iz, iy, ix = i[0][:, :, None, None], i[1][:, None, :, None], i[2][:, None, None, :]
    fz, fy, fx = f[0][:, :, None, None], f[1][:, None, :, None], f[2][:, None, None, :]
    wz, wy, wx = w[0][:, :, None, None], w[1][:, None, :, None], w[2][:, None, None, :]
    out = torch.zeros(B, *shape, device=dev)
    for dz, dy, dx in itertools.product((0, 1), repeat=3):
        G = g[bi, iz + dz, iy + dy, ix + dx]
        dot = G[..., 0] * (fz - dz) + G[..., 1] * (fy - dy) + G[..., 2] * (fx - dx)
        out += ((wz if dz else 1 - wz) * (wy if dy else 1 - wy) * (wx if dx else 1 - wx)) * dot
    return out * 1.7


def fbm01_batch(shape, periods, octaves, gen):
    """Fractal Perlin per crop, each normalised to [0, 1] by its own 1st/99th percentiles."""
    total = torch.zeros(periods.shape[0], *shape, device=periods.device)
    amp = 1.0
    for o in range(octaves):
        total += amp * perlin3d_batch(shape, torch.clamp(periods / 2 ** o, min=1.5), gen)
        amp *= 0.5
    flat = total.view(total.shape[0], -1)
    q = torch.quantile(flat, torch.tensor([0.01, 0.99], device=flat.device), dim=1)  # (2, B)
    lo, hi = q[0][:, None], q[1][:, None]
    return ((flat - lo) / torch.clamp(hi - lo, min=1e-6)).clamp(0, 1)          # (B, cells)


def _ball_offsets(reach, device):
    R = int(math.ceil(reach))
    g = torch.arange(-R, R + 1, device=device)
    off = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)
    return off[off.float().norm(dim=1) <= reach]


class GpuConfetti:
    """Callable: raw (B, C, Z, Y, X) on the GPU; adds confetti in place and returns it."""

    def __init__(self, shift_intensity=False, **params):
        unknown = set(params) - set(DEFAULTS) - {"datasets"}
        if unknown:
            raise ValueError(f"unknown gpu_confetti keys {sorted(unknown)}")
        self.p = {**DEFAULTS, **{k: v for k, v in params.items() if k != "datasets"}}
        self.shift = bool(shift_intensity)
        self._cpu = torch.Generator()   # scalars (densities, counts): no GPU sync per draw
        self._gen = None
        self._dev = None

    def _u(self, rng, n):
        lo, hi = rng
        return lo + (hi - lo) * torch.rand(n, generator=self._cpu)

    def _value(self, level):
        v = level / float(self.p["levels"])
        return v * 2 - 1 if self.shift else v

    @torch.no_grad()
    def __call__(self, raw: torch.Tensor, step: int = 0) -> torch.Tensor:
        dev = raw.device
        if self._gen is None or self._dev != dev:
            self._dev = dev
            self._gen = torch.Generator(device=dev)
            # voxels within r <= spot_radius[1] of a centre lie within r + 0.87 of its
            # nearest voxel (57 offsets for r = 1.5)
            self._off = _ball_offsets(float(self.p["spot_radius"][1]) + 0.87, dev).int()
            self._colours = torch.tensor(COLOURS, dtype=torch.bool, device=dev)
        seed = int(self.p["seed"]) * 1_000_003 + int(step)
        self._gen.manual_seed(seed)
        self._cpu.manual_seed(seed)
        if not raw.is_contiguous():
            raise ValueError("gpu_confetti needs a contiguous raw tensor")

        p, gen = self.p, self._gen
        B, C = raw.shape[:2]
        shape = tuple(raw.shape[2:])
        c = int(p["coarse"])
        cs = tuple(int(math.ceil(n / c)) for n in shape)
        periods = (self._u(p["density_period"], B) / c).to(dev)
        gammas = self._u(p["density_gamma"], B).to(dev)
        prob = fbm01_batch(cs, periods, 2, gen).pow(gammas[:, None])
        prob = prob / prob.sum(dim=1, keepdim=True)
        nvox = shape[0] * shape[1] * shape[2]
        for kind in ("bright", "dark"):
            lo, hi = p[f"{kind}_density"]
            dens = torch.exp(self._u((math.log(lo), math.log(hi)), B))
            counts = torch.poisson(dens * nvox, generator=self._cpu).long()
            nmax = int(counts.max())
            if nmax == 0:
                continue
            cells = torch.multinomial(prob, nmax, replacement=True, generator=gen)  # (B, nmax)
            valid = (torch.arange(nmax, device=dev)[None, :] < counts.to(dev)[:, None])
            crop = torch.arange(B, device=dev)[:, None].expand(B, nmax)[valid]
            cells = cells[valid]
            for s in range(0, len(cells), SPOT_CHUNK):
                self._stamp(raw, kind, crop[s:s + SPOT_CHUNK], cells[s:s + SPOT_CHUNK],
                            cs, shape, C)
        return raw

    def _stamp(self, raw, kind, crop, cells, cs, shape, C):
        """Write spots centred in `cells` of crops `crop` into raw (flat scatter_reduce_)."""
        p, gen, dev = self.p, self._gen, self._dev
        n = len(cells)
        c = int(p["coarse"])
        cz = cells // (cs[1] * cs[2])
        cy = (cells // cs[2]) % cs[1]
        cx = cells % cs[2]
        lim = torch.tensor(shape, device=dev, dtype=torch.float32)
        pos = (torch.stack([cz, cy, cx], 1).float()
               + torch.rand(n, 3, generator=gen, device=dev)) * c - 0.5
        pos = torch.minimum(pos.clamp(min=-0.49), lim - 0.51)
        lo, hi = p["spot_radius"]
        r = lo + (hi - lo) * torch.rand(n, generator=gen, device=dev)
        v = torch.round(pos).int()[:, None, :] + self._off[None]            # (n, K, 3)
        keep = (v.float() - pos[:, None, :]).norm(dim=-1) <= r[:, None]
        keep &= ((v >= 0) & (v < lim.int())).all(dim=-1)
        V = shape[0] * shape[1] * shape[2]
        vflat = ((v[..., 0].long() * shape[1] + v[..., 1]) * shape[2] + v[..., 2])  # (n, K)
        base = crop.long()[:, None] * C * V + vflat                         # channel 0 index
        flat = raw.view(-1)
        if kind == "bright":
            lv = torch.randint(p["bright_level"][0], p["bright_level"][1] + 1, (n,),
                               generator=gen, device=dev).float()
            col = self._colours[torch.randint(len(COLOURS), (n,), generator=gen, device=dev)]
            vals = self._value(lv).to(raw.dtype)[:, None].expand_as(keep)
            idx, val = [], []
            for ch in range(min(C, 3)):
                m = keep & col[:, ch:ch + 1]
                idx.append((base + ch * V)[m])
                val.append(vals[m])
            flat.scatter_reduce_(0, torch.cat(idx), torch.cat(val), reduce="amax")
        else:
            lv = torch.randint(p["dark_level"][0], p["dark_level"][1] + 1, (n,),
                               generator=gen, device=dev).float()
            vals = self._value(lv).to(raw.dtype)[:, None].expand_as(keep)[keep]
            sel = base[keep]
            idx = torch.cat([sel + ch * V for ch in range(C)])
            flat.scatter_reduce_(0, idx, vals.repeat(C), reduce="amin")
