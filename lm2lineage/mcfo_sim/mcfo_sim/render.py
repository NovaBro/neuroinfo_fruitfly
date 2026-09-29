"""Turn skeletons (with radii) into membrane-label point clouds and voxel masks.

MCFO reporters are myristoylated (membrane-targeted) smFPs, so the amount of
label per unit cable is proportional to membrane area (2*pi*r per um). Neurites
thinner than a voxel are splatted on their centreline with that weight; thicker
processes and the soma are sampled as hollow rings / spherical shells, so large
somata appear rim-bright as in real membrane-labelled data.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

TWO_PI = 2 * np.pi


def _perp_basis(d):
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    a = np.where(np.abs(d[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    u = np.cross(d, a)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    return u, np.cross(d, u)


def centerline_samples(xyz, radius, edges, step):
    """Resample every edge at ~`step` um. Returns points, radii, lengths, dirs."""
    a, b = xyz[edges[:, 0]], xyz[edges[:, 1]]
    ra, rb = radius[edges[:, 0]], radius[edges[:, 1]]
    seg = b - a
    L = np.linalg.norm(seg, axis=1)
    n = np.maximum(1, np.ceil(L / step).astype(int))
    idx = np.repeat(np.arange(len(edges)), n)
    k = np.arange(idx.size) - np.repeat(np.cumsum(n) - n, n)
    t = (k + 0.5) / n[idx]
    pts = a[idx] + seg[idx] * t[:, None]
    rad = ra[idx] + (rb[idx] - ra[idx]) * t
    ds = (L / n)[idx]
    return pts, rad, ds, seg[idx]


def membrane_points(pts, rad, ds, dirs, soma_xyz=None, soma_r=None,
                    thin_r=0.25, ring_spacing=0.2, rng=None):
    """Point cloud (positions, weights) proportional to membrane area."""
    rng = rng or np.random.default_rng()
    thin = rad < thin_r
    out_p = [pts[thin]]
    out_w = [TWO_PI * np.maximum(rad[thin], 0.05) * ds[thin]]
    if (~thin).any():
        c, r, l, d = pts[~thin], rad[~thin], ds[~thin], dirs[~thin]
        u, v = _perp_basis(d)
        m = np.clip(np.ceil(TWO_PI * r / ring_spacing).astype(int), 6, 128)
        i = np.repeat(np.arange(len(c)), m)
        k = np.arange(i.size) - np.repeat(np.cumsum(m) - m, m)
        th = TWO_PI * (k + rng.random(len(c))[i]) / m[i]
        out_p.append(c[i] + r[i, None] * (np.cos(th)[:, None] * u[i] + np.sin(th)[:, None] * v[i]))
        out_w.append((TWO_PI * r * l / m)[i])
    if soma_xyz is not None and soma_r and soma_r > thin_r:
        n = int(np.clip(np.ceil(4 * np.pi * soma_r ** 2 / ring_spacing ** 2), 50, 20000))
        g = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * g / n)
        th = np.pi * (1 + 5 ** 0.5) * g
        sph = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
        out_p.append(soma_xyz + soma_r * sph)
        out_w.append(np.full(n, 4 * np.pi * soma_r ** 2 / n))
    return np.concatenate(out_p), np.concatenate(out_w)


def bbox_vox(points_um, voxel, shape_zyx, pad=0):
    """Integer ZYX bbox (lo, hi) of points, clipped to the volume."""
    v = points_um[:, ::-1] / voxel
    lo = np.maximum(np.floor(v.min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(v.max(0)).astype(int) + pad + 1, shape_zyx)
    return lo, hi


def splat(points_um, weights, voxel, lo, hi):
    """Trilinear splat of weighted points into a local ZYX grid [lo, hi)."""
    shape = tuple(int(s) for s in (hi - lo))
    v = points_um[:, ::-1] / voxel - 0.5 - lo
    i0 = np.floor(v).astype(np.int64)
    f = v - i0
    size = int(np.prod(shape))
    sparse = 8 * len(points_um) < size // 20   # few points in a big volume
    out = np.zeros(size, np.float32 if sparse else np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ii = i0 + (dz, dy, dx)
                w = weights * (f[:, 0] if dz else 1 - f[:, 0]) \
                    * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2])
                ok = np.all((ii >= 0) & (ii < shape), axis=1)
                flat = np.ravel_multi_index(ii[ok].T, shape)
                if sparse:
                    np.add.at(out, flat, w[ok].astype(np.float32))
                else:
                    out += np.bincount(flat, w[ok], minlength=size)
    return out.reshape(shape).astype(np.float32)


def _ball(k):
    g = np.arange(-k, k + 1)
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
    return zz ** 2 + yy ** 2 + xx ** 2 <= k ** 2 + 0.5


def rasterize_mask(pts, rad, voxel, lo, hi, soma_xyz=None, soma_r=None, min_r_vox=0.5):
    """Filled-tube voxel mask of the neuron (local ZYX grid [lo, hi))."""
    shape = tuple(int(s) for s in (hi - lo))
    mask = np.zeros(shape, bool)
    P, R = pts, rad
    if soma_xyz is not None and soma_r:
        P = np.vstack([P, soma_xyz[None]])
        R = np.append(R, soma_r)
    iv = np.floor(P[:, ::-1] / voxel).astype(int) - lo
    ok = np.all((iv >= 0) & (iv < shape), axis=1)
    iv, R = iv[ok], R[ok]
    k = np.round(np.maximum(R / voxel, min_r_vox)).astype(int)
    for kk in np.unique(k):
        sel = iv[k == kk]
        if kk <= 0:
            mask[tuple(sel.T)] = True
            continue
        blo = np.maximum(sel.min(0) - kk, 0)
        bhi = np.minimum(sel.max(0) + kk + 1, shape)
        sub = np.zeros(tuple(bhi - blo), bool)
        sub[tuple((sel - blo).T)] = True
        sub = binary_dilation(sub, structure=_ball(int(kk)))
        view = mask[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]]
        view |= sub
    return mask
