"""EM-guided mask proposals: use a matched, template-aligned EM skeleton to
pull the corresponding neuron out of an aligned MCFO stack.

Steps for one (EM neuron, MCFO sample) pair:
1. put the EM skeleton (JRC2018 VNC unisex, um) on the stack's voxel grid,
   trying both the stated and the mirrored orientation;
2. read the MCFO colour of the neuron from bright voxels hugging the skeleton;
3. keep voxels within `tube_um` of the skeleton that are bright in that colour
   (signal z-score and colour cosine similarity), as connected components that
   touch the skeleton;
4. score: coverage (skeleton length with mask nearby), precision proxy (mask
   voxels close to skeleton) and presence (skeleton length with any bright
   signal nearby, a check on the match/registration itself).
The result is a *proposal* for a human to accept or reject, not ground truth.
Large neurons are processed on an x/y-binned grid to bound memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Params:
    tube_um: float = 6.0          # registration slack around the EM skeleton
    near_um: float = 3.0          # "on the skeleton" distance for seeds & scores
    colour_um: float = 2.5        # sampling radius for the neuron's colour
    z_thresh: float = 5.0         # signal threshold in robust noise units
    cos_min: float = 0.85         # colour similarity to the neuron's signature
    min_component_vox: int = 20
    step_um: float = 0.5          # skeleton resampling step
    max_crop_vox: int = 30_000_000  # above this, bin x/y to save memory


@dataclass
class Proposal:
    mask: np.ndarray              # bool, full-res crop grid (Z, Y, X)
    offset: np.ndarray            # crop origin in stack voxels (z, y, x)
    signature: np.ndarray         # normalised colour over signal channels
    channels_on: list
    mirrored: bool
    coverage: float
    precision: float
    presence: float
    bin_xy: int
    skel_vox: np.ndarray = field(repr=False)   # skeleton, (binned) crop voxel coords
    crop_rgb: np.ndarray = field(repr=False)   # uint8 display crop (3, Z, Y, X), binned grid
    mask_disp: np.ndarray = field(repr=False)  # mask on the display grid

    def summary(self):
        return dict(mirrored=self.mirrored, coverage=round(self.coverage, 3),
                    precision=round(self.precision, 3), presence=round(self.presence, 3),
                    channels_on=self.channels_on, signature=np.round(self.signature, 3).tolist(),
                    mask_voxels=int(self.mask.sum()), offset_zyx=self.offset.tolist(),
                    bin_xy=self.bin_xy)


def resample_swc(a, step):
    ids = a[:, 0].astype(int)
    lut = {n: i for i, n in enumerate(ids)}
    par = np.array([lut.get(int(p), -1) for p in a[:, 6]])
    xyz = a[:, 2:5]
    c = np.flatnonzero(par >= 0)
    p0, p1 = xyz[c], xyz[par[c]]
    L = np.linalg.norm(p1 - p0, axis=1)
    n = np.maximum(1, np.ceil(L / step).astype(int))
    i = np.repeat(np.arange(len(c)), n)
    t = (np.arange(i.size) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[i]
    return np.vstack([p0[i] + (p1[i] - p0[i]) * t[:, None], xyz])


def _robust_bg(v):
    s = v[::5, ::3, ::3].ravel()
    s = s[s > 0] if (s > 0).mean() > 0.2 else s
    bg = float(np.median(s))
    noise = 1.4826 * float(np.median(np.abs(s - bg))) + 1e-3
    return bg, max(noise, 1.0)


def _bin_xy(v, f):
    if f == 1:
        return v.astype(np.float32)
    Z, Y, X = v.shape
    Y2, X2 = Y // f * f, X // f * f
    return v[:, :Y2, :X2].reshape(Z, Y2 // f, f, X2 // f, f).mean((2, 4), dtype=np.float32)


def _dist_le(binary, sampling, thr):
    """Boolean 'within thr um of any True voxel' using an EDT on a coarse grid."""
    f = (1, 2, 2)
    Z, Y, X = binary.shape
    pad = [(0, (-s) % k) for s, k in zip(binary.shape, f)]
    b = np.pad(binary, pad)
    c = b.reshape(b.shape[0], b.shape[1] // 2, 2, b.shape[2] // 2, 2).any((2, 4))
    d = ndi.distance_transform_edt(~c, sampling=np.array(sampling) * f).astype(np.float32)
    # half a coarse voxel of slack so the coarse grid never under-reaches
    ok = d <= thr + 0.5 * sampling[1] * 2
    return np.repeat(np.repeat(ok, 2, axis=1), 2, axis=2)[:, :Y, :X]


def propose(stack, kinds, vox_xyz, swc, mirrored_hint=False, p: Params = Params(),
            orientations=None):
    """Best proposal over orientations (default: stated, then mirrored)."""
    sig_idx = [i for i, k in enumerate(kinds) if k == "signal"]
    C, Z, Y, X = stack.shape
    width_um = X * vox_xyz[0]
    pts_um = resample_swc(swc, p.step_um)
    best = None
    for mir in (orientations or (mirrored_hint, not mirrored_hint)):
        q = pts_um.copy()
        if mir:
            q[:, 0] = width_um - q[:, 0]
        res = _propose_one(stack, sig_idx, np.array(vox_xyz)[::-1], q, mir, p)
        if res is not None and (best is None or res.coverage > best.coverage + 0.05):
            best = res
    return best


def _propose_one(stack, sig_idx, vox_zyx, pts_um, mirrored, p):
    C, Z, Y, X = stack.shape
    zyx = pts_um[:, ::-1] / vox_zyx
    inside = np.all((zyx >= 0) & (zyx < [Z, Y, X]), axis=1)
    if inside.sum() < 10:
        return None
    zyx = zyx[inside]
    margin = np.ceil((p.tube_um + 3) / vox_zyx).astype(int)
    lo = np.maximum(np.floor(zyx.min(0)).astype(int) - margin, 0)
    hi = np.minimum(np.ceil(zyx.max(0)).astype(int) + margin + 1, [Z, Y, X])
    full_shape = tuple(hi - lo)
    f = 1
    while np.prod(full_shape) / f ** 2 > p.max_crop_vox:
        f += 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    vz = vox_zyx * [1, f, f]

    zs = []
    for c in sig_idx:
        v = ndi.gaussian_filter(_bin_xy(stack[c][sl], f), (0.5, 1.0 / f, 1.0 / f))
        bg, noise = _robust_bg(v)
        v -= bg
        v /= noise
        zs.append(v)
    zs = np.stack(zs)
    shape = zs.shape[1:]

    iv = np.clip(np.floor((zyx - lo) / [1, f, f]).astype(int), 0, np.array(shape) - 1)
    sk = np.zeros(shape, bool)
    sk[tuple(iv.T)] = True
    tube = _dist_le(sk, vz, p.tube_um)
    near = _dist_le(sk, vz, p.near_um)
    col_zone = _dist_le(sk, vz, p.colour_um)

    zmax = zs.max(0)
    bright = zmax > p.z_thresh
    near_col = bright & col_zone
    disp = _display(zs)
    if near_col.sum() < 10:
        return Proposal(np.zeros(full_shape, bool), lo, np.zeros(len(sig_idx)), [], mirrored,
                        0.0, 0.0, 0.0, f, iv, disp, np.zeros(shape, bool))
    vecs = np.clip(zs[:, near_col], 0, None)
    w = vecs.sum(0)
    unit = vecs / (np.linalg.norm(vecs, axis=0) + 1e-6)
    sig = np.array([_wmedian(unit[i], w) for i in range(len(sig_idx))])
    sig = sig / (np.linalg.norm(sig) + 1e-6)
    on = [int(i) for i in np.flatnonzero(sig >= 0.35 * sig.max())]

    np.clip(zs, 0, None, out=zs)
    norm = np.sqrt((zs ** 2).sum(0)) + 1e-6
    cos = np.tensordot(sig.astype(np.float32), zs, axes=1) / norm
    del norm
    strength = zs[on].max(0)
    cand = tube & (strength > p.z_thresh) & (cos >= p.cos_min)
    del cos, strength
    lab, n = ndi.label(cand, structure=np.ones((3, 3, 3)))
    if n:
        touches = np.zeros(n + 1, bool)
        touches[np.unique(lab[cand & near])] = True
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = touches & (sizes >= max(1, p.min_component_vox // f ** 2))
        keep[0] = False
        mask = keep[lab]
    else:
        mask = cand
    del lab

    if mask.any():
        coverage = float(_dist_le(mask, vz, p.near_um)[tuple(iv.T)].mean())
        precision = float(near[mask].mean())
    else:
        coverage = precision = 0.0
    presence = float(_dist_le(bright, vz, p.near_um)[tuple(iv.T)].mean())

    full = np.repeat(np.repeat(mask, f, axis=1), f, axis=2)
    full = np.pad(full, [(0, 0), (0, max(0, full_shape[1] - full.shape[1])),
                         (0, max(0, full_shape[2] - full.shape[2]))])[:, :full_shape[1], :full_shape[2]]
    return Proposal(full, lo, sig, on, mirrored, coverage, precision, presence, f, iv, disp, mask)


def _wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][min(np.searchsorted(cw, cw[-1] / 2), len(o) - 1)]


def _display(zs):
    out = np.empty(zs.shape, np.uint8)
    for i in range(zs.shape[0]):
        v = np.clip(zs[i], 0, None)
        hi = np.percentile(v[::2, ::2, ::2], 99.8) + 1e-6
        out[i] = (np.clip(v / hi, 0, 1) * 255).astype(np.uint8)
    return out
