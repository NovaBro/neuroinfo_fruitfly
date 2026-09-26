"""Optics, tissue background and camera/detector noise.

Every parameter is drawn per sample from a range (domain randomisation), so a
segmentation network sees a spread of conditions rather than one guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import (gaussian_filter,
                           map_coordinates, zoom)

from .render import splat


@dataclass
class ImagingParams:
    voxel_um: float = 0.44                 # FlyLight 40x Gen1 MCFO: 0.44 um isotropic
    psf_sigma_xy_um: float = 0.30          # Gaussian approx. of confocal PSF + tissue
    psf_sigma_z_um: float = 0.90
    deep_blur_factor: float = 1.6          # PSF widening at the far side of the stack
    atten_length_um: float = 250.0         # exp(-depth / L) signal loss
    illum_sigma: float = 0.15              # smooth multiplicative illumination field
    neuron_photons: float = 120.0          # median photons at a neurite centreline
    haze_photons: tuple = (0.5, 0.5, 2.0)      # neuropil haze per channel
    cortex_photons: tuple = (0.3, 0.3, 1.0)    # cell-body rind haze per channel
    speckle_per_1e4um3: tuple = (0.5, 0.5, 4.0)  # tiny puncta density per channel
    speckle_photons: float = 60.0
    blob_per_1e4um3: tuple = (0.3, 0.05, 0.05)   # soma-sized non-specific blobs
    blob_photons: float = 25.0
    crosstalk: float = 0.03                # channel bleed-through fraction
    texture_sigma: float = 0.5             # neuropil haze texture (log-amplitude)
    texture_scale_um: float = 0.5
    adu_per_photon: float = 6.0            # fallback; normally set by auto-gain
    gain_target: float = 0.25              # p99.9 of neuron voxels / 12-bit range
    pmt_excess_shape: float = 4.0          # gamma multiplicative (PMT excess) noise
    read_noise_adu: float = 1.2            # measured ~1.3 ADU on a real Gen1 VNC tile
    offset_adu: float = 2.5                # real dark level ~2-3 ADU
    max_adu: int = 4095                    # 12-bit detector

    @classmethod
    def random(cls, rng, voxel_um=0.44):
        u = rng.uniform
        perm = rng.permutation(3)  # which channel gets the heavy background
        haze = np.array([4.0, 4.0, 10.0])[perm] * np.exp(u(np.log(0.02), np.log(1.5)))
        cortex = np.array([3.0, 3.0, 6.0])[perm] * np.exp(u(np.log(0.02), np.log(1.5)))
        speck = np.array([0.5, 0.5, 4.0])[perm] * u(0.3, 2.0)
        blob = np.array([0.3, 0.05, 0.05])[rng.permutation(3)] * u(0.3, 2.0)
        return cls(voxel_um=voxel_um,
                   psf_sigma_xy_um=u(0.22, 0.40), psf_sigma_z_um=u(0.6, 1.3),
                   deep_blur_factor=u(1.2, 2.0), atten_length_um=u(150, 500),
                   illum_sigma=u(0.05, 0.25),
                   neuron_photons=float(np.exp(u(np.log(15), np.log(200)))),
                   haze_photons=tuple(haze), cortex_photons=tuple(cortex),
                   speckle_per_1e4um3=tuple(speck), speckle_photons=u(30, 120),
                   blob_per_1e4um3=tuple(blob), blob_photons=u(10, 50),
                   crosstalk=u(0.0, 0.08), adu_per_photon=u(3, 10),
                   texture_sigma=u(0.2, 0.9), texture_scale_um=u(0.3, 0.8),
                   gain_target=u(0.12, 0.6), pmt_excess_shape=u(2, 10),
                   read_noise_adu=u(0.7, 3.0), offset_adu=u(0.0, 6.0))


def psf_blur(vol, z0, p: ImagingParams, total_z):
    """Depth-dependent Gaussian PSF on a local ZYX block starting at slice z0."""
    s = np.array([p.psf_sigma_z_um, p.psf_sigma_xy_um, p.psf_sigma_xy_um]) / p.voxel_um
    a = gaussian_filter(vol, s, truncate=3.0)
    b = gaussian_filter(vol, s * p.deep_blur_factor, truncate=3.0)
    w = ((z0 + np.arange(vol.shape[0])) / max(total_z - 1, 1))[:, None, None]
    return ((1 - w) * a + w * b).astype(np.float32)


@dataclass
class Tissue:
    """Coarse tissue fields in image frame (ZYX, spacing `coarse_um`)."""
    haze: np.ndarray        # neuropil / whole-tissue haze, 0..1
    cortex: np.ndarray      # cell-body rind density, 0..1
    illum: np.ndarray       # multiplicative illumination field
    coarse_um: float
    soma_xyz: np.ndarray = field(repr=False)  # all VNC somata in image um (X, Y, Z)

    def sample(self, name, zyx_um):
        a = getattr(self, name)
        return map_coordinates(a, (zyx_um / self.coarse_um - 0.5).T, order=1, mode="nearest")

    def upsample(self, name, shape, voxel):
        a = getattr(self, name)
        f = [s / c for s, c in zip(shape, a.shape)]
        out = zoom(a, f, order=1, grid_mode=True, mode="grid-constant")
        return out[:shape[0], :shape[1], :shape[2]].astype(np.float32)

    def local_grid(self, name, lo, hi, voxel):
        """Evaluate a coarse field on the full-res voxel grid [lo, hi)."""
        zz, yy, xx = [(np.arange(l, h) + 0.5) * voxel for l, h in zip(lo, hi)]
        a = getattr(self, name)
        idx = np.meshgrid(zz / self.coarse_um - 0.5, yy / self.coarse_um - 0.5,
                          xx / self.coarse_um - 0.5, indexing="ij")
        return map_coordinates(a, idx, order=1, mode="nearest").astype(np.float32)


def build_tissue(soma_img_um, shape_zyx, voxel, rng, p: ImagingParams, coarse_um=2.0,
                 slab_um=12.0):
    """Tissue haze + cortex from the positions of all VNC somata.

    The tissue outline is the convex hull of the somata in each transverse
    (Z-X) slab along the AP axis; the VNC cross-section is close to convex and
    the cell-body rind surrounds the neuropil, so the hull covers both.
    """
    from scipy.spatial import ConvexHull, QhullError
    from skimage.draw import polygon

    cshape = np.maximum(np.ceil(np.array(shape_zyx) * voxel / coarse_um).astype(int), 4)
    lo, hi = np.zeros(3, int), cshape
    dens = splat(soma_img_um, np.ones(len(soma_img_um)), coarse_um, lo, hi)
    cortex = gaussian_filter(dens, 4.0 / coarse_um)
    cortex /= np.percentile(cortex[cortex > 0], 99) + 1e-9 if (cortex > 0).any() else 1.0
    tissue = np.zeros(cshape, np.float32)
    zx = soma_img_um[:, [2, 0]] / coarse_um - 0.5
    ypos = soma_img_um[:, 1]
    for yi in range(cshape[1]):
        yc = (yi + 0.5) * coarse_um
        pts = zx[np.abs(ypos - yc) < slab_um]
        if len(pts) < 8:
            continue
        try:
            h = ConvexHull(pts)
        except QhullError:
            continue
        rr, cc = polygon(pts[h.vertices, 0], pts[h.vertices, 1], shape=(cshape[0], cshape[2]))
        tissue[rr, yi, cc] = 1.0
    tissue = gaussian_filter(tissue, [0.0, slab_um / coarse_um / 2, 0.0]) > 0.5
    haze = gaussian_filter(tissue.astype(np.float32), 3.0 / coarse_um)
    n = gaussian_filter(rng.normal(size=cshape), 5.0)
    haze *= np.exp(0.25 * n / (n.std() + 1e-9))
    il = gaussian_filter(rng.normal(size=cshape), 20.0 / coarse_um)
    illum = np.exp(p.illum_sigma * il / (il.std() + 1e-9))
    return Tissue(haze.astype(np.float32), np.clip(cortex, 0, 1.5).astype(np.float32),
                  illum.astype(np.float32), coarse_um, soma_img_um)


def texture_field(shape, p: ImagingParams, rng):
    """Fine-grained multiplicative texture of neuropil autofluorescence."""
    t = rng.standard_normal(shape, dtype=np.float32)
    t = gaussian_filter(t, p.texture_scale_um / p.voxel_um)
    t *= p.texture_sigma / (t.std() + 1e-9)
    return np.exp(t, out=t)


def puncta_points(tissue: Tissue, per_1e4um3, rng, weight_field="haze"):
    """Random point positions (image um, XYZ) inside tissue."""
    w = getattr(tissue, weight_field).ravel().astype(np.float64)
    vol_um3 = w.sum() * tissue.coarse_um ** 3
    n = rng.poisson(per_1e4um3 * vol_um3 / 1e4)
    if n == 0 or w.sum() == 0:
        return np.zeros((0, 3))
    idx = rng.choice(w.size, n, p=w / w.sum())
    zyx = (np.stack(np.unravel_index(idx, tissue.haze.shape), 1)
           + rng.random((n, 3))) * tissue.coarse_um
    return zyx[:, ::-1]


def detector(expected_photons, p: ImagingParams, rng, adu_per_photon=None):
    """Poisson shot noise + Gaussian read noise, 12-bit quantisation."""
    g = p.adu_per_photon if adu_per_photon is None else adu_per_photon
    counts = rng.poisson(np.clip(expected_photons, 0, None)).astype(np.float32)
    if p.pmt_excess_shape:
        k = p.pmt_excess_shape
        nz = counts > 0
        counts[nz] *= rng.gamma(k, 1.0 / k, int(nz.sum())).astype(np.float32)
    adu = counts * g + p.offset_adu
    adu += rng.normal(0, p.read_noise_adu, adu.shape).astype(np.float32)
    return np.clip(np.round(adu), 0, p.max_adu).astype(np.uint16)
