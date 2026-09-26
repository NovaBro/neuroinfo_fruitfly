"""Placing EM neurons into a light-microscopy-like image frame.

Image frame convention (matches FlyLight / JRC2018 VNC stacks):
    X = left-right, Y = anterior-posterior (long VNC axis), Z = dorso-ventral
    (imaging depth). Arrays are stored Z, Y, X.

In male CNS EM space, x is left-right, z is the long VNC axis and y is
dorso-ventral, so the frame change is (x, y, z)_EM -> (x, z, y)_image.

DPX-cleared light microscopy tissue is smaller than the EM volume. Fitting an
affine between the release's raw-EM and JRC2018U-template skeletons of a brain
neuron gives singular values ~0.63-0.79, so the default scale is 0.70 with
per-sample jitter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

EM_TO_IMAGE = np.array([[1, 0, 0],   # X <- x
                        [0, 0, 1],   # Y <- z
                        [0, 1, 0]],  # Z <- y
                       dtype=float)


def region_box_em(annot: pd.DataFrame, neuromeres=("T2",), margin_um=15.0):
    """EM-space bounding box (min, max) for a VNC region from soma positions."""
    vnc = annot[annot.superclass.isin(["vnc_intrinsic", "ascending_neuron", "vnc_motor"])]
    vnc = vnc.dropna(subset=["soma_x_um"])
    all_xyz = vnc[["soma_x_um", "soma_y_um", "soma_z_um"]].values
    lo = np.percentile(all_xyz, 0.5, axis=0) - margin_um
    hi = np.percentile(all_xyz, 99.5, axis=0) + margin_um
    sel = vnc[vnc.somaNeuromere.isin(list(neuromeres))]
    if len(sel):
        z = sel.soma_z_um.values
        lo[2] = np.percentile(z, 2) - margin_um
        hi[2] = np.percentile(z, 98) + margin_um
    return lo, hi


@dataclass
class Placement:
    """EM (um) -> image (um, origin at crop corner) mapping for one sample."""
    center_em: np.ndarray
    rot: np.ndarray          # 3x3, includes EM->image axis permutation
    scale: np.ndarray        # (3,) anisotropic scale
    shape_um: np.ndarray     # crop size in image frame (X, Y, Z) um
    disp: np.ndarray | None = None       # (3, gx, gy, gz) displacement grid, um
    disp_spacing: float = 40.0

    @classmethod
    def random(cls, box_lo, box_hi, rng, scale=0.70, scale_jitter=0.05,
               max_rot_deg=4.0, inplane_rot_deg=0.0, elastic_um=2.5,
               elastic_spacing=40.0):
        center = (np.asarray(box_lo) + np.asarray(box_hi)) / 2
        s = scale * (1 + rng.uniform(-scale_jitter, scale_jitter, 3))
        ang = rng.uniform(-max_rot_deg, max_rot_deg, 3)
        ang[2] += rng.uniform(-inplane_rot_deg, inplane_rot_deg)  # about imaging axis
        r = Rotation.from_euler("xyz", ang, degrees=True).as_matrix()
        rot = r @ EM_TO_IMAGE
        extent_img = np.abs(EM_TO_IMAGE @ (np.asarray(box_hi) - box_lo)) * s
        disp = None
        if elastic_um > 0:
            g = np.maximum(2, np.ceil(extent_img / elastic_spacing).astype(int) + 3)
            disp = rng.normal(0, 1, (3, *g))
            disp = np.stack([gaussian_filter(d, 1.0) for d in disp])
            disp *= elastic_um / (disp.std() + 1e-9)
        return cls(center, rot, s, extent_img, disp, elastic_spacing)

    def __call__(self, xyz_em: np.ndarray) -> np.ndarray:
        p = (np.asarray(xyz_em) - self.center_em) @ self.rot.T
        p = p * self.scale + self.shape_um / 2
        if self.disp is not None:
            gi = (p / self.disp_spacing + 1.0).T  # grid index coords
            d = np.stack([map_coordinates(c, gi, order=1, mode="nearest")
                          for c in self.disp], axis=1)
            p = p + d
        return p

    @property
    def radius_scale(self) -> float:
        return float(np.mean(self.scale))
