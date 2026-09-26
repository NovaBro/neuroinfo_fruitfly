"""Access to the public male CNS (JRCFIB2022M, v1.0) connectome release.

Everything here uses the public Google Cloud bucket ``gs://flyem-male-cns``
(served over https, no token required):

* body annotations (type, superclass, hemilineage, soma location, ...)
* per-neuron SWC skeletons in raw EM space (8 nm voxels, radii included)
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0"
ANNOT_URL = (f"{BUCKET}/connectome-data/flat-connectome/"
             "body-annotations-male-cns-v1.0-minconf-0.5.feather")
SWC_URL = f"{BUCKET}/segmentation/skeletons-malecns/skeletons-swc/{{bid}}.swc"
EM_VOXEL_UM = 0.008  # male CNS EM voxel size (8 nm isotropic)

# Superclasses that contribute neurites to a VNC image.
VNC_SUPERCLASSES = ("vnc_intrinsic", "ascending_neuron", "descending_neuron",
                    "vnc_motor", "vnc_efferent", "vnc_sensory")


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def load_annotations(cache_dir: str) -> pd.DataFrame:
    """Body annotation table (~210k rows). Adds soma position in microns."""
    fp = _download(ANNOT_URL, os.path.join(cache_dir, os.path.basename(ANNOT_URL)))
    df = pd.read_feather(fp)
    has = df.somaLocation.notna()
    soma = np.full((len(df), 3), np.nan)
    soma[has.values] = np.stack(df.loc[has, "somaLocation"].values) * EM_VOXEL_UM
    df["soma_x_um"], df["soma_y_um"], df["soma_z_um"] = soma.T
    # One hemilineage column: Truman (VNC) naming first, Ito/Lee (brain) second.
    df["hemilineage"] = df["trumanHl"].fillna(df["itoleeHl"])
    return df


@dataclass
class Skeleton:
    body_id: int
    xyz: np.ndarray      # (N, 3) microns, EM space
    radius: np.ndarray   # (N,) microns
    parent: np.ndarray   # (N,) index of parent node, -1 for root
    soma_idx: int | None = None  # node index of the soma (if any)

    @property
    def edges(self) -> np.ndarray:
        child = np.flatnonzero(self.parent >= 0)
        return np.stack([child, self.parent[child]], axis=1)

    def cable_length(self) -> float:
        e = self.edges
        return float(np.linalg.norm(self.xyz[e[:, 0]] - self.xyz[e[:, 1]], axis=1).sum())


def load_skeleton(body_id: int, cache_dir: str, soma_um: np.ndarray | None = None) -> Skeleton:
    """Download (cached) and parse one raw-EM-space SWC skeleton."""
    fp = _download(SWC_URL.format(bid=body_id),
                   os.path.join(cache_dir, "skeletons", f"{body_id}.swc"))
    a = np.loadtxt(fp, comments="#", ndmin=2)
    ids = a[:, 0].astype(np.int64)
    lut = {n: i for i, n in enumerate(ids)}
    parent = np.array([lut.get(int(p), -1) for p in a[:, 6]], dtype=np.int64)
    xyz = a[:, 2:5] * EM_VOXEL_UM
    radius = a[:, 5] * EM_VOXEL_UM
    soma_idx = None
    if soma_um is not None and np.all(np.isfinite(soma_um)):
        d = np.linalg.norm(xyz - soma_um, axis=1)
        near = np.flatnonzero(d < 6.0)
        if near.size:
            soma_idx = int(near[np.argmax(radius[near])])
    return Skeleton(int(body_id), xyz, radius, parent, soma_idx)
