"""All design choices in one place; the config is copied into the manifest so a
dataset can be regenerated exactly from (config, pool, manifest)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class BrainConfig:
    # --- data source (male CNS v1.0, skeletons pre-registered to JRC2018U) ---
    bucket: str = "https://storage.googleapis.com/flyem-male-cns/v1.0"
    swc_dir: str = "segmentation/skeletons-unisex-template/skeletons-swc"  # microns, JRC2018U
    annotations: str = ("connectome-data/flat-connectome/"
                        "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    superclasses: tuple = ("cb_intrinsic", "visual_projection", "visual_centrifugal",
                           "descending_neuron", "ascending_neuron", "cb_motor",
                           "cb_endocrine", "cb_sensory")
    statuses: tuple = ("Traced",)
    template_bbox_um: tuple = (627.38, 293.36, 172.9)     # JRC2018U extent (x, y, z)
    footprint_um: float = 4.0                              # coarse occupancy grid
    radius_scale: float = 0.75                             # EM radii -> DPX-cleared LM

    # --- image geometry (FISBe: native 40x tiles, 0.44 um isotropic, ~400x700x700) ---
    voxel_um: float = 0.44
    crop_um: tuple = (308.0, 293.0, 173.0)                 # x, y, z -> ~700 x 666 x 393 voxels
    crop_center_x_um: tuple = (190.0, 437.0)               # central-brain tiles, some optic lobe
    min_cable_in_crop_um: float = 150.0

    # --- design of the image set ---
    n_images: int = 5000
    count_bins: tuple = ((1, 2), (3, 5), (6, 10), (11, 20), (21, 35), (36, 60))
    count_weights: tuple = (0.10, 0.15, 0.25, 0.25, 0.15, 0.10)
    overlap_modes: tuple = ("isolated", "natural", "crowded")
    overlap_weights: tuple = (0.25, 0.45, 0.30)
    isolated_max_n: int = 20            # above this "isolated" is infeasible -> "natural"
    isolation_margin_cells: int = 1     # footprint dilation (x4 um) for "isolated"
    same_colour_overlap_p: float = 0.5  # crowded images: force >=1 overlapping same-colour pair
    split_fractions: tuple = (0.8, 0.1, 0.1)   # train / val / test, split by cell type
    tags: tuple = ("HA", "V5", "FLAG")

    # --- placement jitter (skeletons are already in template/LM space) ---
    scale_jitter: float = 0.03
    max_rot_deg: float = 3.0
    elastic_um: float = 2.0

    seed: int = 20260926

    def to_json(self):
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        return cls(**d)
