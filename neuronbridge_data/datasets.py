"""FlyEM dataset URLs and constants."""

from __future__ import annotations

NEUPRINT_SERVER = "https://neuprint.janelia.org"
MAX_VOXELS = 2000 ** 3
TIGHT_PAD = 2

# seg_mode:
#   dvid_sparsevol — public DVID sparsevol (body-ID exact, neuron-only)
#   cloudvolume_body — CloudVolume seg masked to neuPrint bodyId
#   cloudvolume_skel_labels — sample labels along skeleton (MANC supervoxels)
DATASETS = {
    "hemibrain": {
        "neuprint": "hemibrain:v1.2.1",
        "seg_mode": "dvid_sparsevol",
        "dvid_server": "https://hemibrain-dvid.janelia.org",
        "dvid_uuid": "52a13",
        "dvid_instance": "segmentation",
        # Used only to clamp skeleton bbox if we ever fall back:
        "seg": (
            "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/"
            "v1.2/segmentation"
        ),
    },
    "malecns": {
        "neuprint": "male-cns:v1.0",
        "seg_mode": "cloudvolume_body",
        "seg": "precomputed://gs://flyem-male-cns/v1.0/segmentation",
    },
    "manc": {
        "neuprint": "manc:v1.2.1",
        # Public MANC DVID is auth-walled; use supervoxel labels along skeleton.
        "seg_mode": "cloudvolume_skel_labels",
        "seg": (
            "precomputed://gs://vnc-v3-seg-3d2f1c08fd4720848061f77362dc6c17/"
            "rc5_wsexp"
        ),
        "dvid_server": "https://emdata5-avempartha.janelia.org",
        "dvid_uuid": "428759",
        "dvid_instance": "segmentation",
    },
}
