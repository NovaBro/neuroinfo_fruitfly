"""Server configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SERVER_DIR.parent.parent

REPO_ROOT = _REPO_ROOT

FISBE_ROOT = Path(
    os.environ.get("FISBE_ROOT", _REPO_ROOT / "fisbe" / "completely")
).resolve()

# Pre-generated maximum-intensity-projection PNGs shipped with FISBe
# (fisbe_v1.0_mips.zip). Each PNG is a side-by-side composite: left half is the
# original colored raw MIP, right half is the GT instance-segmentation MIP.
# Layout: <mips>/<dataset>/<split>/<name>.png (dataset = completely|partly).
FISBE_MIPS_ROOT = Path(
    os.environ.get("FISBE_MIPS_ROOT", FISBE_ROOT.parent / "mips")
).resolve()

SAMPLE_LIST_PATH = Path(
    os.environ.get(
        "SAMPLE_LIST_PATH",
        _REPO_ROOT
        / "evaluate-instance-segmentation"
        / "assets"
        / "sample_list_per_split.txt",
    )
).resolve()

# Base dir scanned for PatchPerPix prediction sets. Each experiment under this
# base can expose two kinds of overlay: the per-voxel count map
# (``test/processed/<ckpt>/<stem>.zarr`` → ``volumes/pred_numinst``) and the
# final vote-instances labels (``test/instanced/.../<stem>.hdf`` → dataset
# ``vote_instances``). See ``services/ppp_loader.py``.
PPP_EXPERIMENTS_BASE = Path(
    os.environ.get(
        "PPP_EXPERIMENTS_BASE",
        _REPO_ROOT / "PatchPerPix" / "experiments" / "ppp_experiments",
    )
).resolve()

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
