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

SAMPLE_LIST_PATH = Path(
    os.environ.get(
        "SAMPLE_LIST_PATH",
        _REPO_ROOT
        / "evaluate-instance-segmentation"
        / "assets"
        / "sample_list_per_split.txt",
    )
).resolve()

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
