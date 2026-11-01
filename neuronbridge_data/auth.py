"""Repo-root .env loading and neuPrint token resolution."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_repo_dotenv() -> None:
    """Load repo-root .env into os.environ for keys that are not already set."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_token(explicit: str | None) -> str:
    """Resolve neuPrint token: --token > env > repo-root .env neuprint_key."""
    load_repo_dotenv()
    token = (
        explicit
        or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("neuprint_key")
    )
    if not token:
        raise SystemExit(
            "No neuPrint token. Pass --token, set NEUPRINT_APPLICATION_CREDENTIALS, "
            "or add neuprint_key to the repo-root .env "
            "(from neuprint.janelia.org → Account)."
        )
    return token.strip()
