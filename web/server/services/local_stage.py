"""Lazy copy-on-first-open of volume inputs onto node-local SSD.

When ``WEB_STAGE_DIR`` is set (or derived from ``$SLURM_TMPDIR``), the first
request for a FISBe zarr / prediction TIFF / PPP hdf copies it under the stage
dir; later requests read from local disk. Concurrent requests for the same
source wait on a lock and share one copy.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path

from config import WEB_STAGE_DIR

_COMPLETE_MARKER_SUFFIX = ".stage_complete"
_LOCK_SUFFIX = ".stage_lock"
_LOCK_STALE_S = 3600


def _stable_key(src: Path) -> str:
    resolved = str(src.resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{src.name}"


def _acquire_lock(lock_path: Path) -> int:
    """Exclusive create lock; returns an open fd. Retries while lock is held."""
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            return fd
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > _LOCK_STALE_S:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            time.sleep(0.25)


def _release_lock(lock_path: Path, fd: int) -> None:
    try:
        os.close(fd)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def ensure_local(src: Path) -> Path:
    """Return a local staged path for ``src``, or ``src`` when staging is off.

    Directories (``.zarr`` roots) are copytree'd to ``STAGE/<key>/``; files are
    copy2'd to ``STAGE/<key>/<filename>``. A sibling ``.stage_complete`` marker
    makes subsequent calls a no-op.
    """
    if WEB_STAGE_DIR is None:
        return src

    src = src.resolve()
    if not src.exists():
        return src

    WEB_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    key = _stable_key(src)
    dest = WEB_STAGE_DIR / key
    marker = WEB_STAGE_DIR / f"{key}{_COMPLETE_MARKER_SUFFIX}"
    lock_path = WEB_STAGE_DIR / f"{key}{_LOCK_SUFFIX}"
    is_dir = src.is_dir()
    staged = dest if is_dir else dest / src.name

    if marker.is_file() and staged.exists():
        return staged

    fd = _acquire_lock(lock_path)
    try:
        if marker.is_file() and staged.exists():
            return staged

        tmp = WEB_STAGE_DIR / f"{key}.partial"
        if tmp.exists():
            if tmp.is_dir():
                shutil.rmtree(tmp)
            else:
                tmp.unlink()

        if is_dir:
            shutil.copytree(src, tmp, symlinks=False)
        else:
            tmp.mkdir(parents=True)
            shutil.copy2(src, tmp / src.name)

        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        os.rename(tmp, dest)
        marker.write_text(f"{src}\n", encoding="utf-8")
        return staged
    finally:
        _release_lock(lock_path, fd)
