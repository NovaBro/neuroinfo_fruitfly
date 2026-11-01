"""Lazy copy-on-first-open of volume inputs onto node-local SSD.

When ``WEB_STAGE_DIR`` is set (or derived from ``$SLURM_TMPDIR``), the first
request for a FISBe zarr / prediction TIFF / PPP hdf starts copying it under the
stage dir in a background thread and is served from the source meanwhile (a
synchronous ~1.4 GB copy off NFS made first opens ~4x slower). Once the copy
completes, later requests read from local disk.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from config import WEB_STAGE_DIR

_logger = logging.getLogger("uvicorn.error")

_COMPLETE_MARKER_SUFFIX = ".stage_complete"
_LOCK_SUFFIX = ".stage_lock"
_LOCK_STALE_S = 3600

# Stage keys with a background copy running in this process.
_inflight: set[str] = set()
_inflight_lock = threading.Lock()


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


def _staged_path(src: Path, key: str) -> Path:
    dest = WEB_STAGE_DIR / key
    return dest if src.is_dir() else dest / src.name


def _is_staged(src: Path, key: str) -> bool:
    marker = WEB_STAGE_DIR / f"{key}{_COMPLETE_MARKER_SUFFIX}"
    return marker.is_file() and _staged_path(src, key).exists()


def ensure_local(src: Path) -> Path:
    """Return the staged copy of ``src`` if ready, else ``src`` (and stage it).

    Returns ``src`` unchanged when staging is off. Otherwise, when no complete
    copy exists yet, starts one in a background thread and returns ``src`` so
    the caller is not blocked on the copy.
    """
    if WEB_STAGE_DIR is None:
        return src

    src = src.resolve()
    if not src.exists():
        return src

    key = _stable_key(src)
    if _is_staged(src, key):
        return _staged_path(src, key)

    with _inflight_lock:
        if key in _inflight:
            return src
        _inflight.add(key)
    threading.Thread(
        target=_stage_in_background, args=(src, key), daemon=True
    ).start()
    return src


def _stage_in_background(src: Path, key: str) -> None:
    try:
        _stage(src, key)
    except Exception:  # noqa: BLE001 — staging is an optimisation only
        _logger.exception("staging failed for %s", src)
    finally:
        with _inflight_lock:
            _inflight.discard(key)


def _stage(src: Path, key: str) -> Path:
    """Copy ``src`` under the stage dir (directories via copytree, files copy2).

    A sibling ``.stage_complete`` marker records a finished copy; a file lock
    keeps concurrent processes from copying the same source twice.
    """
    WEB_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEB_STAGE_DIR / key
    marker = WEB_STAGE_DIR / f"{key}{_COMPLETE_MARKER_SUFFIX}"
    lock_path = WEB_STAGE_DIR / f"{key}{_LOCK_SUFFIX}"
    is_dir = src.is_dir()
    staged = _staged_path(src, key)

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
