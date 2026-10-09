"""Process-wide cached Zarr opens for the web viewer.

Wraps each on-disk zarr root with ``zarr.experimental.cache_store.CacheStore``
backed by an in-memory store so concurrent / repeated reads of the same Blosc
chunks (raw vs RGB vs GT) do not re-decompress from scratch.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock

import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import LocalStore, MemoryStore

from config import WEB_ZARR_CACHE_BYTES

# Cap how many distinct zarr roots stay wrapped; each root owns up to
# WEB_ZARR_CACHE_BYTES of cached chunks. Evict least-recently-used roots.
_MAX_CACHED_ROOTS = 4

_lock = Lock()
_stores: OrderedDict[str, CacheStore] = OrderedDict()


def _cached_store(zarr_path: Path) -> CacheStore:
    key = str(zarr_path.resolve())
    with _lock:
        existing = _stores.get(key)
        if existing is not None:
            _stores.move_to_end(key)
            return existing

        local = LocalStore(root=key, read_only=True)
        store = CacheStore(
            local,
            cache_store=MemoryStore(),
            max_size=WEB_ZARR_CACHE_BYTES,
        )
        _stores[key] = store
        while len(_stores) > _MAX_CACHED_ROOTS:
            _stores.popitem(last=False)
        return store


def open_array(zarr_path: Path | str, path: str) -> zarr.Array:
    """Open a zarr array under ``zarr_path`` via the process-wide chunk cache."""
    store = _cached_store(Path(zarr_path))
    return zarr.open(store, mode="r", path=path)
