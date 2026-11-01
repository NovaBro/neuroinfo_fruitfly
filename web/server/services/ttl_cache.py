"""Small time-bounded memoization for on-disk discovery scans.

Discovering prediction sets walks experiment trees on NFS scratch, which costs
seconds per call; several endpoints need the result on every page load. Entries
expire after ``ttl_s`` so newly finished runs still show up without a restart.
Concurrent callers for the same key wait on one computation instead of each
starting their own scan.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def ttl_cache(ttl_s: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        entries: dict[tuple, tuple[float, T]] = {}
        key_locks: dict[tuple, threading.Lock] = {}
        guard = threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args):
            key = args
            hit = entries.get(key)
            if hit is not None and time.monotonic() - hit[0] < ttl_s:
                return hit[1]
            with guard:
                lock = key_locks.setdefault(key, threading.Lock())
            with lock:
                hit = entries.get(key)
                if hit is not None and time.monotonic() - hit[0] < ttl_s:
                    return hit[1]
                value = fn(*args)
                entries[key] = (time.monotonic(), value)
                return value

        wrapper.cache_clear = entries.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator
