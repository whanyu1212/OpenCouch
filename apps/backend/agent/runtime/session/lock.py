"""Per-thread lock ownership for the persistent runtime."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_MISSING_WAITERS = object()


class ThreadLockManager:
    """Own in-process locks used to serialize work for each thread.

    The manager is intentionally single-threaded and single-event-loop, matching
    the persistent runtime. Locks must be retrieved immediately before use and
    must not be cached outside the operation that acquires them.
    """

    def __init__(self) -> None:
        """Initialize an empty per-thread lock registry."""
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._owner_thread_id: int | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._affinity_guard = threading.Lock()

    def _assert_owner(self) -> None:
        """Bind to, then enforce, one OS thread and asyncio event loop."""
        current_thread_id = threading.get_ident()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        with self._affinity_guard:
            if self._owner_thread_id is None:
                self._owner_thread_id = current_thread_id
            elif self._owner_thread_id != current_thread_id:
                raise RuntimeError("ThreadLockManager accessed from a different thread")

            if current_loop is not None:
                if self._owner_loop is None:
                    self._owner_loop = current_loop
                elif self._owner_loop is not current_loop:
                    raise RuntimeError(
                        "ThreadLockManager accessed from a different event loop"
                    )
            elif self._owner_loop is not None:
                raise RuntimeError(
                    "ThreadLockManager accessed outside its owning event loop"
                )

    def get_lock(self, thread_id: str) -> asyncio.Lock:
        """Return the in-process lock for one thread."""
        self._assert_owner()
        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_locks[thread_id] = lock
        return lock

    @staticmethod
    def _lock_has_live_waiters(lock: asyncio.Lock) -> bool:
        """Return whether a lock has a live waiter or cannot be inspected safely.

        ``asyncio.Lock.release`` clears ``_locked`` and wakes the first waiter's
        future, but it does not remove that waiter from ``_waiters``. During that
        handoff window ``locked()`` is false while a live waiter is still queued;
        pruning then would split the per-thread mutex. Because ``_waiters`` is a
        private implementation detail, an unknown shape fails closed by keeping
        the lock.
        """
        waiters: Any = getattr(lock, "_waiters", _MISSING_WAITERS)
        if waiters is _MISSING_WAITERS:
            return True
        if not waiters:
            return False
        try:
            return any(not waiter.cancelled() for waiter in waiters)
        except (AttributeError, TypeError):
            return True

    def prune_idle_locks(self, *, is_tracked: Callable[[str], bool]) -> int:
        """Drop locks for threads with no held, pending, or tracked work.

        ``is_tracked`` must be synchronous, non-yielding, and must not call back
        into this manager. Callback failures retain the affected lock and do not
        abort pruning of other entries.
        """
        self._assert_owner()
        pruned = 0
        for thread_id, lock in list(self._thread_locks.items()):
            if lock.locked() or self._lock_has_live_waiters(lock):
                continue
            try:
                tracked = is_tracked(thread_id)
            except Exception:
                logger.warning(
                    "failed to determine tracking state for thread %s; retaining lock",
                    thread_id,
                    exc_info=True,
                )
                continue
            if not tracked:
                del self._thread_locks[thread_id]
                pruned += 1
        return pruned
