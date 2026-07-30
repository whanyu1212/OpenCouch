"""The procedural-profile lock registry does not grow without bound.

``_PROCEDURAL_PROFILE_LOCKS`` is module-level, so it outlives any single
runtime and gains an entry per user id. Pruning must reclaim idle entries
without splitting the mutex for a user with work in flight.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.memory.operations import procedural_profile
from agent.memory.operations.procedural_profile import (
    prune_idle_procedural_profile_locks,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clear_lock_registry() -> None:
    """Isolate each test from registry state left by other tests."""

    procedural_profile._PROCEDURAL_PROFILE_LOCKS.clear()  # noqa: SLF001


async def test_idle_locks_are_pruned() -> None:
    for user_id in ("user-a", "user-b", "user-c"):
        procedural_profile._procedural_profile_lock(user_id)  # noqa: SLF001

    assert len(procedural_profile._PROCEDURAL_PROFILE_LOCKS) == 3  # noqa: SLF001
    assert prune_idle_procedural_profile_locks() == 3
    assert procedural_profile._PROCEDURAL_PROFILE_LOCKS == {}  # noqa: SLF001


async def test_held_lock_is_retained() -> None:
    """A user with a mutation in flight keeps its lock."""

    lock = procedural_profile._procedural_profile_lock("user-busy")  # noqa: SLF001
    procedural_profile._procedural_profile_lock("user-idle")  # noqa: SLF001

    async with lock:
        assert prune_idle_procedural_profile_locks() == 1

    assert set(procedural_profile._PROCEDURAL_PROFILE_LOCKS) == {  # noqa: SLF001
        "user-busy"
    }


async def test_lock_with_pending_waiter_is_retained() -> None:
    """Pruning a lock with a queued waiter would split the mutex."""

    user_id = "user-contended"
    lock = procedural_profile._procedural_profile_lock(user_id)  # noqa: SLF001
    await lock.acquire()

    waiter_ran = asyncio.Event()

    async def _waiter() -> None:
        async with procedural_profile._procedural_profile_lock(user_id):  # noqa: SLF001
            waiter_ran.set()

    waiter_task = asyncio.create_task(_waiter())
    await asyncio.sleep(0)  # let the waiter queue up

    assert prune_idle_procedural_profile_locks() == 0
    assert user_id in procedural_profile._PROCEDURAL_PROFILE_LOCKS  # noqa: SLF001

    lock.release()
    await waiter_task
    assert waiter_ran.is_set()


async def test_pruned_registry_recreates_locks_on_demand() -> None:
    """Reclaiming an idle lock must not break the next mutation."""

    first = procedural_profile._procedural_profile_lock("user-recreate")  # noqa: SLF001
    prune_idle_procedural_profile_locks()
    second = procedural_profile._procedural_profile_lock("user-recreate")  # noqa: SLF001

    assert second is not first
    async with second:
        assert second.locked()


async def test_unknown_waiters_shape_fails_closed() -> None:
    """An uninspectable lock is retained rather than assumed idle."""

    class _OpaqueLock:
        def locked(self) -> bool:
            return False

    procedural_profile._PROCEDURAL_PROFILE_LOCKS["user-opaque"] = _OpaqueLock()  # type: ignore[assignment]  # noqa: SLF001

    assert prune_idle_procedural_profile_locks() == 0
    assert "user-opaque" in procedural_profile._PROCEDURAL_PROFILE_LOCKS  # noqa: SLF001
