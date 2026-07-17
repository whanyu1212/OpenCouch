"""Tests for per-thread runtime lock ownership."""

from __future__ import annotations

import asyncio

import pytest

from agent.runtime.session.lock import ThreadLockManager


def _not_tracked(_thread_id: str) -> bool:
    return False


def test_get_lock_returns_same_lock_for_same_thread() -> None:
    manager = ThreadLockManager()

    assert manager.get_lock("thread-a") is manager.get_lock("thread-a")
    assert manager.get_lock("thread-a") is not manager.get_lock("thread-b")


def test_prune_removes_idle_lock_and_allows_recreation() -> None:
    manager = ThreadLockManager()
    original = manager.get_lock("thread-idle")

    assert manager.prune_idle_locks(is_tracked=_not_tracked) == 1
    assert manager.get_lock("thread-idle") is not original


@pytest.mark.asyncio
async def test_prune_keeps_held_lock() -> None:
    manager = ThreadLockManager()
    lock = manager.get_lock("thread-held")
    await lock.acquire()

    try:
        assert manager.prune_idle_locks(is_tracked=_not_tracked) == 0
        assert manager.get_lock("thread-held") is lock
    finally:
        lock.release()


def test_prune_keeps_tracked_thread() -> None:
    manager = ThreadLockManager()
    lock = manager.get_lock("thread-tracked")

    assert (
        manager.prune_idle_locks(
            is_tracked=lambda thread_id: thread_id == "thread-tracked"
        )
        == 0
    )
    assert manager.get_lock("thread-tracked") is lock


@pytest.mark.asyncio
async def test_prune_keeps_lock_with_pending_waiter() -> None:
    manager = ThreadLockManager()
    lock = manager.get_lock("thread-waited")
    await lock.acquire()

    async def _waiter() -> None:
        async with manager.get_lock("thread-waited"):
            pass

    task = asyncio.create_task(_waiter())
    try:
        await asyncio.sleep(0)
        assert manager.prune_idle_locks(is_tracked=_not_tracked) == 0
        assert manager.get_lock("thread-waited") is lock
    finally:
        lock.release()
        await task


@pytest.mark.asyncio
async def test_prune_keeps_lock_in_release_handoff_window() -> None:
    manager = ThreadLockManager()
    lock = manager.get_lock("thread-handoff")
    await lock.acquire()

    async def _waiter() -> None:
        async with manager.get_lock("thread-handoff"):
            pass

    task = asyncio.create_task(_waiter())
    await asyncio.sleep(0)
    assert lock.locked() is True

    lock.release()
    assert lock.locked() is False
    assert manager.prune_idle_locks(is_tracked=_not_tracked) == 0
    assert manager.get_lock("thread-handoff") is lock

    await asyncio.sleep(0)
    await task


@pytest.mark.asyncio
async def test_prune_ignores_cancelled_waiter() -> None:
    manager = ThreadLockManager()
    lock = manager.get_lock("thread-cancelled")
    await lock.acquire()

    async def _waiter() -> None:
        async with manager.get_lock("thread-cancelled"):
            pass

    task = asyncio.create_task(_waiter())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lock.release()
    await asyncio.sleep(0)

    assert manager.prune_idle_locks(is_tracked=_not_tracked) == 1


def test_lock_has_live_waiters_matches_cpython_contract() -> None:
    lock = asyncio.Lock()

    assert hasattr(lock, "_waiters")
    assert ThreadLockManager._lock_has_live_waiters(lock) is False  # noqa: SLF001


def test_unknown_waiter_shape_fails_closed() -> None:
    class _UnknownLock:
        pass

    assert ThreadLockManager._lock_has_live_waiters(_UnknownLock()) is True  # type: ignore[arg-type]  # noqa: SLF001


def test_tracking_failure_retains_affected_lock_and_continues(caplog) -> None:
    manager = ThreadLockManager()
    failed_lock = manager.get_lock("thread-failed")
    manager.get_lock("thread-idle")

    def _is_tracked(thread_id: str) -> bool:
        if thread_id == "thread-failed":
            raise RuntimeError("forced tracking failure")
        return False

    assert manager.prune_idle_locks(is_tracked=_is_tracked) == 1
    assert manager.get_lock("thread-failed") is failed_lock
    assert "failed to determine tracking state" in caplog.text


def test_manager_rejects_access_from_different_event_loop() -> None:
    manager = ThreadLockManager()

    async def _get_lock() -> None:
        manager.get_lock("thread-loop")

    asyncio.run(_get_lock())

    with pytest.raises(RuntimeError, match="different event loop"):
        asyncio.run(_get_lock())


def test_manager_rejects_access_from_different_thread() -> None:
    manager = ThreadLockManager()
    manager.get_lock("thread-main")
    errors: list[BaseException] = []

    def _access_manager() -> None:
        try:
            manager.get_lock("thread-worker")
        except BaseException as exc:
            errors.append(exc)

    import threading

    worker = threading.Thread(target=_access_manager)
    worker.start()
    worker.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "different thread" in str(errors[0])
