"""Tests for high-level session lifecycle orchestration helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime.session import RuntimeSessionTracker
from agent.runtime.session.service import SessionLifecycleService, SessionSweepResult


class _FakeActiveSessionManager:
    def __init__(self) -> None:
        self.persisted_ids: list[str] = []
        self.persisted_sessions: dict[str, object] = {}
        self.expired_threads: set[str] = set()

    async def list_persisted_active_session_ids(self) -> list[str]:
        return list(self.persisted_ids)

    async def load_persisted_active_session(self, thread_id: str) -> object | None:
        return self.persisted_sessions.get(thread_id)

    def session_has_expired(self, session: object) -> bool:
        thread_id = getattr(session, "thread_id", None)
        return isinstance(thread_id, str) and thread_id in self.expired_threads


class _FakeStateStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict[str, object]]] = []
        self.should_fail = False

    async def save_state(self, thread_id: str, state: dict[str, object]) -> None:
        if self.should_fail:
            raise RuntimeError("forced save failure")
        self.saved.append((thread_id, state))


class _FakeMemoryStore:
    pass


class _FakeEmbeddingProvider:
    model_name = "test-embedding"
    dimension = 1


def _build_service(
    *,
    memory_mode: MemoryMode = MemoryMode.LOCAL,
    active_session_manager: _FakeActiveSessionManager | None = None,
    state_store: _FakeStateStore | None = None,
    auto_finalize_excluded: Callable[[str], bool] | None = None,
) -> SessionLifecycleService:
    return SessionLifecycleService(
        memory_mode=memory_mode,
        session_tracker=RuntimeSessionTracker(),
        active_session_manager=active_session_manager or _FakeActiveSessionManager(),
        state_store=state_store or _FakeStateStore(),
        memory_store=_FakeMemoryStore(),
        embedding_provider=_FakeEmbeddingProvider(),
        thread_llm_clients={},
        session_sweep_interval_seconds=1.0,
        auto_finalize_excluded=auto_finalize_excluded,
    )


@pytest.mark.asyncio
async def test_list_active_thread_ids_uses_tracker_in_incognito() -> None:
    service = _build_service(memory_mode=MemoryMode.INCOGNITO)

    service._session_tracker.start_session(  # noqa: SLF001
        "thread-incognito",
        started_at="2026-05-25T00:00:00Z",
        transcript_start_index=0,
    )

    assert await service.list_active_thread_ids() == ["thread-incognito"]


@pytest.mark.asyncio
async def test_clear_session_continuity_in_state_saves_delta() -> None:
    state_store = _FakeStateStore()
    service = _build_service(state_store=state_store)

    await service.clear_session_continuity_in_state(
        "thread-1",
        {
            "therapeutic_approach": "cbt",
            "exercise_state": {
                "exercise_type": "breathing",
                "exercise_step": "intro",
                "exercise_step_id": "1",
                "exercise_version": "v1",
                "exercise_therapeutic_approach": "cbt",
            },
            "session_progress": {"turn_count": 2},
        },
    )

    assert state_store.saved == [
        (
            "thread-1",
            {
                "therapeutic_approach": None,
                "exercise_state": {
                    "exercise_type": None,
                    "exercise_step": None,
                    "exercise_step_id": None,
                    "exercise_version": None,
                    "exercise_therapeutic_approach": None,
                },
                "session_progress": {"turn_count": 2},
            },
        )
    ]


@pytest.mark.asyncio
async def test_clear_session_continuity_in_state_suppresses_errors_when_requested() -> (
    None
):
    state_store = _FakeStateStore()
    state_store.should_fail = True
    service = _build_service(state_store=state_store)

    await service.clear_session_continuity_in_state(
        "thread-1",
        {"therapeutic_approach": "cbt"},
        suppress_errors=True,
    )


def test_thread_lock_returns_same_lock_for_same_thread() -> None:
    service = _build_service()

    assert service.thread_lock("thread-a") is service.thread_lock("thread-a")
    assert service.thread_lock("thread-a") is not service.thread_lock("thread-b")


@pytest.mark.asyncio
async def test_background_tasks_start_and_stop_cleanly() -> None:
    service = _build_service()

    async def _finalize_once() -> SessionSweepResult:
        await asyncio.sleep(60)
        return SessionSweepResult()

    service.start_background_tasks(finalize_expired_sessions_once=_finalize_once)

    assert service._session_sweeper_task is not None  # noqa: SLF001
    assert not service._session_sweeper_task.done()  # noqa: SLF001

    await service.stop_background_tasks()

    assert service._session_sweeper_task is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_background_tasks_stop_is_idempotent() -> None:
    service = _build_service()

    await service.stop_background_tasks()
    await service.stop_background_tasks()

    assert service._session_sweeper_task is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_finalize_expired_sessions_once_counts_outcomes() -> None:
    manager = _FakeActiveSessionManager()
    manager.persisted_ids = [
        "thread-expired",
        "thread-fresh",
        "thread-missing",
        "thread-excluded",
    ]
    manager.persisted_sessions = {
        "thread-expired": type("Session", (), {"thread_id": "thread-expired"})(),
        "thread-fresh": type("Session", (), {"thread_id": "thread-fresh"})(),
        "thread-excluded": type("Session", (), {"thread_id": "thread-excluded"})(),
    }
    manager.expired_threads = {"thread-expired", "thread-excluded"}

    service = _build_service(
        active_session_manager=manager,
        auto_finalize_excluded=lambda thread_id: thread_id == "thread-excluded",
    )
    finalized: list[tuple[str, object | None]] = []

    async def _end_session(
        thread_id: str,
        *,
        llm_client: object | None = None,
    ) -> None:
        finalized.append((thread_id, llm_client))

    result = await service.finalize_expired_sessions_once(
        end_session=_end_session,
        effective_llm_client=lambda thread_id, override: f"llm:{thread_id}",
    )

    assert result == SessionSweepResult(
        checked=4,
        finalized=1,
        skipped_excluded=1,
        skipped_missing=1,
        skipped_not_expired=1,
        failed_to_list=False,
        failed_thread_ids=[],
    )
    assert finalized == [("thread-expired", "llm:thread-expired")]


@pytest.mark.asyncio
async def test_finalize_expired_sessions_once_marks_listing_failure() -> None:
    service = _build_service()

    async def _raise_list_failure() -> list[str]:
        raise RuntimeError("forced list failure")

    result = await service.finalize_expired_sessions_once(
        end_session=lambda *args, **kwargs: None,
        effective_llm_client=lambda thread_id, override: None,
        list_active_thread_ids=_raise_list_failure,
    )

    assert result == SessionSweepResult(failed_to_list=True)


@pytest.mark.asyncio
async def test_finalize_expired_sessions_once_continues_after_thread_failure() -> None:
    manager = _FakeActiveSessionManager()
    manager.persisted_ids = ["thread-fails", "thread-succeeds"]
    manager.persisted_sessions = {
        "thread-fails": type("Session", (), {"thread_id": "thread-fails"})(),
        "thread-succeeds": type("Session", (), {"thread_id": "thread-succeeds"})(),
    }
    manager.expired_threads = {"thread-fails", "thread-succeeds"}

    service = _build_service(active_session_manager=manager)
    finalized: list[str] = []

    async def _end_session(
        thread_id: str,
        *,
        llm_client: object | None = None,
    ) -> None:
        if thread_id == "thread-fails":
            raise RuntimeError("forced end failure")
        finalized.append(thread_id)

    result = await service.finalize_expired_sessions_once(
        end_session=_end_session,
        effective_llm_client=lambda thread_id, override: None,
    )

    assert result.checked == 2
    assert result.finalized == 1
    assert result.failed_thread_ids == ["thread-fails"]
    assert finalized == ["thread-succeeds"]
