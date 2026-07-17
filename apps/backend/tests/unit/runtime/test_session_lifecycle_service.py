"""Tests for high-level session lifecycle orchestration helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest

import agent.runtime.finalization as finalization_module
from agent.audit.capture import SafetyEventCaptureResult
from agent.memory.modes import MemoryMode
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.runtime.session import RuntimeSessionTracker, ThreadLockManager
from agent.runtime.session.active_session import PersistedActiveSessionState
from agent.runtime.session.service import SessionLifecycleService, SessionSweepResult
from agent.runtime.types import SessionStatus
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState


class _FakeActiveSessionManager:
    def __init__(self) -> None:
        self.persisted_ids: list[str] = []
        self.persisted_sessions: dict[str, object] = {}
        self.expired_threads: set[str] = set()
        self.saved_sessions: list[object] = []
        self.rotation_required: list[str] = []
        self.cleared_mutations: list[tuple[str, str]] = []

    async def list_persisted_active_session_ids(self) -> list[str]:
        return list(self.persisted_ids)

    async def load_persisted_active_session(self, thread_id: str) -> object | None:
        return self.persisted_sessions.get(thread_id)

    def session_has_expired(self, session: object) -> bool:
        thread_id = getattr(session, "thread_id", None)
        return isinstance(thread_id, str) and thread_id in self.expired_threads

    @asynccontextmanager
    async def active_session_mutation(
        self, thread_id: str, **_kwargs: object
    ) -> AsyncIterator[str | None]:
        yield "fake-mutation-token"

    async def delete_persisted_active_session(self, thread_id: str) -> None:
        self.persisted_sessions.pop(thread_id, None)

    async def save_persisted_active_session(self, session: object) -> None:
        self.saved_sessions.append(session)

    async def set_active_session_rotation_required(self, thread_id: str) -> None:
        self.rotation_required.append(thread_id)

    async def clear_active_session_mutation(
        self, thread_id: str, mutation_token: str
    ) -> None:
        self.cleared_mutations.append((thread_id, mutation_token))


class _FakeStateStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict[str, object]]] = []
        self.should_fail = False
        self.calls: list[str] = []

    async def save_state(self, thread_id: str, state: dict[str, object]) -> None:
        if self.should_fail:
            raise RuntimeError("forced save failure")
        self.calls.append("save_state")
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
    session_tracker: RuntimeSessionTracker | None = None,
    auto_finalize_excluded: Callable[[str], bool] | None = None,
) -> SessionLifecycleService:
    return SessionLifecycleService(
        memory_mode=memory_mode,
        session_tracker=session_tracker or RuntimeSessionTracker(),
        thread_lock_manager=ThreadLockManager(),
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


def test_prune_delegates_runtime_tracking_state() -> None:
    service = _build_service()
    service._session_tracker.start_session(  # noqa: SLF001
        "thread-tracked",
        started_at="2026-05-25T00:00:00Z",
        transcript_start_index=0,
    )
    lock = service.thread_lock("thread-tracked")

    assert service.prune_idle_thread_locks() == 0
    assert service.thread_lock("thread-tracked") is lock


@pytest.mark.asyncio
async def test_record_successful_turn_persists_tracking_and_marks_rotation() -> None:
    tracker = RuntimeSessionTracker()
    tracker.start_session(
        "thread-1",
        started_at="2026-07-17T00:00:00Z",
        transcript_start_index=1,
    )
    manager = _FakeActiveSessionManager()
    service = _build_service(
        session_tracker=tracker,
        active_session_manager=manager,
    )
    final_state = cast(
        AgentState,
        {
            "therapeutic_approach": "cbt",
            "crisis": {"level": 2},
            "transcript": [
                {"role": "assistant", "content": "prior"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "response"},
            ],
        },
    )

    await service._record_successful_turn(
        "thread-1",
        final_state,
        session_transcript_soft_limit=2,
    )

    persisted = manager.saved_sessions[-1]
    assert getattr(persisted, "max_crisis_level") == 2
    assert getattr(persisted, "session_buffer").approach_counts == {"cbt": 1}
    assert manager.rotation_required == ["thread-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final_state",
    [
        {
            "diagnostics": {
                "openai_triage_no_clarification_reason": "explicit_privacy_control"
            },
            "transcript": [],
        },
        {
            "diagnostics": {},
            "transcript": [{"role": "user", "content": "please forget that"}],
        },
    ],
)
async def test_record_successful_turn_clears_held_candidates_for_memory_control(
    final_state: dict[str, Any],
) -> None:
    tracker = RuntimeSessionTracker()
    tracker.start_session(
        "thread-1",
        started_at="2026-07-17T00:00:00Z",
        transcript_start_index=0,
    )
    buffer = tracker.session_memory_buffer_for_thread("thread-1")
    buffer.held_semantic_candidates.append(cast(Any, object()))
    buffer.held_procedural_candidates.append(cast(Any, object()))
    service = _build_service(session_tracker=tracker)

    await service._record_successful_turn(
        "thread-1",
        cast(AgentState, final_state),
        session_transcript_soft_limit=None,
    )

    assert buffer.held_semantic_candidates == []
    assert buffer.held_procedural_candidates == []


@pytest.mark.asyncio
async def test_complete_successful_turn_tracks_before_shared_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = RuntimeSessionTracker()
    tracker.start_session(
        "thread-1",
        started_at="2026-07-17T00:00:00Z",
        transcript_start_index=0,
    )
    manager = _FakeActiveSessionManager()
    service = _build_service(
        session_tracker=tracker,
        active_session_manager=manager,
    )
    state = cast(AgentState, {"transcript": []})
    context = cast(WorkflowContext, object())
    calls: list[str] = []

    async def _finalize_successful_turn(**kwargs: Any) -> SafetyEventCaptureResult:
        assert manager.saved_sessions
        assert kwargs["thread_id"] == "thread-1"
        assert kwargs["final_state"] is state
        assert kwargs["workflow_context"] is context
        assert kwargs["mutation_token"] == "mutation-token"
        assert kwargs["capture_safety_event"] is False
        calls.append("finalize")
        return SafetyEventCaptureResult(
            kind="crisis_response",
            status="skipped",
            reason="safety_capture_not_required",
        )

    async def _ensure_sdk_turn_recorded(
        thread_id: str,
        *,
        user_message: str,
        final_state: AgentState,
    ) -> None:
        del thread_id, user_message, final_state

    monkeypatch.setattr(
        finalization_module,
        "finalize_successful_turn",
        _finalize_successful_turn,
    )

    result = await service.complete_successful_turn(
        thread_id="thread-1",
        user_message="hello",
        final_state=state,
        workflow_context=context,
        mutation_token="mutation-token",
        ensure_sdk_turn_recorded=_ensure_sdk_turn_recorded,
        session_transcript_soft_limit=None,
        capture_safety_event=False,
    )

    assert result.status == "skipped"
    assert calls == ["finalize"]


@pytest.mark.asyncio
async def test_complete_successful_turn_stops_when_tracking_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _build_service()
    finalized = False

    async def _fail_tracking(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("tracking failed")

    async def _finalize_successful_turn(**kwargs: Any) -> SafetyEventCaptureResult:
        nonlocal finalized
        del kwargs
        finalized = True
        return SafetyEventCaptureResult(kind="crisis_response", status="captured")

    monkeypatch.setattr(service, "_record_successful_turn", _fail_tracking)
    monkeypatch.setattr(
        finalization_module,
        "finalize_successful_turn",
        _finalize_successful_turn,
    )

    with pytest.raises(RuntimeError, match="tracking failed"):
        await service.complete_successful_turn(
            thread_id="thread-1",
            user_message="hello",
            final_state=cast(AgentState, {}),
            workflow_context=cast(WorkflowContext, object()),
            mutation_token="mutation-token",
            ensure_sdk_turn_recorded=cast(Any, object()),
            session_transcript_soft_limit=None,
        )

    assert finalized is False


@pytest.mark.asyncio
async def test_complete_successful_turn_propagates_finalization_failure_after_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = RuntimeSessionTracker()
    tracker.start_session(
        "thread-1",
        started_at="2026-07-17T00:00:00Z",
        transcript_start_index=0,
    )
    manager = _FakeActiveSessionManager()
    service = _build_service(
        session_tracker=tracker,
        active_session_manager=manager,
    )

    async def _fail_finalization(**kwargs: Any) -> SafetyEventCaptureResult:
        del kwargs
        raise RuntimeError("finalization failed")

    monkeypatch.setattr(
        finalization_module,
        "finalize_successful_turn",
        _fail_finalization,
    )

    with pytest.raises(RuntimeError, match="finalization failed"):
        await service.complete_successful_turn(
            thread_id="thread-1",
            user_message="hello",
            final_state=cast(AgentState, {"transcript": []}),
            workflow_context=cast(WorkflowContext, object()),
            mutation_token="mutation-token",
            ensure_sdk_turn_recorded=cast(Any, object()),
            session_transcript_soft_limit=None,
        )

    assert manager.saved_sessions


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
        finalize_only_if_expired: bool = False,
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
        finalize_only_if_expired: bool = False,
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


def _active_session(thread_id: str) -> PersistedActiveSessionState:
    return PersistedActiveSessionState(
        thread_id=thread_id,
        started_at="2026-06-19T00:00:00Z",
        last_active_at="2026-06-19T00:01:00Z",
        transcript_start_index=0,
        max_crisis_level=0,
        session_buffer=SessionMemoryBuffer(session_id=thread_id),
    )


@pytest.mark.asyncio
async def test_end_session_unlocked_skips_renewed_session_under_lock() -> None:
    # Regression for #164 (sweeper TOCTOU): the sweeper's expiry check runs
    # without the lock; a concurrent turn can renew the session before the lock
    # is acquired. With finalize_only_if_expired=True, end_session_unlocked must
    # re-check expiry against the fresh persisted row under the lock and skip
    # finalizing a session that is no longer expired.
    manager = _FakeActiveSessionManager()
    manager.persisted_sessions = {"thread-1": _active_session("thread-1")}
    manager.expired_threads = set()  # session is NOT expired anymore (renewed)
    service = _build_service(active_session_manager=manager)

    get_state_calls: list[str] = []

    async def _get_state(thread_id: str) -> object | None:
        get_state_calls.append(thread_id)
        return {"transcript": []}

    result = await service.end_session_unlocked(
        "thread-1",
        effective_llm_client=lambda thread_id, override: None,
        session_status_unlocked=lambda thread_id: _coro(SessionStatus.ACTIVE),
        get_state=_get_state,
        finalize_only_if_expired=True,
    )

    assert result is None  # skipped, not finalized
    assert get_state_calls == []  # short-circuited before any finalize work


@pytest.mark.asyncio
async def test_end_session_unlocked_finalizes_unconditionally_by_default() -> None:
    # The default (explicit/shutdown callers) must finalize even a non-expired
    # session — the renewal guard is sweeper-only. We assert it proceeds past the
    # renewal check into finalize work (reaches get_state), confirming the guard
    # does not block unconditional callers.
    manager = _FakeActiveSessionManager()
    manager.persisted_sessions = {"thread-1": _active_session("thread-1")}
    manager.expired_threads = set()  # not expired, but default must still finalize
    service = _build_service(active_session_manager=manager)

    get_state_calls: list[str] = []

    async def _get_state(thread_id: str) -> object | None:
        get_state_calls.append(thread_id)
        return None  # triggers the early delete-and-return path; finalize attempted

    await service.end_session_unlocked(
        "thread-1",
        effective_llm_client=lambda thread_id, override: None,
        session_status_unlocked=lambda thread_id: _coro(SessionStatus.ACTIVE),
        get_state=_get_state,
        # finalize_only_if_expired defaults to False
    )

    assert get_state_calls == ["thread-1"]  # did NOT skip; proceeded to finalize


async def _coro(value: object) -> object:
    return value
