"""Tests for runtime-local active-session tracking helpers."""

from __future__ import annotations

from agent.runtime.active_session import PersistedActiveSessionState
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.runtime.session import RuntimeSessionTracker


def test_tracker_starts_records_and_persists_session_state() -> None:
    tracker = RuntimeSessionTracker()

    assert not tracker.has_tracking("thread-1")

    tracker.start_session(
        "thread-1",
        started_at="2026-05-05T00:00:00Z",
        transcript_start_index=3,
    )
    tracker.record_crisis_level("thread-1", 1)
    tracker.record_crisis_level("thread-1", 0)
    tracker.session_memory_buffer_for_thread("thread-1").record_approach("cbt")

    persisted = tracker.to_persisted_session(
        "thread-1",
        last_active_at="2026-05-05T00:01:00Z",
    )

    assert persisted is not None
    assert persisted.thread_id == "thread-1"
    assert persisted.started_at == "2026-05-05T00:00:00Z"
    assert persisted.last_active_at == "2026-05-05T00:01:00Z"
    assert persisted.transcript_start_index == 3
    assert persisted.max_crisis_level == 1
    assert persisted.session_buffer.approach_counts == {"cbt": 1}


def test_tracker_hydrates_with_deep_copied_session_buffer() -> None:
    source_buffer = SessionMemoryBuffer(
        session_id="thread-1",
        approach_counts={"mindfulness": 2},
    )
    session = PersistedActiveSessionState(
        thread_id="thread-1",
        started_at="2026-05-05T00:00:00Z",
        last_active_at="2026-05-05T00:01:00Z",
        transcript_start_index=4,
        max_crisis_level=2,
        session_buffer=source_buffer,
    )
    tracker = RuntimeSessionTracker()

    tracker.hydrate(session)
    source_buffer.record_approach("cbt")

    assert tracker.has_tracking("thread-1")
    assert tracker.thread_ids() == ["thread-1"]
    assert tracker.started_at("thread-1", default="fallback") == session.started_at
    assert tracker.max_crisis_level("thread-1") == 2
    assert tracker.transcript_start_index("thread-1") == 4
    assert tracker.session_memory_buffer_for_thread("thread-1").approach_counts == {
        "mindfulness": 2
    }


def test_tracker_clear_removes_session_tracking() -> None:
    tracker = RuntimeSessionTracker()
    tracker.start_session(
        "thread-1",
        started_at="2026-05-05T00:00:00Z",
        transcript_start_index=0,
    )

    tracker.clear("thread-1")

    assert not tracker.has_tracking("thread-1")
    assert (
        tracker.to_persisted_session(
            "thread-1",
            last_active_at="2026-05-05T00:01:00Z",
        )
        is None
    )
