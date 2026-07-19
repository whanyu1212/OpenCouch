"""Tests for read-side active-session status classification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.memory.modes import MemoryMode
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.runtime.session import RuntimeSessionTracker
from agent.runtime.session.active_session import (
    PersistedActiveSessionRow,
    PersistedActiveSessionState,
)
from agent.runtime.thread_state_reader import ThreadStateReader
from agent.runtime.types import SessionStatus


class _FakeStateStore:
    async def load_state(self, thread_id: str) -> None:
        del thread_id
        return None


class _FakeActiveSessionManager:
    def __init__(self, session: PersistedActiveSessionState) -> None:
        self._row = PersistedActiveSessionRow(
            payload_json=session.to_json(),
            mutation_token=None,
            mutation_kind=None,
            rotate_after_this_turn=False,
            finalize_required_reason=None,
        )

    async def load_persisted_active_session_row(
        self,
        thread_id: str,
    ) -> PersistedActiveSessionRow:
        del thread_id
        return self._row

    def is_mutation_in_flight(self, token: str | None) -> bool:
        del token
        return False


def _session_timestamp(*, age: timedelta) -> str:
    return (datetime.now(timezone.utc) - age).isoformat()


def _reader(last_active_at: str) -> ThreadStateReader:
    session = PersistedActiveSessionState(
        thread_id="thread-1",
        started_at="2026-07-17T00:00:00Z",
        last_active_at=last_active_at,
        transcript_start_index=0,
        max_crisis_level=0,
        session_buffer=SessionMemoryBuffer(session_id="thread-1"),
    )
    return ThreadStateReader(
        state_store=_FakeStateStore(),
        text_session_store=None,
        active_session_manager=_FakeActiveSessionManager(session),
        session_tracker=RuntimeSessionTracker(),
        memory_mode=MemoryMode.LOCAL,
        session_timeout=timedelta(minutes=30),
    )


@pytest.mark.asyncio
async def test_session_status_classifies_expired_session() -> None:
    reader = _reader(_session_timestamp(age=timedelta(minutes=31)))

    assert (
        await reader.session_status_unlocked("thread-1")
        == SessionStatus.EXPIRED_UNFINALIZED
    )


@pytest.mark.asyncio
async def test_session_status_classifies_fresh_session() -> None:
    reader = _reader(_session_timestamp(age=timedelta(minutes=29)))

    assert await reader.session_status_unlocked("thread-1") == SessionStatus.ACTIVE
