"""Contracts for non-durable session-feedback backends."""

from __future__ import annotations

from datetime import date

import pytest

from agent.feedback.models import SessionFeedbackRecord
from agent.feedback.session_feedback import (
    InMemorySessionFeedbackBackend,
    NullSessionFeedbackBackend,
    SessionFeedbackBackend,
)


def _record(
    *,
    id_suffix: str = "000000000001",
    session: str = "abc",
    label: str = "positive",
    source: str = "cli_end",
    modality: str = "text",
    recorded_at: str = "2026-04-16T10:00:00Z",
    user_id: str | None = None,
    turn_count: int = 3,
) -> SessionFeedbackRecord:
    """Produce a valid session-feedback record for tests."""

    return SessionFeedbackRecord(
        id=f"00000000-0000-4000-8000-{id_suffix}",
        session_id_opaque=session,
        user_id_or_null=user_id,
        recorded_at=recorded_at,
        label=label,  # type: ignore[arg-type]
        turn_count_at_end=turn_count,
        source=source,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
    )


def test_inmemory_backend_satisfies_protocol() -> None:
    backend: SessionFeedbackBackend = InMemorySessionFeedbackBackend()
    assert backend is not None


def test_null_backend_satisfies_protocol() -> None:
    backend: SessionFeedbackBackend = NullSessionFeedbackBackend()
    assert backend is not None


class TestInMemoryBackend:
    @pytest.mark.asyncio
    async def test_append_and_list_by_session(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        record = _record(session="abc")
        await backend.aappend(record)

        assert await backend.alist_by_session("abc") == [record]

    @pytest.mark.asyncio
    async def test_list_unknown_session_returns_empty(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        assert await backend.alist_by_session("nonexistent") == []

    @pytest.mark.asyncio
    async def test_record_count_aggregates_across_sessions(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        await backend.aappend(_record(id_suffix="000000000001", session="a"))
        await backend.aappend(_record(id_suffix="000000000002", session="a"))
        await backend.aappend(_record(id_suffix="000000000003", session="b"))
        assert await backend.arecord_count() == 3

    @pytest.mark.asyncio
    async def test_aclose_clears_contents_and_blocks_access(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        await backend.aappend(_record())
        await backend.aclose()

        assert await backend.arecord_count() == 0
        with pytest.raises(RuntimeError):
            await backend.aappend(_record())

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        await backend.aclose()
        await backend.aclose()

    @pytest.mark.asyncio
    async def test_apurge_before_is_exclusive(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        await backend.aappend(
            _record(id_suffix="000000000001", recorded_at="2026-04-14T10:00:00Z")
        )
        await backend.aappend(
            _record(id_suffix="000000000002", recorded_at="2026-04-15T10:00:00Z")
        )
        await backend.aappend(
            _record(id_suffix="000000000003", recorded_at="2026-04-16T10:00:00Z")
        )

        assert await backend.apurge_before(date(2026, 4, 15)) == 1
        assert await backend.arecord_count() == 2

    @pytest.mark.asyncio
    async def test_apurge_before_empties_session_bucket(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        await backend.aappend(
            _record(session="one-day-only", recorded_at="2026-04-10T10:00:00Z")
        )
        await backend.apurge_before(date(2026, 4, 16))
        assert await backend.alist_by_session("one-day-only") == []


class TestNullBackend:
    @pytest.mark.asyncio
    async def test_append_is_noop(self) -> None:
        backend = NullSessionFeedbackBackend()
        await backend.aappend(_record())
        assert await backend.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_list_always_empty(self) -> None:
        backend = NullSessionFeedbackBackend()
        await backend.aappend(_record())
        assert await backend.alist_by_session("anything") == []

    @pytest.mark.asyncio
    async def test_purge_always_zero(self) -> None:
        backend = NullSessionFeedbackBackend()
        assert await backend.apurge_before(date(2099, 1, 1)) == 0

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        backend = NullSessionFeedbackBackend()
        await backend.aclose()
        await backend.aclose()
        await backend.aappend(_record())
