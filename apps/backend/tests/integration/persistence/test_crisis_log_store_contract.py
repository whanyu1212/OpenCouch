"""Persistence contracts for the supported durable crisis-log backend."""

from __future__ import annotations

import asyncio
from datetime import date
from uuid import uuid4

import psycopg
import pytest

from agent.audit.models import CrisisLogRecord
from tests.support.persistence_contracts import open_postgres_crisis_log_backend

pytestmark = pytest.mark.asyncio


def _record(
    *,
    record_id: str | None = None,
    detected_at: str = "2099-04-10T10:00:00Z",
    level: int = 2,
    reason: str = "contract test",
) -> CrisisLogRecord:
    """Build a complete crisis record for durable-store contracts."""

    return CrisisLogRecord(
        id=record_id or f"crisis-{uuid4()}",
        event_type="crisis_response",
        session_id_opaque="a" * 64,
        user_id_or_null="user-contract",
        detected_at=detected_at,
        level=level,  # type: ignore[arg-type]
        override_kind="none",
        classifier_path="llm_primary",
        reason=reason,
        response_node_completed=True,
        llm_failure_occurred=False,
        response_path="sdk_tool_fallback",
        response_style="crisis_response",
        resource_lookup_status="no_verified_results",
        resource_count=2,
        tool_calls=["lookup_crisis_resources"],
        fallback_reason="crisis_resource_tool_not_called",
        trace_id="trace-1",
        trace_session_id="trace-session-1",
        trace_turn_id="turn-1",
        trace_runtime_mode="voice",
    )


async def test_crisis_log_round_trip_preserves_complete_record() -> None:
    """JSONB storage should preserve every field without changing values."""

    record = _record(reason="caf\u00e9 | \u65e5\u672c\u8a9e | 0.30000000000000004")

    async with open_postgres_crisis_log_backend() as backend:
        await backend.aappend(record)
        assert await backend.alist_by_date(date(2099, 4, 10)) == [record]


async def test_crisis_log_groups_dates_and_counts_all_records() -> None:
    """Date reads should be isolated while count spans every date bucket."""

    first = _record(detected_at="2099-04-10T23:59:59Z")
    second = _record(detected_at="2099-04-11T00:00:00Z")

    async with open_postgres_crisis_log_backend() as backend:
        assert await backend.alist_by_date(date(2099, 4, 9)) == []
        assert await backend.arecord_count() == 0
        await backend.aappend(first)
        await backend.aappend(second)

        assert await backend.alist_by_date(date(2099, 4, 10)) == [first]
        assert await backend.alist_by_date(date(2099, 4, 11)) == [second]
        assert await backend.arecord_count() == 2


async def test_crisis_log_persists_order_across_close_and_reopen() -> None:
    """Records and append ordering should survive a new backend connection."""

    records = [_record(detected_at=f"2099-04-12T10:0{index}:00Z") for index in range(3)]

    async with open_postgres_crisis_log_backend() as backend:
        for record in records:
            await backend.aappend(record)

    async with open_postgres_crisis_log_backend() as backend:
        assert await backend.alist_by_date(date(2099, 4, 12)) == records
        assert await backend.arecord_count() == 3


async def test_crisis_log_writes_and_purges_are_visible_across_connections() -> None:
    """Committed inserts and deletes should be visible to another live backend."""

    record = _record(detected_at="2099-04-13T10:00:00Z")

    async with open_postgres_crisis_log_backend() as writer:
        async with open_postgres_crisis_log_backend() as reader:
            await writer.aappend(record)
            assert await reader.alist_by_date(date(2099, 4, 13)) == [record]

            assert await writer.apurge_before(date(2099, 4, 14)) == 1
            assert await reader.alist_by_date(date(2099, 4, 13)) == []


async def test_crisis_log_purge_is_exclusive_idempotent_and_durable() -> None:
    """Purge should delete only older rows and persist that deletion."""

    old = _record(detected_at="2099-04-14T10:00:00Z")
    cutoff = _record(detected_at="2099-04-15T10:00:00Z")

    async with open_postgres_crisis_log_backend() as backend:
        assert await backend.apurge_before(date(2099, 4, 15)) == 0
        await backend.aappend(old)
        await backend.aappend(cutoff)
        assert await backend.apurge_before(date(2099, 4, 15)) == 1
        assert await backend.apurge_before(date(2099, 4, 15)) == 0

    async with open_postgres_crisis_log_backend() as backend:
        assert await backend.alist_by_date(date(2099, 4, 14)) == []
        assert await backend.alist_by_date(date(2099, 4, 15)) == [cutoff]
        assert await backend.arecord_count() == 1


async def test_crisis_log_close_contract() -> None:
    """Close should be idempotent and block stateful operations."""

    async with open_postgres_crisis_log_backend() as backend:
        await backend.aappend(_record())
        await backend.aclose()
        await backend.aclose()

        assert await backend.arecord_count() == 0
        assert await backend.apurge_before(date(2099, 4, 11)) == 0
        with pytest.raises(RuntimeError, match="closed"):
            await backend.aappend(_record())
        with pytest.raises(RuntimeError, match="closed"):
            await backend.alist_by_date(date(2099, 4, 10))


async def test_crisis_log_rejects_duplicate_ids() -> None:
    """Crisis record IDs are unique and duplicate appends are caller errors."""

    record = _record(record_id=f"duplicate-{uuid4()}")

    async with open_postgres_crisis_log_backend() as backend:
        await backend.aappend(record)
        with pytest.raises(psycopg.errors.UniqueViolation):
            await backend.aappend(record)


async def test_crisis_log_rejects_malformed_timestamp() -> None:
    """Malformed timestamps must fail instead of entering an invalid date bucket."""

    async with open_postgres_crisis_log_backend() as backend:
        with pytest.raises(ValueError):
            await backend.aappend(_record(detected_at="not-a-timestamp"))


async def test_crisis_log_concurrent_appends_all_persist() -> None:
    """Concurrent callers should not lose records on the shared connection."""

    records = [_record(detected_at="2099-04-16T10:00:00Z") for _ in range(10)]

    async with open_postgres_crisis_log_backend() as backend:
        await asyncio.gather(*(backend.aappend(record) for record in records))
        stored = await backend.alist_by_date(date(2099, 4, 16))

        assert {record.id for record in stored} == {record.id for record in records}
        assert await backend.arecord_count() == len(records)
