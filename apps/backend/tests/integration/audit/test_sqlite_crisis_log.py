"""Unit tests for the v0.8 SqliteCrisisLogBackend.

Parallels the ``InMemoryCrisisLogBackend`` tests in
``test_memory_store.py`` (the CrisisLog tests section), so both
backends have symmetric coverage. Any test that passes for the
in-memory version should have an equivalent here — both backends
satisfy the same protocol and should behave identically except for
persistence (where the SQLite version's whole point is that it
survives close/reopen).

Most tests use ``:memory:`` SQLite databases for speed and isolation.
The persistence-across-restart tests use ``tmp_path`` fixtures for
real files on disk.

Test structure:
    1. Round-trip tests (append / list_by_date / arecord_count)
    2. Date-bucket isolation
    3. Close lifecycle
    4. Persistence across close/reopen (the core v0.8 contract)
    5. Schema constraints (CHECK, UNIQUE)
    6. Protocol conformance
"""

from __future__ import annotations

from datetime import date

import pytest

from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.models import CrisisLogRecord
from agent.audit.sqlite_crisis_log import SqliteCrisisLogBackend


# ─── Test helpers ──────────────────────────────────────────────────────


def _crisis_record(
    *,
    record_id: str = "rec-1",
    detected_at: str = "2026-04-10T12:00:00Z",
    level: int = 2,
    user_id: str | None = None,
    session_id_opaque: str | None = None,
) -> CrisisLogRecord:
    """Build a valid CrisisLogRecord for tests.

    Defaults match the in-memory backend's test helper in
    ``test_memory_store.py`` so the two test suites exercise the
    same record shape.
    """

    return CrisisLogRecord(
        id=record_id,
        session_id_opaque=session_id_opaque or "a" * 64,
        user_id_or_null=user_id,
        detected_at=detected_at,
        level=level,  # type: ignore[arg-type]
        override_kind="none",
        classifier_path="llm_primary",
        reason="test",
        response_node_completed=True,
        llm_failure_occurred=False,
        response_path="sdk_tool_fallback",
        response_style="crisis_response",
        resource_lookup_status="no_verified_results",
        resource_count=2,
        tool_calls=["lookup_crisis_resources"],
        fallback_reason="crisis_resource_tool_not_called",
    )


# ─── Round-trip tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_append_and_list_by_date_round_trip() -> None:
    """An appended record should be retrievable via list_by_date."""

    backend = SqliteCrisisLogBackend(":memory:")
    record = _crisis_record(detected_at="2026-04-10T12:00:00Z")
    await backend.aappend(record)

    results = await backend.alist_by_date(date(2026, 4, 10))
    assert len(results) == 1
    assert results[0].id == "rec-1"
    assert results[0].level == 2
    assert results[0].session_id_opaque == "a" * 64
    await backend.aclose()


@pytest.mark.asyncio
async def test_list_by_date_returns_empty_for_unknown_day() -> None:
    """Asking for a day with no records should return an empty list."""

    backend = SqliteCrisisLogBackend(":memory:")
    results = await backend.alist_by_date(date(2026, 4, 10))
    assert results == []
    await backend.aclose()


@pytest.mark.asyncio
async def test_multiple_records_same_day_returned_in_insertion_order() -> None:
    """Records appended on the same day should come back in insertion
    order — that's the protocol contract for ``alist_by_date``."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aappend(
        _crisis_record(record_id="first", detected_at="2026-04-10T09:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="second", detected_at="2026-04-10T12:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="third", detected_at="2026-04-10T23:00:00Z")
    )

    results = await backend.alist_by_date(date(2026, 4, 10))
    assert [r.id for r in results] == ["first", "second", "third"]
    await backend.aclose()


@pytest.mark.asyncio
async def test_records_grouped_by_date() -> None:
    """Records from different days should land in different date
    buckets. Matches the in-memory backend's test_crisis_log_groups_by_date."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aappend(
        _crisis_record(record_id="a", detected_at="2026-04-10T11:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="b", detected_at="2026-04-10T23:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="c", detected_at="2026-04-11T00:00:00Z")
    )

    day_10 = await backend.alist_by_date(date(2026, 4, 10))
    day_11 = await backend.alist_by_date(date(2026, 4, 11))

    assert [r.id for r in day_10] == ["a", "b"]
    assert [r.id for r in day_11] == ["c"]
    await backend.aclose()


@pytest.mark.asyncio
async def test_arecord_count_reports_total_across_dates() -> None:
    """arecord_count should report the total regardless of date."""

    backend = SqliteCrisisLogBackend(":memory:")
    assert await backend.arecord_count() == 0

    await backend.aappend(
        _crisis_record(record_id="a", detected_at="2026-04-10T11:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="b", detected_at="2026-04-11T11:00:00Z")
    )

    assert await backend.arecord_count() == 2
    await backend.aclose()


@pytest.mark.asyncio
async def test_records_preserve_all_fields_round_trip() -> None:
    """Every field on CrisisLogRecord should survive serialization +
    deserialization. Guards against silent data loss when new fields
    are added to the pydantic model."""

    backend = SqliteCrisisLogBackend(":memory:")
    original = _crisis_record(
        record_id="detailed",
        detected_at="2026-04-10T14:30:45Z",
        level=3,
        user_id="user-42",
        session_id_opaque="b" * 64,
    )
    await backend.aappend(original)

    results = await backend.alist_by_date(date(2026, 4, 10))
    assert len(results) == 1
    restored = results[0]
    assert restored.id == "detailed"
    assert restored.detected_at == "2026-04-10T14:30:45Z"
    assert restored.level == 3
    assert restored.user_id_or_null == "user-42"
    assert restored.session_id_opaque == "b" * 64
    assert restored.override_kind == "none"
    assert restored.classifier_path == "llm_primary"
    assert restored.reason == "test"
    assert restored.response_node_completed is True
    assert restored.llm_failure_occurred is False
    assert restored.event_type == "crisis_response"
    assert restored.response_path == "sdk_tool_fallback"
    assert restored.response_style == "crisis_response"
    assert restored.resource_lookup_status == "no_verified_results"
    assert restored.resource_count == 2
    assert restored.tool_calls == ["lookup_crisis_resources"]
    assert restored.fallback_reason == "crisis_resource_tool_not_called"
    await backend.aclose()


# ─── Close lifecycle ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_blocks_further_use() -> None:
    """After aclose, append and list should raise RuntimeError."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aclose()

    with pytest.raises(RuntimeError):
        await backend.aappend(_crisis_record())
    with pytest.raises(RuntimeError):
        await backend.alist_by_date(date(2026, 4, 10))


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    """Calling aclose on an already-closed backend should not raise."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aclose()
    await backend.aclose()  # must not raise


@pytest.mark.asyncio
async def test_arecord_count_returns_zero_when_closed() -> None:
    """After close, arecord_count should return 0 rather than raising.
    This matches the memory store's close-safety contract so CLI
    code can call it without defensive try/except."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aappend(_crisis_record())
    await backend.aclose()

    assert await backend.arecord_count() == 0


# ─── Persistence across close/reopen (the v0.8 core feature) ──────────


@pytest.mark.asyncio
async def test_persists_across_close_and_reopen(tmp_path) -> None:
    """The core v0.8 contract: records written to a file-backed crisis
    log must survive close + reopen. Without this, the entire v0.8
    refactor for the crisis log is worthless."""

    db_path = tmp_path / "test_crisis_persistence.sqlite3"

    # First runtime lifetime: write three records, close
    backend_a = SqliteCrisisLogBackend(db_path)
    await backend_a.aappend(
        _crisis_record(record_id="rec-1", detected_at="2026-04-10T09:00:00Z", level=1)
    )
    await backend_a.aappend(
        _crisis_record(record_id="rec-2", detected_at="2026-04-10T14:00:00Z", level=2)
    )
    await backend_a.aappend(
        _crisis_record(record_id="rec-3", detected_at="2026-04-11T08:00:00Z", level=3)
    )
    assert await backend_a.arecord_count() == 3
    await backend_a.aclose()

    # Second runtime lifetime: reopen, verify records come back
    backend_b = SqliteCrisisLogBackend(db_path)
    assert await backend_b.arecord_count() == 3

    day_10 = await backend_b.alist_by_date(date(2026, 4, 10))
    assert [r.id for r in day_10] == ["rec-1", "rec-2"]
    assert [r.level for r in day_10] == [1, 2]

    day_11 = await backend_b.alist_by_date(date(2026, 4, 11))
    assert len(day_11) == 1
    assert day_11[0].id == "rec-3"
    assert day_11[0].level == 3

    await backend_b.aclose()


@pytest.mark.asyncio
async def test_insertion_order_survives_reopen(tmp_path) -> None:
    """Insertion order within a date bucket should be preserved across
    close + reopen — this depends on the explicit insertion_order
    column in the schema."""

    db_path = tmp_path / "test_insertion_persistence.sqlite3"

    backend_a = SqliteCrisisLogBackend(db_path)
    for i, rid in enumerate(["first", "second", "third", "fourth"]):
        await backend_a.aappend(
            _crisis_record(
                record_id=rid,
                detected_at=f"2026-04-10T{10 + i:02d}:00:00Z",
            )
        )
    await backend_a.aclose()

    backend_b = SqliteCrisisLogBackend(db_path)
    results = await backend_b.alist_by_date(date(2026, 4, 10))
    assert [r.id for r in results] == ["first", "second", "third", "fourth"]
    await backend_b.aclose()


# ─── Schema constraints ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_id_raises() -> None:
    """The single-column UNIQUE constraint on ``id`` should reject
    duplicate writes. This is stricter than the in-memory backend
    (which silently appends), matching the protocol's note that
    duplicate ids are a caller bug."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aappend(_crisis_record(record_id="duplicate-id"))

    # A second write with the same id should fail at the SQLite
    # constraint layer. The specific exception type depends on
    # aiosqlite's wrapping, so we accept any database error.
    with pytest.raises(Exception):  # noqa: B017
        await backend.aappend(_crisis_record(record_id="duplicate-id"))
    await backend.aclose()


@pytest.mark.asyncio
async def test_invalid_level_rejected_by_check_constraint() -> None:
    """The CHECK constraint on ``level`` should reject values outside
    the allowed (0, 1, 2, 3) range. In practice, the pydantic model
    rejects these first — but the schema-level check is
    defense-in-depth against direct SQL writes or future model
    drift. Test the pydantic validation path: level=5 should fail
    at CrisisLogRecord construction, not at the SQL layer."""

    # Pydantic validation catches this before we even reach the
    # backend. Include the test anyway because it pins the
    # record-construction contract.
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        _crisis_record(level=5)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_malformed_detected_at_raises_at_append() -> None:
    """A record with a non-parseable ``detected_at`` should fail at
    ``aappend`` time via the date-prefix extraction helper. The
    in-memory backend has the same failure mode but via
    ``date.fromisoformat`` on the string directly."""

    backend = SqliteCrisisLogBackend(":memory:")
    # Pydantic accepts any string in detected_at — it's a str field,
    # not a datetime. So we can construct a bad record and let the
    # backend's date-prefix validator catch it.
    bad_record = _crisis_record(detected_at="not-a-real-timestamp")
    with pytest.raises(ValueError):
        await backend.aappend(bad_record)
    await backend.aclose()


# ─── Protocol conformance ──────────────────────────────────────────────


def test_satisfies_crisis_log_backend_protocol() -> None:
    """The class must satisfy the CrisisLogBackend protocol. The
    import-time assertion in sqlite_crisis_log.py would fail if this
    weren't true; this test is belt-and-suspenders."""

    backend = SqliteCrisisLogBackend(":memory:")
    # Protocol check via isinstance works because CrisisLogBackend
    # is a plain Protocol (not @runtime_checkable) — but we can
    # still assert attribute presence as a structural check.
    assert hasattr(backend, "aappend")
    assert hasattr(backend, "alist_by_date")
    assert hasattr(backend, "arecord_count")
    assert hasattr(backend, "apurge_before")
    assert hasattr(backend, "aclose")
    # And the module-level type assertion in sqlite_crisis_log.py
    # serves the same purpose at import time.
    _: type[CrisisLogBackend] = SqliteCrisisLogBackend


# ─── v0.8.1 retention purge ────────────────────────────────────────────
#
# Symmetric tests for ``InMemoryCrisisLogBackend.apurge_before`` live in
# ``test_crisis_log.py::TestInMemoryCrisisLogRetentionPurge``. The two
# suites must behave identically — both backends satisfy the same
# protocol and the CLI command path is polymorphic across them.


@pytest.mark.asyncio
async def test_purge_before_deletes_only_records_before_cutoff() -> None:
    """Records strictly older than the cutoff are deleted; cutoff-day
    records are preserved. Exclusive boundary."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aappend(
        _crisis_record(record_id="old", detected_at="2025-12-01T10:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="on_cutoff", detected_at="2026-04-01T10:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="recent", detected_at="2026-04-12T10:00:00Z")
    )

    deleted = await backend.apurge_before(date(2026, 4, 1))
    assert deleted == 1
    assert await backend.arecord_count() == 2

    on_cutoff_bucket = await backend.alist_by_date(date(2026, 4, 1))
    assert len(on_cutoff_bucket) == 1
    assert on_cutoff_bucket[0].id == "on_cutoff"

    recent_bucket = await backend.alist_by_date(date(2026, 4, 12))
    assert len(recent_bucket) == 1
    assert recent_bucket[0].id == "recent"

    old_bucket = await backend.alist_by_date(date(2025, 12, 1))
    assert len(old_bucket) == 0

    await backend.aclose()


@pytest.mark.asyncio
async def test_purge_before_is_idempotent() -> None:
    """Running the same purge twice is a safe no-op on the second call."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aappend(
        _crisis_record(record_id="old", detected_at="2025-12-01T10:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="recent", detected_at="2026-04-12T10:00:00Z")
    )

    first = await backend.apurge_before(date(2026, 4, 1))
    second = await backend.apurge_before(date(2026, 4, 1))
    assert first == 1
    assert second == 0
    assert await backend.arecord_count() == 1
    await backend.aclose()


@pytest.mark.asyncio
async def test_purge_before_on_empty_backend_returns_zero() -> None:
    """Purging an empty backend returns 0 without raising."""

    backend = SqliteCrisisLogBackend(":memory:")
    deleted = await backend.apurge_before(date(2026, 4, 1))
    assert deleted == 0
    await backend.aclose()


@pytest.mark.asyncio
async def test_purge_before_on_closed_backend_returns_zero() -> None:
    """Closed backends return 0 without reopening the connection —
    matches the closed-safe contract of arecord_count."""

    backend = SqliteCrisisLogBackend(":memory:")
    await backend.aclose()
    deleted = await backend.apurge_before(date(2026, 4, 1))
    assert deleted == 0


@pytest.mark.asyncio
async def test_purge_before_persists_across_reopen(tmp_path) -> None:
    """The purge's effect survives a close/reopen cycle. This is the
    SQLite-specific durability pin — the in-memory backend's test
    suite can't exercise this path."""

    db_path = tmp_path / "purge_test.sqlite3"
    backend = SqliteCrisisLogBackend(str(db_path))
    await backend.aappend(
        _crisis_record(record_id="old", detected_at="2025-12-01T10:00:00Z")
    )
    await backend.aappend(
        _crisis_record(record_id="recent", detected_at="2026-04-12T10:00:00Z")
    )
    deleted = await backend.apurge_before(date(2026, 4, 1))
    assert deleted == 1
    await backend.aclose()

    # Reopen and verify the purge stuck.
    backend2 = SqliteCrisisLogBackend(str(db_path))
    assert await backend2.arecord_count() == 1
    recent_bucket = await backend2.alist_by_date(date(2026, 4, 12))
    assert len(recent_bucket) == 1
    assert recent_bucket[0].id == "recent"
    old_bucket = await backend2.alist_by_date(date(2025, 12, 1))
    assert len(old_bucket) == 0
    await backend2.aclose()
