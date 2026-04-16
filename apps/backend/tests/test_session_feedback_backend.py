"""Tests for the v0.10 session-feedback backends.

Covers :class:`InMemorySessionFeedbackBackend`,
:class:`NullSessionFeedbackBackend`, and
:class:`SqliteSessionFeedbackBackend`. Structurally parallel to the
crisis_log backend tests — the two subsystems share the same "append
+ query + count + purge + close" shape and the tests should read
symmetrically.

What these tests assert:
- Protocol conformance for all three implementations
- Append / list_by_session / record_count contracts
- Retention-purge boundary semantics (cutoff is exclusive)
- Close-safety: aclose is idempotent; post-close access raises for
  stateful backends but returns safe defaults for Null
- SQL ``CHECK`` constraint enforcement on ``label`` and ``source``
- Phase 1 idempotency contract: duplicate ``id`` produces duplicate
  rows (the SQLite schema intentionally has NO ``UNIQUE`` on ``id``)
- Records survive across a SQLite-file reopen (real persistence)
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from agent.memory.models import SessionFeedbackRecord
from agent.memory.session_feedback import (
    InMemorySessionFeedbackBackend,
    NullSessionFeedbackBackend,
    SessionFeedbackBackend,
)
from agent.memory.sqlite_session_feedback import SqliteSessionFeedbackBackend


# ─── Test helpers ────────────────────────────────────────────────────


def _record(
    *,
    id_suffix: str = "000000000001",
    session: str = "abc",
    label: str = "positive",
    source: str = "cli_end",
    recorded_at: str = "2026-04-16T10:00:00Z",
    user_id: str | None = None,
    turn_count: int = 3,
) -> SessionFeedbackRecord:
    """Produce a valid SessionFeedbackRecord for testing."""
    return SessionFeedbackRecord(
        id=f"00000000-0000-4000-8000-{id_suffix}",
        session_id_opaque=session,
        user_id_or_null=user_id,
        recorded_at=recorded_at,
        label=label,  # type: ignore[arg-type]
        turn_count_at_end=turn_count,
        source=source,  # type: ignore[arg-type]
    )


# ─── Protocol conformance ────────────────────────────────────────────


def test_inmemory_backend_satisfies_protocol() -> None:
    """The in-memory backend must satisfy the
    :class:`SessionFeedbackBackend` protocol."""
    backend: SessionFeedbackBackend = InMemorySessionFeedbackBackend()
    assert backend is not None


def test_null_backend_satisfies_protocol() -> None:
    """The null backend must also satisfy the protocol — otherwise
    test fixtures asserting "no feedback written" would break."""
    backend: SessionFeedbackBackend = NullSessionFeedbackBackend()
    assert backend is not None


def test_sqlite_backend_satisfies_protocol() -> None:
    """The SQLite backend's module-level assertion already runs at
    import, but assert here too for explicit coverage."""
    backend: SessionFeedbackBackend = SqliteSessionFeedbackBackend(":memory:")
    assert backend is not None


# ─── InMemoryBackend ─────────────────────────────────────────────────


class TestInMemoryBackend:
    @pytest.mark.asyncio
    async def test_append_and_list_by_session(self) -> None:
        backend = InMemorySessionFeedbackBackend()
        r = _record(session="abc")
        await backend.aappend(r)

        records = await backend.alist_by_session("abc")
        assert len(records) == 1
        assert records[0].id == r.id

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
        await backend.aclose()  # second close must not raise

    @pytest.mark.asyncio
    async def test_apurge_before_is_exclusive(self) -> None:
        """Records recorded on the cutoff date itself must survive —
        matches the crisis_log purge boundary contract."""
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

        # Purge before 2026-04-15 → drops only the 04-14 record.
        deleted = await backend.apurge_before(date(2026, 4, 15))
        assert deleted == 1
        assert await backend.arecord_count() == 2

    @pytest.mark.asyncio
    async def test_apurge_before_empties_session_bucket(self) -> None:
        """If purge drops every record for a session, the session
        bucket should disappear — not a bug, just a cleanup detail."""
        backend = InMemorySessionFeedbackBackend()
        await backend.aappend(
            _record(session="one-day-only", recorded_at="2026-04-10T10:00:00Z")
        )
        await backend.apurge_before(date(2026, 4, 16))
        assert await backend.alist_by_session("one-day-only") == []


# ─── NullBackend ─────────────────────────────────────────────────────


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
        # Null backend survives repeated close and continues to
        # accept calls — "null" means "unconditionally safe".
        await backend.aclose()
        await backend.aappend(_record())


# ─── SqliteBackend ───────────────────────────────────────────────────


class TestSqliteBackend:
    @pytest.mark.asyncio
    async def test_schema_is_idempotent_across_opens(self, tmp_path: Path) -> None:
        """Running the DDL twice (across two backend instances
        pointing at the same file) must not raise."""
        db = tmp_path / "feedback.sqlite3"
        first = SqliteSessionFeedbackBackend(db)
        await first.aappend(_record())
        await first.aclose()

        second = SqliteSessionFeedbackBackend(db)
        records = await second.alist_by_session("abc")
        assert len(records) == 1
        await second.aclose()

    @pytest.mark.asyncio
    async def test_in_memory_database_isolates_instances(self) -> None:
        """Each SqliteSessionFeedbackBackend with ``:memory:`` gets
        its own private DB — no cross-talk between tests."""
        a = SqliteSessionFeedbackBackend(":memory:")
        b = SqliteSessionFeedbackBackend(":memory:")
        await a.aappend(_record(id_suffix="000000000001"))
        assert await b.arecord_count() == 0
        await a.aclose()
        await b.aclose()

    @pytest.mark.asyncio
    async def test_records_round_trip_through_json_value(self) -> None:
        """The full record shape survives the JSON-blob round trip,
        including ``user_id_or_null`` and ``schema_version``."""
        backend = SqliteSessionFeedbackBackend(":memory:")
        original = _record(user_id="alice", turn_count=7)
        await backend.aappend(original)

        records = await backend.alist_by_session("abc")
        assert records == [original]

    @pytest.mark.asyncio
    async def test_apurge_before_uses_exclusive_boundary(self) -> None:
        """SQLite purge boundary must match the in-memory
        implementation — records on ``cutoff`` survive."""
        backend = SqliteSessionFeedbackBackend(":memory:")
        await backend.aappend(
            _record(id_suffix="000000000001", recorded_at="2026-04-14T10:00:00Z")
        )
        await backend.aappend(
            _record(id_suffix="000000000002", recorded_at="2026-04-15T10:00:00Z")
        )
        await backend.aappend(
            _record(id_suffix="000000000003", recorded_at="2026-04-16T10:00:00Z")
        )

        deleted = await backend.apurge_before(date(2026, 4, 15))
        assert deleted == 1
        assert await backend.arecord_count() == 2
        await backend.aclose()

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self) -> None:
        backend = SqliteSessionFeedbackBackend(":memory:")
        await backend.aclose()
        await backend.aclose()  # second close must not raise

    @pytest.mark.asyncio
    async def test_record_count_returns_zero_after_close(self) -> None:
        backend = SqliteSessionFeedbackBackend(":memory:")
        await backend.aclose()
        # Matches crisis_log contract — CLI callers don't need
        # defensive try/except around a closed backend.
        assert await backend.arecord_count() == 0
        assert await backend.apurge_before(date(2099, 1, 1)) == 0

    @pytest.mark.asyncio
    async def test_duplicate_id_appends_produce_two_rows(self) -> None:
        """Phase 1 does not claim idempotency. The SQLite schema
        intentionally has NO UNIQUE constraint on ``id``, so two
        appends with the same UUID result in two rows. Explicit
        idempotency will come later via an explicit idempotency
        key, not a retrofitted UNIQUE on the opaque ``id``."""
        backend = SqliteSessionFeedbackBackend(":memory:")
        r = _record()
        await backend.aappend(r)
        await backend.aappend(r)  # same id intentionally

        assert await backend.arecord_count() == 2
        records = await backend.alist_by_session("abc")
        assert len(records) == 2
        # Both rows are the same semantic record, just stored twice.
        assert records[0].id == records[1].id
        await backend.aclose()

    @pytest.mark.asyncio
    async def test_check_constraint_rejects_invalid_label(self, tmp_path: Path) -> None:
        """Defense in depth: the SQL ``CHECK (label IN (...))``
        constraint catches any drift between the Python enum and the
        DB schema. Pydantic should reject first; this test reaches
        past Pydantic to prove the DB guard fires too.

        Uses a raw sqlite3 connection (not the aiosqlite backend)
        to bypass Pydantic and exercise the CHECK constraint
        directly at the SQL level.
        """
        db = tmp_path / "check.sqlite3"
        # First, let the backend create the schema so we have a
        # table to test against.
        backend = SqliteSessionFeedbackBackend(db)
        await backend.aappend(_record())
        await backend.aclose()

        # Then insert a bad label directly via sqlite3 to prove the
        # CHECK constraint rejects it.
        conn = sqlite3.connect(db)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO session_feedback
                        (id, session_id_opaque, recorded_at, recorded_date,
                         label, turn_count_at_end, source, schema_version, value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "bogus-id",
                        "bogus-session",
                        "2026-04-16T10:00:00Z",
                        "2026-04-16",
                        "ecstatic",  # ← invalid label
                        0,
                        "cli_end",
                        1,
                        "{}",
                    ),
                )
                conn.commit()
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_check_constraint_rejects_invalid_source(
        self, tmp_path: Path
    ) -> None:
        """Same defense-in-depth test, now for ``source``."""
        db = tmp_path / "check_source.sqlite3"
        backend = SqliteSessionFeedbackBackend(db)
        await backend.aappend(_record())
        await backend.aclose()

        conn = sqlite3.connect(db)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO session_feedback
                        (id, session_id_opaque, recorded_at, recorded_date,
                         label, turn_count_at_end, source, schema_version, value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "bogus-id",
                        "bogus-session",
                        "2026-04-16T10:00:00Z",
                        "2026-04-16",
                        "positive",
                        0,
                        "voice_disconnect",  # ← not in the Phase 1 enum
                        1,
                        "{}",
                    ),
                )
                conn.commit()
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_malformed_recorded_at_raises_valueerror(
        self, tmp_path: Path
    ) -> None:
        """The backend's ``_extract_date_prefix`` helper validates the
        ISO-8601 date portion at insert time. A malformed
        ``recorded_at`` should fail loudly rather than silently
        landing the record in a bad date bucket (and breaking
        subsequent purge queries)."""
        backend = SqliteSessionFeedbackBackend(":memory:")
        # Pydantic doesn't validate ISO format on the string — it just
        # checks type. So we can construct a record with a malformed
        # recorded_at and the backend should reject it on insert.
        bad = _record(recorded_at="not-an-iso-date")
        with pytest.raises(ValueError):
            await backend.aappend(bad)
