"""Tests for the crisis_log_node always-on safety audit trail.

Covers three concerns:
    1. ``run_crisis_log_node`` writes a record when the turn is flagged
       as a crisis (and skips writes otherwise).
    2. The written record stores only an opaque session-id hash.
    3. End-to-end via ``run_agent`` — a crisis turn lands a record in
       the backend when an explicit backend is injected.

These tests exercise the crisis log backend's contract (aappend +
alist_by_date + record_count) without hitting any LLM providers.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator
from datetime import date
from typing import Any, cast

import pytest

from agent.graph import run_agent
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.audit.models import (
    CrisisClassifierPath,
    CrisisLogPathCounts,
    CrisisLogRecord,
)
from agent.gates.safety.service import CrisisAssessmentSchema
from agent.models import AgentInput, CrisisAssessment, ResponseCategory
from agent.nodes.crisis_log import run_crisis_log_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient, StructuredResponseT
from tests.support.persistence import FakeCrossRestartLLM


# ─── Test helpers ────────────────────────────────────────────────────────


def test_crisis_path_counts_cover_all_classifier_paths() -> None:
    """Crisis path aggregates should cover every classifier path literal."""

    counts = CrisisLogPathCounts()
    for path in CrisisClassifierPath.__args__:
        assert hasattr(counts, path)


class _MockRuntime:
    """Minimal runtime stand-in exposing ``.context`` only."""

    def __init__(self, *, crisis_log_backend: InMemoryCrisisLogBackend) -> None:
        self.context = WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=crisis_log_backend,
            memory_mode=MemoryMode.LOCAL,
        )


def _build_crisis_state(
    *,
    level: int = 3,
    needs_crisis_response: bool = True,
    reason: str = "imminent self-harm language detected",
    user_id: str | None = "user-123",
    session_id: str | None = "thread-abc",
    crisis_audit: dict[str, Any] | None = None,
) -> AgentState:
    """Build a minimal AgentState for the crisis log node to read."""

    state: Any = {
        "message": "test",
        "history": [],
        "user_id": user_id,
        "session_id": session_id,
        "crisis": CrisisAssessment(
            level=level,  # type: ignore[arg-type]
            confidence="high",
            reason=reason,
            needs_crisis_response=needs_crisis_response,
            needs_clarification=False,
        ),
        "crisis_audit": crisis_audit or {},
    }
    return cast(AgentState, state)


class _FakeCrisisLLM(BaseLLMClient):
    """Fake LLM client for graph-level crisis log tests."""

    def __init__(
        self,
        *,
        level: int,
        reason: str,
        stream_text: str = "fake crisis response",
    ) -> None:
        self.level = level
        self.reason = reason
        self.stream_text = stream_text

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "unused"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield self.stream_text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__
        if schema_name == "CrisisAssessmentSchema":
            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=self.level,
                    confidence="high",
                    reason=self.reason,
                    needs_crisis_response=self.level >= 2,
                    needs_clarification=self.level == 1,
                ),
            )
        if schema_name == "CrisisLocationDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                status="not_provided",
                location="",
                reasoning="No user location in this test fixture.",
            )
        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


# ─── run_crisis_log_node unit tests ──────────────────────────────────────


class TestCrisisLogNode:
    """Unit tests for the crisis_log_node with mocked runtime + backend."""

    @pytest.mark.asyncio
    async def test_crisis_turn_writes_record(self) -> None:
        """When crisis.needs_crisis_response is True, a record is appended."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=3)

        delta = await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        assert delta == {}  # no state changes; side effect only
        assert await backend.arecord_count() == 1

    @pytest.mark.asyncio
    async def test_record_fields_match_state(self) -> None:
        """The written record should carry the crisis assessment details."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(
            level=2,
            reason="user said 'I want to end it'",
            user_id="user-xyz",
            session_id="thread-xyz",
        )

        await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        today = date.today()  # noqa: DTZ011 - local date for bucketing
        # detected_at is UTC so the bucket might be today or today±1 day.
        # Try today first; if empty, check yesterday.
        records = await backend.alist_by_date(today)
        if not records:
            from datetime import timedelta

            records = await backend.alist_by_date(today - timedelta(days=1))
            if not records:
                records = await backend.alist_by_date(today + timedelta(days=1))

        assert len(records) == 1
        record = records[0]
        assert isinstance(record, CrisisLogRecord)
        assert record.level == 2
        assert record.reason == "user said 'I want to end it'"
        assert record.user_id_or_null == "user-xyz"
        assert record.response_node_completed is True
        assert record.llm_failure_occurred is False

    @pytest.mark.asyncio
    async def test_crisis_audit_metadata_is_written_to_record(self) -> None:
        """The record should reflect the dedicated crisis-audit channel."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(
            crisis_audit={
                "crisis_override_kind": "none",
                "crisis_classifier_path": "llm_primary",
                "crisis_llm_failure_occurred": False,
            }
        )

        await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.override_kind == "none"
        assert record.classifier_path == "llm_primary"
        assert record.llm_failure_occurred is False

    @pytest.mark.asyncio
    async def test_session_id_stored_as_opaque_hash(self) -> None:
        """The session_id should be hashed, not stored in plaintext."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(session_id="very-private-session-id")

        await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        # Fetch the record from whatever bucket it landed in
        from datetime import timedelta

        today = date.today()  # noqa: DTZ011
        all_records = []
        for delta in (-1, 0, 1):
            all_records.extend(
                await backend.alist_by_date(today + timedelta(days=delta))
            )

        assert len(all_records) == 1
        record = all_records[0]
        # The plaintext session_id should NOT appear in the opaque field
        assert "very-private-session-id" not in record.session_id_opaque
        # The opaque field should be the SHA-256 hash of the plaintext
        expected_hash = hashlib.sha256(b"very-private-session-id").hexdigest()
        assert record.session_id_opaque == expected_hash

    @pytest.mark.asyncio
    async def test_incognito_mode_nulls_user_id(self) -> None:
        """When user_id is None (incognito), the record stores null."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(user_id=None, session_id="anonymous-session")

        await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        from datetime import timedelta

        today = date.today()  # noqa: DTZ011
        all_records = []
        for delta in (-1, 0, 1):
            all_records.extend(
                await backend.alist_by_date(today + timedelta(days=delta))
            )

        assert len(all_records) == 1
        assert all_records[0].user_id_or_null is None
        # But session_id_opaque is still populated
        assert all_records[0].session_id_opaque != ""

    @pytest.mark.asyncio
    async def test_non_crisis_turn_is_noop(self) -> None:
        """A non-crisis turn (needs_crisis_response=False) should not write."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=0, needs_crisis_response=False)

        delta = await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        assert delta == {}
        assert await backend.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_backend_failure_is_logged_but_does_not_crash(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A backend write failure should be logged at ERROR level but not raise."""

        class FailingBackend(InMemoryCrisisLogBackend):
            async def aappend(self, record: CrisisLogRecord) -> None:  # type: ignore[override]
                raise RuntimeError("simulated backend failure")

        backend = FailingBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state()

        with caplog.at_level(logging.ERROR, logger="agent.nodes.crisis_log"):
            delta = await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        assert delta == {}  # must not raise
        assert any("audit trail lost" in r.message for r in caplog.records)


# ─── 3. End-to-end crisis log behavior via run_agent ─────────────────────


class TestCrisisLogEndToEnd:
    """Drive the full compiled graph and verify the log captures crisis events."""

    @pytest.mark.asyncio
    async def test_crisis_turn_lands_in_backend(self) -> None:
        """A crisis message should write exactly one record via the graph."""

        backend = InMemoryCrisisLogBackend()
        result = await run_agent(
            AgentInput(message="I have pills and I am going to kill myself tonight."),
            crisis_log_backend=backend,
            llm_client=_FakeCrisisLLM(
                level=3,
                reason="LLM classified imminent self-harm risk.",
            ),
        )

        assert result.response_type == ResponseCategory.CRISIS
        assert await backend.arecord_count() == 1

    @pytest.mark.asyncio
    async def test_therapeutic_turn_does_not_write_to_crisis_log(self) -> None:
        """Non-crisis turns should never write to the crisis log."""

        backend = InMemoryCrisisLogBackend()
        result = await run_agent(
            AgentInput(message="I had a rough day at work today."),
            crisis_log_backend=backend,
            llm_client=FakeCrossRestartLLM(),
        )

        assert result.response_type == ResponseCategory.THERAPEUTIC
        assert await backend.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_crisis_record_carries_the_classifier_level(self) -> None:
        """The logged record should have the same level the classifier assigned."""

        backend = InMemoryCrisisLogBackend()
        await run_agent(
            AgentInput(message="I've been thinking about ending it all."),
            crisis_log_backend=backend,
            llm_client=_FakeCrisisLLM(
                level=2,
                reason="LLM classified suicidal ideation.",
            ),
        )

        # Fetch across a 3-day window to handle timezone edge cases
        from datetime import timedelta

        today = date.today()  # noqa: DTZ011
        all_records = []
        for delta in (-1, 0, 1):
            all_records.extend(
                await backend.alist_by_date(today + timedelta(days=delta))
            )

        assert len(all_records) == 1
        assert all_records[0].level == 2


async def _fetch_all_records(
    backend: InMemoryCrisisLogBackend,
) -> list[CrisisLogRecord]:
    """Fetch all records from the backend across a 3-day UTC window.

    The backend buckets records by ``detected_at.date()`` (UTC), which
    may fall a day before or after the test machine's local date near
    midnight. The 3-day window is a paranoid but cheap way to avoid
    timezone-induced flakes.
    """

    from datetime import timedelta

    today = date.today()  # noqa: DTZ011
    records: list[CrisisLogRecord] = []
    for delta in (-1, 0, 1):
        records.extend(await backend.alist_by_date(today + timedelta(days=delta)))
    return records


# ─── 4. Crisis-debug metadata propagation ─────────────────────────────────


class TestCrisisLogMetadata:
    """End-to-end checks that the crisis gate's debug metadata reaches the log.

    The graph now uses only the structured LLM classifier for crisis
    assessment, so graph-produced crisis records should carry the
    ``llm_primary`` classifier path.
    """

    @pytest.mark.asyncio
    async def test_llm_level_3_path(self) -> None:
        """A level-3 LLM verdict should be logged as llm_primary."""

        backend = InMemoryCrisisLogBackend()
        await run_agent(
            AgentInput(message="I have pills and I am going to kill myself tonight."),
            crisis_log_backend=backend,
            llm_client=_FakeCrisisLLM(
                level=3,
                reason="LLM classified imminent self-harm risk.",
            ),
        )

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.override_kind == "none"
        assert record.classifier_path == "llm_primary"
        assert record.llm_failure_occurred is False
        assert record.level == 3

    @pytest.mark.asyncio
    async def test_llm_level_2_path(self) -> None:
        """A level-2 LLM verdict should be logged as llm_primary."""

        backend = InMemoryCrisisLogBackend()
        await run_agent(
            AgentInput(message="I've been thinking about ending it all."),
            crisis_log_backend=backend,
            llm_client=_FakeCrisisLLM(
                level=2,
                reason="LLM classified suicidal ideation.",
            ),
        )

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.classifier_path == "llm_primary"
        assert record.override_kind == "none"
        assert record.llm_failure_occurred is False
        assert record.level == 2

    @pytest.mark.asyncio
    async def test_llm_safe_path_does_not_write(self) -> None:
        """A level-0 LLM verdict should not write to the crisis log."""

        backend = InMemoryCrisisLogBackend()
        await run_agent(
            AgentInput(message="Work is killing me this week."),
            crisis_log_backend=backend,
            llm_client=FakeCrossRestartLLM(),
        )

        records = await _fetch_all_records(backend)
        assert len(records) == 0


# ─── 5. Retention purge (v0.8.1) ──────────────────────────────────────────


def _build_retention_record(
    *,
    record_id: str,
    detected_at: str,
    level: int = 1,
) -> CrisisLogRecord:
    """Build a minimal CrisisLogRecord for retention-purge tests.

    The retention tests don't care about classifier path or response
    metadata — only ``detected_at`` (which drives the cutoff filter)
    and the record ``id`` (so we can assert which records survived).
    Everything else gets sensible defaults.
    """

    return CrisisLogRecord(
        id=record_id,
        session_id_opaque="sess-hashed",
        user_id_or_null=None,
        detected_at=detected_at,
        level=level,
        classifier_path="llm_primary",
        confidence="medium",
        reason="retention purge test",
        override_kind="none",
        response_node_completed=True,
        llm_failure_occurred=False,
    )


class TestInMemoryCrisisLogRetentionPurge:
    """Pin the v0.8.1 ``apurge_before`` behavior on the in-memory backend.

    Symmetric tests for the SQLite backend live in
    ``tests/integration/audit/test_sqlite_crisis_log.py``. Both backends MUST behave
    identically — the runtime picks between them based on memory mode
    and the operator-facing contract must not depend on which backend
    is wired.
    """

    @pytest.mark.asyncio
    async def test_purge_deletes_only_records_before_cutoff(self) -> None:
        """Records strictly older than the cutoff are deleted; cutoff-day
        records are preserved. This is the exclusive-boundary contract
        the backend docstring promises."""

        backend = InMemoryCrisisLogBackend()
        # Three records across three dates bracketing the cutoff.
        await backend.aappend(
            _build_retention_record(record_id="old", detected_at="2025-12-01T10:00:00Z")
        )
        await backend.aappend(
            _build_retention_record(
                record_id="on_cutoff", detected_at="2026-04-01T10:00:00Z"
            )
        )
        await backend.aappend(
            _build_retention_record(
                record_id="recent", detected_at="2026-04-12T10:00:00Z"
            )
        )

        deleted = await backend.apurge_before(date(2026, 4, 1))

        # Only the old record was before the cutoff.
        assert deleted == 1
        assert await backend.arecord_count() == 2

        # Cutoff-day record survives (exclusive boundary).
        on_cutoff_bucket = await backend.alist_by_date(date(2026, 4, 1))
        assert len(on_cutoff_bucket) == 1
        assert on_cutoff_bucket[0].id == "on_cutoff"

        # Recent record survives.
        recent_bucket = await backend.alist_by_date(date(2026, 4, 12))
        assert len(recent_bucket) == 1
        assert recent_bucket[0].id == "recent"

        # Old record is gone.
        old_bucket = await backend.alist_by_date(date(2025, 12, 1))
        assert len(old_bucket) == 0

    @pytest.mark.asyncio
    async def test_purge_idempotent(self) -> None:
        """Running the same purge twice deletes records on the first call
        and zero on the second. The contract is idempotent because the
        predicate ``detected_date < cutoff`` is stable."""

        backend = InMemoryCrisisLogBackend()
        await backend.aappend(
            _build_retention_record(record_id="old", detected_at="2025-12-01T10:00:00Z")
        )
        await backend.aappend(
            _build_retention_record(
                record_id="recent", detected_at="2026-04-12T10:00:00Z"
            )
        )

        first = await backend.apurge_before(date(2026, 4, 1))
        second = await backend.apurge_before(date(2026, 4, 1))
        assert first == 1
        assert second == 0
        assert await backend.arecord_count() == 1

    @pytest.mark.asyncio
    async def test_purge_empty_backend_returns_zero(self) -> None:
        """Purging an empty backend returns 0 without raising."""

        backend = InMemoryCrisisLogBackend()
        deleted = await backend.apurge_before(date(2026, 4, 1))
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_purge_closed_backend_returns_zero(self) -> None:
        """Purge on a closed backend is a safe no-op (matches the closed-
        safe contract of the other methods)."""

        backend = InMemoryCrisisLogBackend()
        await backend.aclose()
        deleted = await backend.apurge_before(date(2026, 4, 1))
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_purge_preserves_multi_record_days(self) -> None:
        """Multiple records on the same day are treated as a unit — if
        the day is before the cutoff, all records on that day are
        deleted; if the day is on or after the cutoff, all records
        on that day are preserved."""

        backend = InMemoryCrisisLogBackend()
        # Two records on the same old day.
        await backend.aappend(
            _build_retention_record(
                record_id="old1", detected_at="2025-12-01T09:00:00Z"
            )
        )
        await backend.aappend(
            _build_retention_record(
                record_id="old2", detected_at="2025-12-01T15:00:00Z"
            )
        )
        # Two records on the same recent day.
        await backend.aappend(
            _build_retention_record(
                record_id="new1", detected_at="2026-04-12T09:00:00Z"
            )
        )
        await backend.aappend(
            _build_retention_record(
                record_id="new2", detected_at="2026-04-12T15:00:00Z"
            )
        )

        deleted = await backend.apurge_before(date(2026, 4, 1))
        assert deleted == 2
        assert await backend.arecord_count() == 2


class TestNullCrisisLogRetentionPurge:
    """NullCrisisLogBackend's apurge_before is a no-op that returns 0.

    This is protocol-conformance machinery, not a behavior test — but
    without this pin, a future refactor that forgets to implement the
    method on the null backend would surface as a mysterious
    AttributeError during retention cleanup rather than a test failure.
    """

    @pytest.mark.asyncio
    async def test_null_purge_returns_zero(self) -> None:
        from agent.audit.crisis_log import NullCrisisLogBackend

        backend = NullCrisisLogBackend()
        deleted = await backend.apurge_before(date(2026, 4, 1))
        assert deleted == 0
