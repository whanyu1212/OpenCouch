"""Tests for the crisis-log always-on safety audit trail.

Covers three concerns:
    1. ``write_crisis_log`` writes a record when the turn is flagged
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

from agent.runtime import run_agent
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.audit.models import (
    CrisisClassifierPath,
    CrisisLogPathCounts,
    CrisisLogRecord,
)
from agent.audit.summary import summarize_crisis_log_records
from agent.guardrails.service import CrisisAssessmentSchema
from agent.audit.crisis_log import write_crisis_log
from agent.models import AgentInput, CrisisAssessment, ResponseCategory
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

    def __init__(
        self,
        *,
        crisis_log_backend: InMemoryCrisisLogBackend,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=crisis_log_backend,
            memory_mode=memory_mode,
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


# ─── write_crisis_log unit tests ─────────────────────────────────────────


class TestCrisisLogNode:
    """Unit tests for crisis-log writes with mocked runtime + backend."""

    @pytest.mark.asyncio
    async def test_crisis_turn_writes_record(self) -> None:
        """When crisis.needs_crisis_response is True, a record is appended."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=3)

        delta = await write_crisis_log(state, runtime.context)

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

        await write_crisis_log(state, runtime.context)

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

        await write_crisis_log(state, runtime.context)

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.override_kind == "none"
        assert record.classifier_path == "llm_primary"
        assert record.llm_failure_occurred is False

    @pytest.mark.asyncio
    async def test_operational_review_metadata_is_written_to_record(self) -> None:
        """The record should carry non-prompt operational review metadata."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(
            crisis_audit={
                "crisis_override_kind": "none",
                "crisis_classifier_path": "llm_primary",
                "crisis_llm_failure_occurred": False,
            }
        )
        state["response_style"] = "crisis_response"
        state["resource_lookup_status"] = "found"
        state["found_resources"] = [
            {"name": "988", "phone": "988", "url": "https://988lifeline.org"}
        ]
        state["diagnostics"] = {
            "openai_crisis_tool_calls": ["lookup_crisis_resources"],
            "openai_crisis_tool_fallback": False,
        }

        await write_crisis_log(state, runtime.context)

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.event_type == "crisis_response"
        assert record.response_style == "crisis_response"
        assert record.resource_lookup_status == "found"
        assert record.resource_count == 1
        assert record.tool_calls == ["lookup_crisis_resources"]
        assert record.response_path == "sdk"
        assert record.fallback_reason is None

    @pytest.mark.asyncio
    async def test_response_llm_override_path_is_written_to_record(self) -> None:
        """The record should identify direct response-LLM crisis fallback paths."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state()
        state["response_style"] = "crisis_response"
        state["resource_lookup_status"] = "no_location"
        state["found_resources"] = []
        state["diagnostics"] = {
            "openai_response_llm_override": True,
            "openai_crisis_tool_fallback": True,
        }

        await write_crisis_log(state, runtime.context)

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.response_path == "response_llm_override"
        assert record.fallback_reason == "response_llm_override"
        assert record.resource_lookup_status == "no_location"
        assert record.resource_count == 0

    @pytest.mark.asyncio
    async def test_session_id_stored_as_opaque_hash(self) -> None:
        """The session_id should be hashed, not stored in plaintext."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(session_id="very-private-session-id")

        await write_crisis_log(state, runtime.context)

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
        """Incognito mode must scrub user_id even if state carries one."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(
            crisis_log_backend=backend,
            memory_mode=MemoryMode.INCOGNITO,
        )
        state = _build_crisis_state(user_id="alice", session_id="anonymous-session")

        await write_crisis_log(state, runtime.context)

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

        delta = await write_crisis_log(state, runtime.context)

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

        with caplog.at_level(logging.ERROR, logger="agent.audit.crisis_log"):
            delta = await write_crisis_log(state, runtime.context)

        assert delta == {}  # must not raise
        assert any("audit trail lost" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_pre_append_failure_is_logged_but_does_not_crash(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failure *before* aappend must also degrade to a logged error.

        The guard previously wrapped only ``backend.aappend``; record
        construction and backend resolution ran unguarded. A crisis audit
        write is a safety side-channel reached from inside the turn-finalize
        path, so any failure here -- not just an I/O failure -- must not
        propagate and break the turn lifecycle. This drives a failure at
        backend resolution, which now sits inside the widened guard.
        """

        class _RaisingContext:
            memory_mode = MemoryMode.LOCAL

            @property
            def crisis_log_backend(self) -> InMemoryCrisisLogBackend:
                raise RuntimeError("simulated backend resolution failure")

        state = _build_crisis_state()

        with caplog.at_level(logging.ERROR, logger="agent.audit.crisis_log"):
            delta = await write_crisis_log(state, cast(Any, _RaisingContext()))

        assert delta == {}  # must not raise
        assert any("audit trail lost" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_voice_crisis_record_uses_sdk_response_path(self) -> None:
        """Voice crisis turns must record a meaningful response_path, not unknown.

        Voice crisis handling answers in the Realtime model's live reply, so the
        text SDK fallback diagnostics keys are never set. Without a voice-aware
        branch every voice crisis record would read ``response_path="unknown"``,
        making the field useless for voice audit reads.
        """

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=2)
        state["response_style"] = "crisis_response"
        state["diagnostics"] = {
            "voice_runtime": "openai_realtime",
            "openai_crisis_tool_calls": ["lookup_crisis_resources"],
        }

        await write_crisis_log(state, runtime.context)

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        assert records[0].response_path == "sdk"


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


# ─── 5. Crisis-log aggregate summaries ───────────────────────────────────────


class TestCrisisLogSummaries:
    """Summaries turn daily crisis records into operator-facing counts."""

    def test_daily_summary_counts_levels_paths_and_fallbacks(self) -> None:
        records = [
            _build_retention_record(
                record_id="a",
                detected_at="2026-04-10T08:00:00Z",
                level=2,
                response_path="sdk",
                llm_failure_occurred=False,
                response_node_completed=True,
            ),
            _build_retention_record(
                record_id="b",
                detected_at="2026-04-10T09:00:00Z",
                level=2,
                response_path="sdk_tool_fallback",
                llm_failure_occurred=True,
                response_node_completed=True,
            ),
            _build_retention_record(
                record_id="c",
                detected_at="2026-04-10T10:00:00Z",
                level=3,
                response_path="response_llm_override",
                llm_failure_occurred=False,
                response_node_completed=False,
            ),
        ]

        aggregate = summarize_crisis_log_records(date(2026, 4, 10), records)

        assert aggregate.date == "2026-04-10"
        assert aggregate.events_total == 3
        assert aggregate.events_by_level.level_2 == 2
        assert aggregate.events_by_level.level_3 == 1
        assert aggregate.events_by_classifier_path.llm_primary == 3
        assert aggregate.llm_failures_total == 1
        assert aggregate.tool_fallbacks_total == 1
        assert aggregate.response_llm_overrides_total == 1
        assert aggregate.response_node_completion_rate == pytest.approx(2 / 3)

    def test_daily_summary_handles_empty_records(self) -> None:
        aggregate = summarize_crisis_log_records(date(2026, 4, 10), [])

        assert aggregate.date == "2026-04-10"
        assert aggregate.events_total == 0
        assert aggregate.llm_failures_total == 0
        assert aggregate.tool_fallbacks_total == 0
        assert aggregate.response_llm_overrides_total == 0
        assert aggregate.response_node_completion_rate == 1.0


# ─── 5. Retention purge (v0.8.1) ──────────────────────────────────────────


def _build_retention_record(
    *,
    record_id: str,
    detected_at: str,
    level: int = 1,
    response_path: str = "sdk",
    llm_failure_occurred: bool = False,
    response_node_completed: bool = True,
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
        response_node_completed=response_node_completed,
        llm_failure_occurred=llm_failure_occurred,
        response_path=response_path,
        response_style="crisis_response",
        resource_lookup_status="not_attempted",
        resource_count=0,
        tool_calls=[],
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
