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

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from datetime import date
from typing import Any, cast

import pytest

from agent.runtime import run_agent
from agent.audit.capture import capture_crisis_outcome, capture_voice_missed_crisis
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.events import AUDIT_CRISIS_LOG_APPEND
from agent.observability.recorder import InMemoryTraceRecorder
from agent.audit.models import (
    CrisisClassifierPath,
    CrisisLogPathCounts,
    CrisisLogRecord,
)
from agent.audit.summary import summarize_crisis_log_records
from agent.guardrails.service import CrisisAssessmentSchema
from agent.audit.crisis_log import write_crisis_log
from agent.models import AgentInput, CrisisAssessment, ResponseCategory
from agent.runtime.workflow_context import WorkflowContext
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
    async def test_over_long_reason_is_truncated_not_dropped(self) -> None:
        # Regression for #159: CrisisAssessment.reason is uncapped LLM output, but
        # CrisisLogRecord.reason enforces max_length=500. An over-long reason used to
        # raise ValidationError at record construction, which write_crisis_log swallows
        # — silently dropping the ENTIRE crisis audit record. The reason must be
        # truncated so the record is still written for this acute-crisis event.
        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=3, reason="x" * 600)

        await write_crisis_log(state, runtime.context)

        assert await backend.arecord_count() == 1
        today = date.today()  # noqa: DTZ011 - local date for bucketing
        records = await backend.alist_by_date(today)
        if not records:
            from datetime import timedelta

            records = await backend.alist_by_date(today - timedelta(days=1))
        assert len(records) == 1
        assert len(records[0].reason) == 500

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
    async def test_trace_context_is_written_to_crisis_record_and_safe_event(
        self,
    ) -> None:
        """Trace correlation should not export raw crisis payloads."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(
            level=2,
            reason="user said 'I want to end it'",
            user_id="user-sensitive",
            session_id="raw-thread-sensitive",
        )
        state["resource_lookup_status"] = "found"
        state["found_resources"] = [{"name": "Sensitive Hotline", "phone": "123"}]
        state["diagnostics"] = {"openai_crisis_tool_calls": ["lookup_crisis_resources"]}
        recorder = InMemoryTraceRecorder()
        trace_context = TraceContext(
            trace_id="trace-crisis-1",
            session_id="trace-session-1",
            turn_id="turn-1",
            runtime_mode="text",
            config=TraceConfig(enabled=True),
        )

        with use_trace_context(trace_context, recorder):
            await write_crisis_log(state, runtime.context)

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.trace_id == "trace-crisis-1"
        assert record.trace_session_id == "trace-session-1"
        assert record.trace_turn_id == "turn-1"
        assert record.trace_runtime_mode == "text"

        assert len(recorder.events) == 1
        event = recorder.events[0]
        assert event.name == AUDIT_CRISIS_LOG_APPEND
        assert event.attributes == {
            "audit_recorded": True,
            "event_type": "crisis_response",
            "level": 2,
            "classifier_path": "llm_primary",
            "resource_lookup_status": "found",
            "resource_count": 1,
            "response_path": "unknown",
            "runtime_mode": "text",
            "trace_correlated": True,
        }
        assert "user said" not in str(event.attributes)
        assert "user-sensitive" not in str(event.attributes)
        assert "raw-thread-sensitive" not in str(event.attributes)
        assert "reason" not in event.attributes

    @pytest.mark.asyncio
    async def test_disabled_trace_context_is_not_persisted(
        self,
    ) -> None:
        """Disabled trace contexts should behave like no trace for audit rows."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=2)
        recorder = InMemoryTraceRecorder()
        trace_context = TraceContext(
            trace_id="disabled-trace",
            session_id="disabled-session",
            turn_id="disabled-turn",
            runtime_mode="text",
        )

        with use_trace_context(trace_context, recorder):
            await write_crisis_log(state, runtime.context)

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.trace_id is None
        assert record.trace_session_id is None
        assert record.trace_turn_id is None
        assert record.trace_runtime_mode is None
        assert recorder.events == []

    @pytest.mark.asyncio
    async def test_crisis_record_trace_fields_are_optional_without_context(
        self,
    ) -> None:
        """Audit writes must not require active tracing."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=2)

        await write_crisis_log(state, runtime.context)

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.trace_id is None
        assert record.trace_session_id is None
        assert record.trace_turn_id is None
        assert record.trace_runtime_mode is None

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
    async def test_capture_seam_skips_non_crisis_turn(self) -> None:
        """The runtime capture seam should not call the backend for safe turns."""

        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=0, needs_crisis_response=False)

        result = await capture_crisis_outcome(state, runtime.context)

        assert result.status == "skipped"
        assert result.reason == "not_crisis_response"
        assert await backend.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_capture_seam_bounds_backend_latency(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Slow event storage should time out instead of holding the turn open."""

        class SlowBackend(InMemoryCrisisLogBackend):
            async def aappend(self, record: CrisisLogRecord) -> None:  # type: ignore[override]
                await asyncio.sleep(0.2)
                await super().aappend(record)

        backend = SlowBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=3)

        start = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="agent.audit.capture"):
            result = await capture_crisis_outcome(
                state,
                runtime.context,
                timeout_seconds=0.01,
            )
        elapsed = time.monotonic() - start

        assert result.status == "timeout"
        assert result.reason == "timeout"
        assert elapsed < 0.15
        assert await backend.arecord_count() == 0
        assert "safety event capture timed out" in caplog.text

    @pytest.mark.asyncio
    async def test_capture_seam_reports_backend_failure(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Capture status must fail when the lower-level append never persists."""

        class FailingBackend(InMemoryCrisisLogBackend):
            async def aappend(self, record: CrisisLogRecord) -> None:  # type: ignore[override]
                raise RuntimeError("simulated backend failure")

        backend = FailingBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state()

        with caplog.at_level(logging.WARNING, logger="agent.audit.capture"):
            result = await capture_crisis_outcome(state, runtime.context)

        assert result.status == "failed"
        assert result.reason == "exception"
        assert await backend.arecord_count() == 0
        assert "safety event capture failed" in caplog.text

    @pytest.mark.asyncio
    async def test_voice_missed_crisis_deduplication_is_observable(self) -> None:
        backend = InMemoryCrisisLogBackend()
        runtime = _MockRuntime(crisis_log_backend=backend)
        state = _build_crisis_state(level=3)
        state["diagnostics"] = {
            "voice_missed_crisis_audit_id": "voice-missed-crisis:stable-turn"
        }
        assessment = CrisisAssessment(
            level=3,
            confidence="high",
            reason="missed crisis",
            needs_crisis_response=True,
            needs_clarification=False,
        )
        recorder = InMemoryTraceRecorder()
        trace_context = TraceContext(
            trace_id="voice-missed-crisis-dedup",
            runtime_mode="voice",
            config=TraceConfig(enabled=True),
        )

        with use_trace_context(trace_context, recorder):
            first = await capture_voice_missed_crisis(
                state,
                runtime.context,
                assessment=assessment,
            )
            second = await capture_voice_missed_crisis(
                state,
                runtime.context,
                assessment=assessment,
            )

        append_events = [
            event for event in recorder.events if event.name == AUDIT_CRISIS_LOG_APPEND
        ]
        assert first.status == "captured"
        assert second.status == "captured"
        assert await backend.arecord_count() == 1
        assert [event.attributes["audit_recorded"] for event in append_events] == [
            True,
            False,
        ]

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
            _build_retention_record(
                record_id="d",
                detected_at="2026-04-10T11:00:00Z",
                level=2,
                event_type="voice_missed_crisis",
                classifier_path="voice_post_turn",
                response_path="not_routed",
                llm_failure_occurred=False,
                response_node_completed=False,
            ),
        ]

        aggregate = summarize_crisis_log_records(date(2026, 4, 10), records)

        assert aggregate.date == "2026-04-10"
        assert aggregate.events_total == 4
        assert aggregate.events_by_level.level_2 == 3
        assert aggregate.events_by_level.level_3 == 1
        assert aggregate.events_by_classifier_path.llm_primary == 3
        assert aggregate.events_by_classifier_path.voice_post_turn == 1
        assert aggregate.llm_failures_total == 1
        assert aggregate.tool_fallbacks_total == 1
        assert aggregate.response_llm_overrides_total == 1
        assert aggregate.voice_missed_crises_total == 1
        assert aggregate.response_node_completion_rate == pytest.approx(2 / 4)

    def test_daily_summary_handles_empty_records(self) -> None:
        aggregate = summarize_crisis_log_records(date(2026, 4, 10), [])

        assert aggregate.date == "2026-04-10"
        assert aggregate.events_total == 0
        assert aggregate.llm_failures_total == 0
        assert aggregate.tool_fallbacks_total == 0
        assert aggregate.response_llm_overrides_total == 0
        assert aggregate.voice_missed_crises_total == 0
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
    event_type: str = "crisis_response",
    classifier_path: str = "llm_primary",
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
        event_type=event_type,
        classifier_path=classifier_path,
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
    """Pin ``apurge_before`` behavior on the non-durable backend.

    The durable Postgres contract is covered under integration/persistence.
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


class TestCrisisStatusLiteralConsolidation:
    """Structural guards, not behavior tests.

    The crisis-resource-lookup status type used to be declared in three
    places under two names, plus a hand-maintained validation set — so
    adding a value meant editing four copies in lockstep. These pins keep
    the type at one definition under one name, and keep the runtime
    validation set derived from (not copied from) that type, so the two
    can never drift again.
    """

    def test_status_literal_has_the_expected_values(self) -> None:
        from typing import get_args

        from agent.audit.models import CrisisResourceLookupStatus

        assert set(get_args(CrisisResourceLookupStatus)) == {
            "not_attempted",
            "pending",
            "found",
            "no_location",
            "location_refused",
            "no_verified_results",
            "lookup_error",
        }

    def test_validation_set_is_derived_from_the_literal(self) -> None:
        from typing import get_args

        from agent.audit import crisis_log
        from agent.audit.models import CrisisResourceLookupStatus

        assert crisis_log._VALID_RESOURCE_LOOKUP_STATUSES == frozenset(
            get_args(CrisisResourceLookupStatus)
        )

    def test_old_tool_status_name_is_gone(self) -> None:
        from agent.runtime import context

        assert not hasattr(context, "CrisisResourceToolStatus")


class TestCrisisAuditSeam:
    """The text crisis branch should not own runtime audit persistence.

    Crisis flow builds the response state only. The outer runtime finalization
    boundary owns bounded post-finalization capture through
    ``capture_crisis_outcome`` so audit writes cannot hold the live response
    branch open indefinitely.
    """

    def test_text_flow_does_not_write_crisis_audit_directly(self) -> None:
        import inspect

        from agent.flows import crisis as crisis_flow
        from agent.runtime import finalization as runtime_finalization
        from agent.runtime import turn as one_shot_turn

        crisis_source = inspect.getsource(crisis_flow)
        assert "record_crisis_outcome" not in crisis_source
        assert "write_crisis_log" not in crisis_source
        assert "capture_crisis_outcome" in inspect.getsource(runtime_finalization)
        assert "capture_crisis_outcome" in inspect.getsource(one_shot_turn)


class TestCrisisRecordSerializationSeam:
    """The shared serialization boundary round-trips a record losslessly.

    Both crisis drivers route the Pydantic boundary through
    ``serialize_crisis_record`` / ``deserialize_crisis_record`` while
    keeping their own storage encoding (SQLite TEXT via ``json.dumps``,
    Postgres JSONB via ``Jsonb``). These pin the boundary's contract:
    serialize→deserialize is the identity, and the serialized form is a
    plain JSON-encodable dict that both storage paths can encode.
    """

    def test_round_trip_is_identity(self) -> None:
        from agent.audit.crisis_log_serialization import (
            deserialize_crisis_record,
            serialize_crisis_record,
        )

        record = _build_retention_record(
            record_id="seam-roundtrip",
            detected_at="2026-04-16T10:00:00Z",
        )
        restored = deserialize_crisis_record(serialize_crisis_record(record))
        assert restored == record

    def test_serialized_form_is_json_encodable_dict(self) -> None:
        import json

        from agent.audit.crisis_log_serialization import serialize_crisis_record

        record = _build_retention_record(
            record_id="seam-jsonable",
            detected_at="2026-04-16T10:00:00Z",
        )
        serialized = serialize_crisis_record(record)
        assert isinstance(serialized, dict)
        # Must survive json.dumps (SQLite TEXT path) without a custom encoder;
        # the same plain-dict shape is what Postgres wraps in Jsonb.
        json.dumps(serialized)
