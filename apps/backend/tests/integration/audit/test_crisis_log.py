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
from agent.models import AgentInput, CrisisAssessment, ResponseCategory
from agent.nodes.crisis_log import run_crisis_log_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
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
                "crisis_override_kind": "imminent_risk",
                "crisis_classifier_path": "override",
                "crisis_llm_failure_occurred": False,
            }
        )

        await run_crisis_log_node(state, runtime)  # type: ignore[arg-type]

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.override_kind == "imminent_risk"
        assert record.classifier_path == "override"
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
        # "ending it all" is a CLEAR_SELF_HARM_PATTERN → level 2
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


# ─── 4. Crisis-debug metadata propagation (v0.2) ──────────────────────────


class TestCrisisLogMetadata:
    """End-to-end checks that the crisis gate's debug metadata reaches the log.

    Each test exercises one of the five dispatch paths in
    ``run_crisis_gate_node`` and asserts the resulting audit record
    carries the expected ``override_kind`` / ``classifier_path`` /
    ``llm_failure_occurred`` values. Tests that need an LLM client use
    a fake one; the no-LLM paths run against ``run_agent`` with no
    client configured.
    """

    @pytest.mark.asyncio
    async def test_override_path_imminent_risk(self) -> None:
        """Path 1a: imminent-risk regex override → override_kind=imminent_risk."""

        backend = InMemoryCrisisLogBackend()
        await run_agent(
            AgentInput(message="I have pills and I am going to kill myself tonight."),
            crisis_log_backend=backend,
        )

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.override_kind == "imminent_risk"
        assert record.classifier_path == "override"
        assert record.llm_failure_occurred is False
        assert record.level == 3

    @pytest.mark.asyncio
    async def test_deterministic_high_path(self) -> None:
        """Path 2: deterministic ladder returns level ≥ 2 → classifier_path=deterministic."""

        backend = InMemoryCrisisLogBackend()
        await run_agent(
            AgentInput(message="I've been thinking about ending it all."),
            crisis_log_backend=backend,
        )

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.classifier_path == "deterministic"
        assert record.override_kind == "none"
        assert record.llm_failure_occurred is False
        # "ending it all" → CLEAR_SELF_HARM_PATTERN → level 2
        assert record.level == 2

    @pytest.mark.asyncio
    async def test_llm_success_path(self) -> None:
        """LLM classifier succeeds as primary path → classifier_path=llm_primary.

        Uses a fake LLM client that returns a crisis-flagged assessment
        for a message the deterministic ladder would rate as level 1.
        """

        from collections.abc import AsyncIterator
        from typing import cast

        from agent.gates.safety.service import CrisisAssessmentSchema
        from llm.base import BaseLLMClient, StructuredResponseT

        class _FakeCrisisLLM(BaseLLMClient):
            async def generate_text(
                self,
                *,
                prompt: str,
                system_instruction: str | None = None,
                use_search: bool = False,
            ) -> str:
                return "fake crisis response"

            async def generate_text_stream(
                self,
                *,
                prompt: str,
                system_instruction: str | None = None,
            ) -> AsyncIterator[str]:
                yield "fake"

            async def generate_structured(
                self,
                *,
                prompt: str,
                response_schema: type[StructuredResponseT],
                system_instruction: str | None = None,
            ) -> StructuredResponseT:
                # The crisis gate calls this for its classifier; return
                # a level-2 escalation.
                return cast(
                    StructuredResponseT,
                    CrisisAssessmentSchema(
                        level=2,
                        confidence="high",
                        reason="LLM escalated an ambiguous message to level 2",
                        needs_crisis_response=True,
                        needs_clarification=False,
                    ),
                )

        backend = InMemoryCrisisLogBackend()
        # Ambiguous phrase — deterministic ladder returns level 1 (not
        # crisis), so the LLM classifier runs and upgrades it to level 2.
        await run_agent(
            AgentInput(message="I just wish I could disappear for a while."),
            llm_client=_FakeCrisisLLM(),
            crisis_log_backend=backend,
        )

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.classifier_path == "llm_primary"
        assert record.override_kind == "none"
        assert record.llm_failure_occurred is False
        assert record.level == 2

    @pytest.mark.asyncio
    async def test_llm_failure_path(self) -> None:
        """Path 4: deterministic < 2, LLM raises → llm_failure_occurred=True.

        This is the distinguishing metadata path: classifier_path is
        still "deterministic" (because that's what we ultimately used)
        but llm_failure_occurred is True so an operator can tell "we
        fell back because we had to" from "we never tried the LLM".

        The deterministic fallback will report level 1 for the chosen
        message, which does NOT trigger crisis routing — meaning the
        crisis_log_node's defensive guard skips the write. So this test
        uses a message that the deterministic ladder rates at level ≥ 2
        AFTER the override check (no override pattern match). We pick
        a CLEAR_SELF_HARM phrase which the override won't match but the
        deterministic ladder will rate at level 2. That means the LLM
        path is never reached — deterministic-high fires first.

        Since path 4 requires the specific combination of (deterministic
        level 1) + (LLM called) + (LLM raised), and the deterministic
        fallback yields level 1 → non-crisis → no log write, this
        end-to-end path is unreachable for v0.1. The regression guard
        for llm_failure_occurred is covered by the Stage G1 test
        ``test_llm_failure_falls_back_to_regex_with_warning`` in
        ``test_therapeutic_routing.py`` — same control flow pattern.

        This test is a placeholder so future v0.3+ refactors remember
        to add path-4 coverage once deterministic level 1 can still
        write to the crisis log (e.g. via a clarification-path audit).
        """

        pytest.skip(
            "Path 4 (LLM failure on deterministic-low turn) is unreachable "
            "in v0.1 because the defensive guard skips writes on non-crisis "
            "turns. Revisit in v0.3+ if level-1 ambiguous turns start "
            "writing audit records."
        )

    @pytest.mark.asyncio
    async def test_no_llm_client_path(self) -> None:
        """Path 5: no LLM client → classifier_path=deterministic, llm_failure=False.

        Same-shape record as path 2 (deterministic-high), but the
        distinction is whether an LLM was even attempted. For a message
        that triggers the deterministic ladder directly (level 2), the
        records from path 2 and path 5 are indistinguishable — both
        report classifier_path="deterministic". The llm_failure field
        is the only discriminator between them.
        """

        backend = InMemoryCrisisLogBackend()
        # Same message as test_deterministic_high_path, no llm_client.
        # Path 2 and path 5 converge on the same record for this case.
        await run_agent(
            AgentInput(message="I've been thinking about ending it all."),
            crisis_log_backend=backend,
        )

        records = await _fetch_all_records(backend)
        assert len(records) == 1
        record = records[0]
        assert record.classifier_path == "deterministic"
        assert record.override_kind == "none"
        assert record.llm_failure_occurred is False

    @pytest.mark.asyncio
    async def test_idiomatic_safe_override_does_not_write(self) -> None:
        """Path 1b: idiomatic-safe override → non-crisis → no log write.

        The idiomatic-safe override returns a non-crisis assessment
        (``needs_crisis_response=False``), so the crisis_log_node's
        defensive guard skips the write. This test verifies the guard
        fires for this path — a regression where idiomatic-safe
        accidentally writes to the log would be a privacy issue.
        """

        backend = InMemoryCrisisLogBackend()
        await run_agent(
            AgentInput(message="Work is killing me this week."),
            crisis_log_backend=backend,
            llm_client=FakeCrossRestartLLM(),
        )

        # No record should be written — idiomatic-safe is a safe
        # negative, not a crisis event.
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
        classifier_path="deterministic",
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
