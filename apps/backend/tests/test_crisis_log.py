"""Tests for the crisis_log_node always-on safety audit trail.

Covers three concerns:
    1. ``run_crisis_log_node`` writes a record when the turn is flagged
       as a crisis (and skips writes otherwise).
    2. The session_id hash is stable, opaque, and not reversible.
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
from agent.memory.crisis_log import InMemoryCrisisLogBackend
from agent.memory.models import CrisisLogRecord
from agent.models import AgentInput, CrisisAssessment, ResponseKind
from agent.nodes.crisis_log import _hash_session_id, run_crisis_log_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


# ─── Test helpers ────────────────────────────────────────────────────────


class _MockRuntime:
    """Minimal runtime stand-in exposing ``.context`` only."""

    def __init__(self, *, crisis_log_backend: InMemoryCrisisLogBackend) -> None:
        self.context: WorkflowContext = {
            "llm_client": None,
            "memory_store": None,  # type: ignore[typeddict-item]
            "crisis_log_backend": crisis_log_backend,
            "memory_mode": None,  # type: ignore[typeddict-item]
        }


def _build_crisis_state(
    *,
    level: int = 3,
    needs_crisis_response: bool = True,
    reason: str = "imminent self-harm language detected",
    user_id: str | None = "user-123",
    session_id: str | None = "thread-abc",
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
    }
    return cast(AgentState, state)


# ─── 1. _hash_session_id tests ───────────────────────────────────────────


class TestHashSessionId:
    """Unit tests for the SHA-256 session-id hashing helper."""

    def test_hash_is_stable(self) -> None:
        """Same input → same hash every call."""

        assert _hash_session_id("thread-123") == _hash_session_id("thread-123")

    def test_hash_is_sha256_hex(self) -> None:
        """The hash should be a 64-character hex digest matching SHA-256."""

        result = _hash_session_id("thread-123")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_matches_sha256_of_input(self) -> None:
        """The hash should equal sha256(input) for a known input."""

        expected = hashlib.sha256(b"thread-123").hexdigest()
        assert _hash_session_id("thread-123") == expected

    def test_hash_differs_for_different_inputs(self) -> None:
        """Different inputs produce different hashes."""

        assert _hash_session_id("a") != _hash_session_id("b")

    def test_hash_handles_none_with_placeholder(self) -> None:
        """Passing None should produce a stable hash from a placeholder."""

        result = _hash_session_id(None)
        expected = hashlib.sha256(b"__no_session_id__").hexdigest()
        assert result == expected


# ─── 2. run_crisis_log_node unit tests ───────────────────────────────────


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
        assert backend.record_count() == 1

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
        assert backend.record_count() == 0

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

        assert result.response_type == ResponseKind.CRISIS
        assert backend.record_count() == 1

    @pytest.mark.asyncio
    async def test_therapeutic_turn_does_not_write_to_crisis_log(self) -> None:
        """Non-crisis turns should never write to the crisis log."""

        backend = InMemoryCrisisLogBackend()
        result = await run_agent(
            AgentInput(message="I had a rough day at work today."),
            crisis_log_backend=backend,
        )

        assert result.response_type == ResponseKind.THERAPEUTIC
        assert backend.record_count() == 0

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
