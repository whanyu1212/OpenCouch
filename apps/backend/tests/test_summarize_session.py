"""Unit tests for the session summarizer function.

The summarizer is NOT a graph node — it's a standalone async function
invoked by ``PersistentAgentRuntime.end_session`` at session end. These
tests exercise its full behavior matrix: early exits, LLM success with
an arc, LLM success with None (thin session), LLM failure, store
write failure, and the ``SessionArc → StoredSessionArc`` promotion.

All tests are deterministic — no live API calls. The fake LLM client
dispatches on ``response_schema`` so it can coexist with the crisis
classifier and dispatcher in future integration tests (same pattern as
``_FakeExtractionLLM`` in test_extract_facts.py).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.memory.models import (
    CBTContext,
    MoodArc,
    SessionArc,
    StoredSessionArc,
    SummarizationResult,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.episodic import (
    session_arc_to_stored as _session_arc_to_stored,
)
from agent.nodes.summarize_session import run_summarize_session
from agent.state import AgentState
from services.llm.base import BaseLLMClient, StructuredResponseT


# ─── Test helpers ──────────────────────────────────────────────────────


def _make_session_arc(
    *,
    session_id: str = "session-test",
    summary: str = "User talked about work anxiety and an upcoming meeting.",
    primary_themes: list[str] | None = None,
    opened: str = "anxious",
    closed: str = "calmer",
    open_loops: list[str] | None = None,
    resolved_threads: list[str] | None = None,
) -> SessionArc:
    """Build a SessionArc with sensible defaults for testing.

    Note: ``crisis_level_max`` is NOT a SessionArc field anymore — it
    lives only on StoredSessionArc and is populated from a runtime-
    computed parameter during ``_session_arc_to_stored`` promotion.
    Tests that need a non-zero crisis level should pass it to
    ``_session_arc_to_stored`` explicitly.
    """

    return SessionArc(
        session_id=session_id,
        started_at="2026-04-10T12:00:00Z",
        ended_at="2026-04-10T12:30:00Z",
        duration_seconds=1800,
        turn_count=8,
        primary_themes=(
            primary_themes if primary_themes is not None else ["work stress"]
        ),
        summary=summary,
        mood_arc=MoodArc(opened=opened, closed=closed),
        open_loops=open_loops if open_loops is not None else [],
        resolved_threads=resolved_threads if resolved_threads is not None else [],
    )


def _partial_state(
    *,
    transcript: list[dict[str, str]] | None = None,
    user_id: str | None = None,
    session_id: str | None = "session-test",
) -> AgentState:
    """Build a partial AgentState for summarizer unit tests.

    Only the fields the summarizer reads (transcript, user_id, session_id)
    are populated. Cast to AgentState — the test is asserting behavior,
    not schema completeness.
    """

    state: Any = {
        "transcript": transcript
        or [
            {"role": "user", "content": "im feeling anxious"},
            {"role": "assistant", "content": "tell me more"},
            {"role": "user", "content": "work has been stressful"},
            {"role": "assistant", "content": "what's the hardest part"},
        ],
        "user_id": user_id,
        "session_id": session_id,
    }
    return cast(AgentState, state)


class _FakeSummarizerLLM(BaseLLMClient):
    """Fake LLM client for summarizer tests.

    Dispatches on ``response_schema.__name__``:
    - SummarizationResult → returns the canned ``summarization_result``
    - Any other schema → raises (tests should not hit this path).
    """

    def __init__(
        self,
        *,
        summarization_result: SummarizationResult,
        raise_on_summarization: bool = False,
    ) -> None:
        self.summarization_result = summarization_result
        self.raise_on_summarization = raise_on_summarization
        self.summarization_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "fake response"

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
        if response_schema.__name__ == "SummarizationResult":
            self.summarization_calls += 1
            if self.raise_on_summarization:
                raise RuntimeError("simulated summarization LLM failure")
            return cast(StructuredResponseT, self.summarization_result)

        raise RuntimeError(
            f"_FakeSummarizerLLM: unexpected schema {response_schema.__name__}"
        )


# ─── 1. _session_arc_to_stored helper tests ────────────────────────────


class TestSessionArcToStored:
    """Tests for the SessionArc → StoredSessionArc promotion helper."""

    def test_adds_store_metadata_fields(self) -> None:
        """The helper adds id, owner_id, created_at, last_referenced_at,
        and user_visible to the base SessionArc."""

        arc = _make_session_arc()
        stored = _session_arc_to_stored(arc, owner_id="user-42")

        assert stored.id  # uuid4 string
        assert len(stored.id) == 36  # standard uuid4 format
        assert stored.owner_id == "user-42"
        assert stored.created_at  # non-empty iso timestamp
        assert stored.last_referenced_at  # non-empty iso timestamp
        assert stored.user_visible is True
        assert stored.write_timing == "session_end"
        assert stored.write_reason
        assert stored.policy_version == "phase5_v1"

    def test_preserves_original_arc_fields(self) -> None:
        """All SessionArc fields pass through unchanged."""

        arc = _make_session_arc(
            summary="User discussed grief and sleep difficulties.",
            primary_themes=["grief", "sleep"],
            open_loops=["hasn't processed the funeral yet"],
        )
        stored = _session_arc_to_stored(arc, owner_id="user-1")

        assert stored.summary == "User discussed grief and sleep difficulties."
        assert stored.primary_themes == ["grief", "sleep"]
        assert stored.open_loops == ["hasn't processed the funeral yet"]
        assert stored.session_id == "session-test"
        # crisis_level_max now comes from the promotion-time parameter,
        # not the arc itself. Default is 0 when not explicitly passed.
        assert stored.crisis_level_max == 0

    def test_crisis_level_max_comes_from_promotion_parameter(self) -> None:
        """v0.4 refactor: ``crisis_level_max`` is NOT a SessionArc field
        (not LLM-produced). The runtime computes it as the peak crisis-
        gate level observed during the session and passes it to
        ``_session_arc_to_stored`` as a keyword argument. This test
        pins the new data flow."""

        arc = _make_session_arc()
        stored_with_crisis = _session_arc_to_stored(
            arc, owner_id="user-1", crisis_level_max=2
        )
        stored_without = _session_arc_to_stored(arc, owner_id="user-1")

        assert stored_with_crisis.crisis_level_max == 2
        assert stored_without.crisis_level_max == 0
        # And the arc itself must NOT have this field — pydantic would
        # raise on an unexpected-keyword SessionArc construction.
        assert not hasattr(arc, "crisis_level_max")

    def test_created_at_and_last_referenced_at_match(self) -> None:
        """On fresh creation, both timestamps should be the same
        (the record hasn't been re-read yet)."""

        arc = _make_session_arc()
        stored = _session_arc_to_stored(arc, owner_id="user-1")
        assert stored.created_at == stored.last_referenced_at


# ─── 2. Early-exit tests ───────────────────────────────────────────────


class TestSummarizerEarlyExits:
    """Silent-skip branches: no LLM, incognito mode."""

    @pytest.mark.asyncio
    async def test_no_llm_client_returns_none(self) -> None:
        """If llm_client is None, the summarizer should return None
        without touching the store. Same contract as extract_facts."""

        store = OpenCouchMemoryStore()
        state = _partial_state()

        result = await run_summarize_session(
            state,
            llm_client=None,
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
        )

        assert result is None
        assert await store.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_incognito_mode_returns_none(self) -> None:
        """Incognito mode should skip summarization entirely, even with
        a valid LLM client. No episodic writes in incognito is the
        symmetric contract with semantic extraction."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc()
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(
                arc=arc, reason="would have summarized"
            )
        )
        state = _partial_state()

        result = await run_summarize_session(
            state,
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.INCOGNITO,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
        )

        assert result is None
        assert await store.arecord_count() == 0
        # Critical: the LLM should NOT have been called — we skip before
        # making the expensive API call.
        assert fake.summarization_calls == 0


# ─── 3. Happy path + skip-with-reason tests ────────────────────────────


class TestSummarizerHappyPath:
    """The normal flow: LLM returns a valid arc or None with reason."""

    @pytest.mark.asyncio
    async def test_writes_arc_to_episodic_namespace(self) -> None:
        """A successful summarization should persist the arc to
        (owner_id, 'episodic') and return the StoredSessionArc."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc(
            summary="User discussed work anxiety and upcoming deadlines.",
            primary_themes=["work stress"],
            opened="anxious",
            closed="more grounded",
        )
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(
                arc=arc, reason="captured work anxiety arc over 4 turns"
            )
        )
        state = _partial_state(user_id="user-42")

        result = await run_summarize_session(
            state,
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
        )

        assert result is not None
        assert isinstance(result, StoredSessionArc)
        assert result.owner_id == "user-42"
        assert result.summary == "User discussed work anxiety and upcoming deadlines."
        assert result.primary_themes == ["work stress"]
        assert fake.summarization_calls == 1

        # Verify the record actually landed in the store
        assert await store.arecord_count(("user-42", "episodic")) == 1
        records = await store.asearch(("user-42", "episodic"), query=None)
        assert len(records) == 1
        assert records[0].value["summary"] == result.summary

    @pytest.mark.asyncio
    async def test_llm_returns_none_arc_skips_write(self) -> None:
        """When the LLM judges the session too thin to summarize, it
        returns arc=None with a reason. The summarizer should return
        None and NOT touch the store."""

        store = OpenCouchMemoryStore()
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(
                arc=None,
                reason="session had only 2 turns of small talk",
            )
        )
        state = _partial_state()

        result = await run_summarize_session(
            state,
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
        )

        assert result is None
        assert await store.arecord_count() == 0
        assert fake.summarization_calls == 1  # LLM WAS called — it just returned None

    @pytest.mark.asyncio
    async def test_reason_logged_at_info_level(self, caplog) -> None:
        """The LLM's reason should be logged at INFO so dogfood sessions
        see it without rewiring log levels — same contract as the v0.3.1
        extraction reason log."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc()
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(
                arc=arc, reason="captured 8-turn work anxiety arc"
            )
        )
        state = _partial_state()

        with caplog.at_level(logging.INFO, logger="agent.nodes.summarize_session"):
            await run_summarize_session(
                state,
                llm_client=fake,
                memory_store=store,
                memory_mode=MemoryMode.LOCAL,
                session_id="session-test",
                started_at="2026-04-10T12:00:00Z",
            )

        # The reason string appears in the logs
        assert any(
            "captured 8-turn work anxiety arc" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_owner_id_falls_back_to_session_id(self) -> None:
        """When state.user_id is None, the owner_id for the episodic
        namespace derives from session_id. This matches the load_memory
        and extract_facts convention."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc()
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(arc=arc, reason="summarized")
        )
        state = _partial_state(user_id=None, session_id="session-abc")

        result = await run_summarize_session(
            state,
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
            session_id="session-abc",
            started_at="2026-04-10T12:00:00Z",
        )

        assert result is not None
        assert result.owner_id == "session-abc"
        assert await store.arecord_count(("session-abc", "episodic")) == 1


# ─── 4. Failure-mode tests ─────────────────────────────────────────────


class TestSummarizerFailureModes:
    """LLM and store errors must degrade silently."""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none_and_logs_warning(self, caplog) -> None:
        """An exception from the LLM call should be caught, logged at
        WARNING, and the summarizer returns None. The session-end flow
        must not fail because of a summarization error."""

        store = OpenCouchMemoryStore()
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(
                arc=_make_session_arc(), reason="unused"
            ),
            raise_on_summarization=True,
        )
        state = _partial_state()

        with caplog.at_level(logging.WARNING, logger="agent.nodes.summarize_session"):
            result = await run_summarize_session(
                state,
                llm_client=fake,
                memory_store=store,
                memory_mode=MemoryMode.LOCAL,
                session_id="session-test",
                started_at="2026-04-10T12:00:00Z",
            )

        assert result is None
        assert await store.arecord_count() == 0
        assert any(
            "LLM structured-output call failed" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_malformed_timestamps_degrade_to_zero_duration(self, caplog) -> None:
        """If the caller passes unparseable started_at / ended_at, the
        summarizer should log a warning and continue with duration=0.
        It must NOT crash — the session end should still produce a
        summary even if the timestamps are bad."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc()
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(
                arc=arc, reason="summarized despite bad timestamps"
            )
        )
        state = _partial_state()

        with caplog.at_level(logging.WARNING, logger="agent.nodes.summarize_session"):
            result = await run_summarize_session(
                state,
                llm_client=fake,
                memory_store=store,
                memory_mode=MemoryMode.LOCAL,
                session_id="session-test",
                started_at="not-a-real-timestamp",
                ended_at="also-bad",
            )

        # Still wrote the arc despite the bad timestamps
        assert result is not None
        assert await store.arecord_count() == 1
        # And logged the warning
        assert any("could not parse" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_default_ended_at_when_none(self) -> None:
        """If ended_at is None, the summarizer should default to now().
        This is the common call pattern from the runtime."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc()
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(arc=arc, reason="summarized")
        )
        state = _partial_state()

        result = await run_summarize_session(
            state,
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
            ended_at=None,  # explicitly None — summarizer should fill in
        )

        assert result is not None
        # The stored arc has a created_at that's close to now
        assert result.created_at  # non-empty


# ─── 5. Approach context tests ──────────────────────────────────────────


class TestSummarizerApproachContext:
    """Tests for the approach_used / approach_context fields on SessionArc."""

    def test_promotion_preserves_approach_fields(self) -> None:
        """approach_used and approach_context on SessionArc should
        survive promotion to StoredSessionArc via _session_arc_to_stored."""

        arc = _make_session_arc()
        arc.approach_used = "cbt"
        arc.approach_context = CBTContext(
            thought_examined="I'm going to get fired",
            action_step="speak up in one meeting",
            tool_used="thought_record",
        )
        stored = _session_arc_to_stored(arc, owner_id="user-1")

        assert stored.approach_used == "cbt"
        assert isinstance(stored.approach_context, CBTContext)
        assert stored.approach_context.thought_examined == "I'm going to get fired"

    def test_promotion_without_approach_defaults_to_none(self) -> None:
        """SessionArcs without approach fields (backward compat) should
        promote cleanly with None defaults."""

        arc = _make_session_arc()
        stored = _session_arc_to_stored(arc, owner_id="user-1")

        assert stored.approach_used is None
        assert stored.approach_context is None

    def test_approach_context_survives_json_round_trip(self) -> None:
        """The approach_context discriminated union must survive
        model_dump → model_validate for store persistence."""

        arc = _make_session_arc()
        arc.approach_used = "cbt"
        arc.approach_context = CBTContext(
            thought_examined="Nobody values my work",
            action_step="ask for feedback from one colleague",
        )
        stored = _session_arc_to_stored(arc, owner_id="user-1")

        # Round-trip through JSON (same path as store.aput / store.aget)
        dumped = stored.model_dump(mode="json")
        reloaded = StoredSessionArc.model_validate(dumped)

        assert reloaded.approach_used == "cbt"
        assert isinstance(reloaded.approach_context, CBTContext)
        assert reloaded.approach_context.thought_examined == "Nobody values my work"
        assert (
            reloaded.approach_context.action_step
            == "ask for feedback from one colleague"
        )

    @pytest.mark.asyncio
    async def test_approach_hint_passed_through_to_llm(self) -> None:
        """When approach_hint is provided, the summarizer should pass it
        to the prompt builder and the LLM should receive it."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc()
        arc.approach_used = "cbt"
        arc.approach_context = CBTContext(
            thought_examined="I always fail",
            action_step="try one small task",
        )
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(arc=arc, reason="captured CBT arc")
        )
        state = _partial_state(user_id="user-42")

        result = await run_summarize_session(
            state,
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
            approach_hint="cbt",
        )

        assert result is not None
        assert result.approach_used == "cbt"
        assert isinstance(result.approach_context, CBTContext)

    @pytest.mark.asyncio
    async def test_no_approach_hint_produces_none_fields(self) -> None:
        """Without approach_hint, the approach fields should be None
        (backward-compatible behavior)."""

        store = OpenCouchMemoryStore()
        arc = _make_session_arc()  # no approach fields set
        fake = _FakeSummarizerLLM(
            summarization_result=SummarizationResult(arc=arc, reason="no approach")
        )
        state = _partial_state()

        result = await run_summarize_session(
            state,
            llm_client=fake,
            memory_store=store,
            memory_mode=MemoryMode.LOCAL,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
        )

        assert result is not None
        assert result.approach_used is None
        assert result.approach_context is None
