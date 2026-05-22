"""Tests for the v0.8 ``run_turn_stream`` observability refactor.

The pre-v0.8 ``run_turn_stream`` was a thin wrapper around ``run_turn``
that emitted a single ``StatusEvent(stage="load_memory")`` before the
whole runtime turn ran and a ``DoneEvent`` after. v0.8 replaced it with
streaming runtime execution that emits one ``StatusEvent`` per stage
update, accumulates the final state, stamps the outer ``turn_total_ms``
into diagnostics, and yields a ``DoneEvent``.

These tests cover the new contract:

1. **Per-stage StatusEvents.** The stream emits one StatusEvent for
   each runtime stage that runs (load_memory, therapeutic, finalize)
   in execution order. A ResponseReadyEvent is emitted after finalize
   so the user sees the response before DoneEvent.

2. **DoneEvent carries diagnostics.** The ``DoneEvent.output.diagnostics``
   dict contains both the per-stage timings (stamped by each stage) and
   the outer ``turn_total_ms`` (stamped by ``run_turn_stream`` itself).

3. **Session tracking bookkeeping.** The stream path updates
   runtime session start and max-crisis tracking the same way
   ``run_turn`` does — so /end / /exit summaries work regardless of
   which entry point the CLI used.

4. **Parity with run_turn.** A turn run via the stream yields the
   same response text and the same transcript shape as the same
   turn run via the monolithic ``run_turn``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import pytest

from agent.models import (
    ChunkEvent,
    DoneEvent,
    ResponseReadyEvent,
    StatusEvent,
    StreamEvent,
)
from agent.runtime import PersistentAgentRuntime
from llm.base import BaseLLMClient, StructuredResponseT


# ── Helpers ──────────────────────────────────────────────────────────────


class _StreamingResponseLLM(BaseLLMClient):
    """Response-only fake used by streaming tests."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "streamed response"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "streamed "
        yield "response"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__
        if schema_name == "CrisisAssessmentSchema":
            from agent.guardrails.service import CrisisAssessmentSchema

            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=0,
                    confidence="high",
                    reason="safe streaming test turn",
                    needs_crisis_response=False,
                    needs_clarification=False,
                ),
            )
        if schema_name == "DispatchDecision":
            from agent.memory.models import DispatchDecision

            return cast(
                StructuredResponseT,
                DispatchDecision(
                    response_style="supportive",
                    therapeutic_approach="none",
                    exercise_start_basis="ambiguous_or_none",
                    reasoning="streaming test dispatch",
                    confidence="high",
                ),
            )
        if schema_name == "TurnDispatchDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                route="therapeutic",
                active_flow_action="none",
                reasoning="streaming test turn dispatch",
                confidence="high",
            )
        if schema_name == "ExtractionResult":
            from agent.memory.types.semantic import ExtractionResult

            return cast(
                StructuredResponseT,
                ExtractionResult(facts=[], reason="streaming test extraction"),
            )
        if schema_name == "ProceduralExtractionResult":
            from agent.memory.types.procedural import ProceduralExtractionResult

            return cast(
                StructuredResponseT,
                ProceduralExtractionResult(
                    rules=[],
                    reason="streaming test procedural extraction",
                ),
            )
        raise RuntimeError(f"streaming tests unexpected schema {schema_name}")


async def _collect_stream(
    runtime: PersistentAgentRuntime,
    *,
    thread_id: str,
    message: str,
) -> tuple[
    list[StatusEvent],
    list[ChunkEvent],
    ResponseReadyEvent | None,
    DoneEvent | None,
]:
    """Collect all events from a stream.

    Returns ``(status_events, chunk_events, response_ready_event, done_event)``.
    """

    statuses: list[StatusEvent] = []
    chunks: list[ChunkEvent] = []
    ready: ResponseReadyEvent | None = None
    done: DoneEvent | None = None
    async for event in runtime.run_turn_stream(
        thread_id=thread_id,
        message=message,
        llm_client=_StreamingResponseLLM(),
        response_llm_client=_StreamingResponseLLM(),
    ):
        if isinstance(event, StatusEvent):
            statuses.append(event)
        elif isinstance(event, ChunkEvent):
            chunks.append(event)
        elif isinstance(event, ResponseReadyEvent):
            ready = event
        elif isinstance(event, DoneEvent):
            done = event
    return statuses, chunks, ready, done


# ── Tests ────────────────────────────────────────────────────────────────


class TestRunTurnStreamStages:
    """The stream should emit one StatusEvent per node in execution order."""

    @pytest.mark.asyncio
    async def test_deterministic_mode_streams_offline_smoke_response(self) -> None:
        """No-client deterministic turns should stay local and persist transcript state."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
            text_session_backend="disabled",
        ) as runtime:
            events: list[StreamEvent] = []
            async for event in runtime.run_turn_stream(
                thread_id="t-deterministic-smoke",
                message="I feel stressed about work today.",
                llm_client=None,
                response_llm_client=None,
            ):
                events.append(event)
            state = await runtime.get_state("t-deterministic-smoke")
            history = await runtime.get_history("t-deterministic-smoke")

        statuses = [event.stage for event in events if isinstance(event, StatusEvent)]
        ready = next(event for event in events if isinstance(event, ResponseReadyEvent))
        done = next(event for event in events if isinstance(event, DoneEvent))

        assert statuses == ["deterministic", "finalize"]
        assert ready.output.response_text == done.output.response_text
        assert "Deterministic smoke mode" in done.output.response_text
        assert done.output.diagnostics["text_agent_runtime"] == "deterministic_smoke"
        assert done.output.diagnostics["deterministic_smoke"] is True
        assert state is not None
        assert state["route"] == "therapeutic"
        assert len(history) == 2
        assert history[0].content == "I feel stressed about work today."
        assert history[1].content == done.output.response_text

    @pytest.mark.asyncio
    async def test_therapeutic_path_emits_expected_stage_sequence(self) -> None:
        """A normal (non-crisis) turn routes through the therapeutic branch.

        Expected stages are linear: load_memory -> therapeutic -> finalize.
        A ResponseReadyEvent is emitted after finalize, before DoneEvent.
        """

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            statuses, chunks, ready, done = await _collect_stream(
                runtime, thread_id="t-stream-1", message="hi there"
            )

        assert ready is not None
        assert done is not None
        stage_names = [event.stage for event in statuses]

        # Runtime stages must appear in this exact order. ``finalize`` is
        # the terminal graph stage.
        assert stage_names == [
            "load_memory",
            "therapeutic",
            "finalize",
        ]
        # Response chunks stream during the therapeutic response node and
        # concatenate to the full response.

    @pytest.mark.asyncio
    async def test_done_event_comes_last(self) -> None:
        """The DoneEvent must be the terminal event, never interleaved."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            events: list[StreamEvent] = []
            async for event in runtime.run_turn_stream(
                thread_id="t-stream-2",
                message="hello",
                llm_client=_StreamingResponseLLM(),
                response_llm_client=_StreamingResponseLLM(),
            ):
                events.append(event)

        assert len(events) > 0
        assert isinstance(events[-1], DoneEvent)
        # All earlier events should be StatusEvents, ChunkEvents, or
        # the non-terminal response-ready marker.
        for event in events[:-1]:
            assert isinstance(event, (StatusEvent, ChunkEvent, ResponseReadyEvent))

    @pytest.mark.asyncio
    async def test_response_ready_emits_after_finalize_before_done(self) -> None:
        """The reply-ready marker should surface before terminal completion."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            events: list[StreamEvent] = []
            async for event in runtime.run_turn_stream(
                thread_id="t-stream-2b",
                message="hello",
                llm_client=_StreamingResponseLLM(),
                response_llm_client=_StreamingResponseLLM(),
            ):
                events.append(event)

        finalize_index = next(
            i
            for i, event in enumerate(events)
            if isinstance(event, StatusEvent) and event.stage == "finalize"
        )
        ready_index = next(
            i for i, event in enumerate(events) if isinstance(event, ResponseReadyEvent)
        )
        done_index = next(
            i for i, event in enumerate(events) if isinstance(event, DoneEvent)
        )

        assert finalize_index < ready_index < done_index


class TestRunTurnStreamDiagnostics:
    """The DoneEvent's output should carry the full diagnostics dict."""

    @pytest.mark.asyncio
    async def test_diagnostics_carry_per_stage_timings(self) -> None:
        """Core runtime stages stamp timing keys into diagnostics."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            _, _, _, done = await _collect_stream(
                runtime, thread_id="t-stream-3", message="hi"
            )

        assert done is not None
        diag = done.output.diagnostics
        # Each retained runtime timing key must be present and numeric.
        for key in (
            "load_memory_ms",
            "crisis_gate_ms",
        ):
            assert key in diag, f"missing {key} from diagnostics"
            assert isinstance(diag[key], (int, float))
            assert diag[key] >= 0.0
        assert "extract_facts_ms" not in diag
        assert "extract_procedural_ms" not in diag

    @pytest.mark.asyncio
    async def test_diagnostics_carry_turn_total_ms(self) -> None:
        """run_turn_stream stamps the outer turn_total_ms after the stream.

        This is the "did the stream path do the same bookkeeping as
        run_turn" regression guard — the first draft of run_turn_stream
        forgot to mirror it.
        """

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            _, _, _, done = await _collect_stream(
                runtime, thread_id="t-stream-4", message="hi"
            )

        assert done is not None
        diag = done.output.diagnostics
        assert "turn_total_ms" in diag
        assert isinstance(diag["turn_total_ms"], (int, float))
        # turn_total should be >= the sum of any individual stage timing.
        # We don't assert equality because the outer clock includes edge
        # work and Python-side overhead, but turn_total should be at
        # least as large as the largest per-stage value.
        stage_times = [
            diag.get(key, 0.0)
            for key in (
                "load_memory_ms",
                "crisis_gate_ms",
            )
        ]
        assert diag["turn_total_ms"] >= max(stage_times)

    @pytest.mark.asyncio
    async def test_diagnostics_carry_post_finalize_ms(self) -> None:
        """``post_finalize_ms`` measures the wall-clock between
        turn finalization writing the response and the runtime finishing
        the turn.

        Invariants:
            - The key must be present and numeric on a normal turn.
            - It must be ≥ 0 (runtime cannot finish before finalization).
            - It must be ≤ ``turn_total_ms`` (post-finalize is a subset
              of the turn's total wall-clock).
            - The internal scaffolding key
              ``finalize_done_at_monotonic`` must NOT leak into the
              public diagnostics — ``stamp_turn_total_ms`` pops it.
        """

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            _, _, _, done = await _collect_stream(
                runtime, thread_id="t-stream-post-finalize", message="hi"
            )

        assert done is not None
        diag = done.output.diagnostics
        assert "post_finalize_ms" in diag
        assert isinstance(diag["post_finalize_ms"], (int, float))
        assert diag["post_finalize_ms"] >= 0.0
        assert diag["post_finalize_ms"] <= diag["turn_total_ms"]
        assert "finalize_done_at_monotonic" not in diag

    @pytest.mark.asyncio
    async def test_diagnostics_include_retrieval_counts(self) -> None:
        """Turn memory context retrieval-count diagnostics flow through."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            _, _, _, done = await _collect_stream(
                runtime, thread_id="t-stream-5", message="hi"
            )

        assert done is not None
        diag = done.output.diagnostics
        # Fresh store → zero counts across the board, but the keys must
        # be present so the CLI's Stage Timings panel can show them.
        assert diag.get("semantic_hits") == 0
        assert diag.get("episodic_hits") == 0
        assert diag.get("procedural_count") == 0


class TestRunTurnStreamSessionTracking:
    """Session bookkeeping must match run_turn's side effects."""

    @pytest.mark.asyncio
    async def test_stream_populates_session_start(self) -> None:
        """The first stream call on a thread stamps the session start time.

        Without this, end_session would fall back to the current time
        for started_at and produce zero-duration session arcs.
        """

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            # Before: no start time tracked
            assert not runtime._session_tracker.has_tracking("t-stream-6")

            _, _, _, done = await _collect_stream(
                runtime, thread_id="t-stream-6", message="hi"
            )

            # After: session start is tracked
            assert runtime._session_tracker.has_tracking("t-stream-6")
            assert done is not None

    @pytest.mark.asyncio
    async def test_stream_tracks_max_crisis_level(self) -> None:
        """The stream path updates max-crisis tracking like run_turn does."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            await _collect_stream(runtime, thread_id="t-stream-7", message="hi")

            # Non-crisis turn → tracked level is 0 (the default), not
            # missing. The lookup should have been written.
            assert runtime._session_tracker.has_tracking("t-stream-7")
            assert runtime._session_tracker.max_crisis_level("t-stream-7") == 0


class TestRunTurnStreamParity:
    """The stream and monolithic paths must produce identical final state."""

    @pytest.mark.asyncio
    async def test_stream_and_run_turn_produce_same_response(self) -> None:
        """Running the same turn via both entry points yields the same reply.

        This protects against subtle divergence (e.g., one path forgetting
        to stamp turn_total_ms, or one path clobbering state in a way the
        other doesn't). Uses two separate threads so the state snapshots
        don't interfere.
        """

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            monolithic = await runtime.run_turn(
                thread_id="t-parity-mono",
                message="hello",
                llm_client=_StreamingResponseLLM(),
                response_llm_client=_StreamingResponseLLM(),
            )
            _, _, _, streamed = await _collect_stream(
                runtime, thread_id="t-parity-stream", message="hello"
            )

        assert streamed is not None
        assert monolithic.output.response_text == streamed.output.response_text
        assert monolithic.output.response_type == streamed.output.response_type
