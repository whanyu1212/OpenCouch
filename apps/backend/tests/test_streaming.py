"""Tests for the v0.8 ``run_turn_stream`` observability refactor.

The pre-v0.8 ``run_turn_stream`` was a thin wrapper around ``run_turn``
that emitted a single ``StatusEvent(stage="load_memory")`` before the
whole graph ran and a ``DoneEvent`` after. v0.8 replaced it with a
multi-mode ``graph.astream`` call that emits one ``StatusEvent`` per
node update, accumulates the final state from the ``values`` chunks,
stamps the outer ``turn_total_ms`` into diagnostics, and yields a
``DoneEvent``.

These tests cover the new contract:

1. **Per-node StatusEvents.** The stream emits one StatusEvent for
   each node that runs (crisis_gate, load_memory, therapeutic OR
   crisis_response+crisis_log, extract_facts, extract_procedural,
   finalize) in execution order.

2. **DoneEvent carries diagnostics.** The ``DoneEvent.output.diagnostics``
   dict contains both the per-node timings (stamped by each node) and
   the outer ``turn_total_ms`` (stamped by ``run_turn_stream`` itself).

3. **Session tracking bookkeeping.** The stream path updates
   ``_session_starts`` and ``_max_crisis_levels`` the same way
   ``run_turn`` does — so /end / /exit summaries work regardless of
   which entry point the CLI used.

4. **Parity with run_turn.** A turn run via the stream yields the
   same response text and the same transcript shape as the same
   turn run via the monolithic ``run_turn``.
"""

from __future__ import annotations

import pytest

from agent.models import (
    DoneEvent,
    StatusEvent,
    StreamEvent,
)
from agent.persistence import PersistentAgentRuntime


# ── Helpers ──────────────────────────────────────────────────────────────


async def _collect_stream(
    runtime: PersistentAgentRuntime,
    *,
    thread_id: str,
    message: str,
) -> tuple[list[StatusEvent], DoneEvent | None]:
    """Collect all StatusEvents and the terminal DoneEvent from a stream.

    Returns a tuple ``(status_events, done_event)`` where ``done_event``
    is None if the stream terminates without one (which would be a bug
    in ``run_turn_stream`` and the calling test should fail on that
    assertion).
    """

    statuses: list[StatusEvent] = []
    done: DoneEvent | None = None
    async for event in runtime.run_turn_stream(thread_id=thread_id, message=message):
        if isinstance(event, StatusEvent):
            statuses.append(event)
        elif isinstance(event, DoneEvent):
            done = event
    return statuses, done


# ── Tests ────────────────────────────────────────────────────────────────


class TestRunTurnStreamStages:
    """The stream should emit one StatusEvent per node in execution order."""

    @pytest.mark.asyncio
    async def test_therapeutic_path_emits_expected_stage_sequence(self) -> None:
        """A normal (non-crisis) turn routes through the therapeutic branch.

        Expected stage order (v0.9 safety reorder):
            crisis_gate → load_memory → therapeutic →
            extract_facts → extract_procedural → finalize
        """

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            statuses, done = await _collect_stream(
                runtime, thread_id="t-stream-1", message="hi there"
            )

        assert done is not None
        # The stages should appear in the order the graph executes them.
        stage_names = [event.stage for event in statuses]
        assert stage_names == [
            "crisis_gate",
            "load_memory",
            "therapeutic",
            "extract_facts",
            "extract_procedural",
            "finalize",
        ]

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
                thread_id="t-stream-2", message="hello"
            ):
                events.append(event)

        assert len(events) > 0
        assert isinstance(events[-1], DoneEvent)
        # All earlier events should be StatusEvents (no ChunkEvents yet
        # in the deterministic path — that's a streaming-LLM future).
        for event in events[:-1]:
            assert isinstance(event, StatusEvent)


class TestRunTurnStreamDiagnostics:
    """The DoneEvent's output should carry the full diagnostics dict."""

    @pytest.mark.asyncio
    async def test_diagnostics_carry_per_stage_timings(self) -> None:
        """Each node stamps its own timing key into diagnostics."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            _, done = await _collect_stream(
                runtime, thread_id="t-stream-3", message="hi"
            )

        assert done is not None
        diag = done.output.diagnostics
        # Each node's timing key must be present and numeric.
        for key in (
            "load_memory_ms",
            "crisis_gate_ms",
            "extract_facts_ms",
            "extract_procedural_ms",
        ):
            assert key in diag, f"missing {key} from diagnostics"
            assert isinstance(diag[key], (int, float))
            assert diag[key] >= 0.0

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
            _, done = await _collect_stream(
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
                "extract_facts_ms",
                "extract_procedural_ms",
            )
        ]
        assert diag["turn_total_ms"] >= max(stage_times)

    @pytest.mark.asyncio
    async def test_diagnostics_include_retrieval_counts(self) -> None:
        """load_memory_node's retrieval-count diagnostics flow through."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            _, done = await _collect_stream(
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
            assert "t-stream-6" not in runtime._session_starts

            _, done = await _collect_stream(
                runtime, thread_id="t-stream-6", message="hi"
            )

            # After: session start is tracked
            assert "t-stream-6" in runtime._session_starts
            assert done is not None

    @pytest.mark.asyncio
    async def test_stream_tracks_max_crisis_level(self) -> None:
        """The stream path updates _max_crisis_levels like run_turn does."""

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            await _collect_stream(runtime, thread_id="t-stream-7", message="hi")

            # Non-crisis turn → tracked level is 0 (the default), not
            # missing. The lookup should have been written.
            assert "t-stream-7" in runtime._max_crisis_levels
            assert runtime._max_crisis_levels["t-stream-7"] == 0


class TestRunTurnStreamParity:
    """The stream and monolithic paths must produce identical final state."""

    @pytest.mark.asyncio
    async def test_stream_and_run_turn_produce_same_response(self) -> None:
        """Running the same turn via both entry points yields the same reply.

        This protects against subtle divergence (e.g., one path forgetting
        to stamp turn_total_ms, or one path clobbering state in a way the
        other doesn't). Uses two separate threads so the checkpoints
        don't interfere.
        """

        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_sqlite_path=":memory:",
            crisis_log_sqlite_path=":memory:",
        ) as runtime:
            monolithic = await runtime.run_turn(
                thread_id="t-parity-mono", message="hello"
            )
            _, streamed = await _collect_stream(
                runtime, thread_id="t-parity-stream", message="hello"
            )

        assert streamed is not None
        assert monolithic.output.response_text == streamed.output.response_text
        assert monolithic.output.response_type == streamed.output.response_type
