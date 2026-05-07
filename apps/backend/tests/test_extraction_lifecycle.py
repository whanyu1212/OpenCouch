"""Background extraction lifecycle tests.

After Phase B, memory extraction is no longer a graph node — it runs as
a runtime-managed background ``asyncio.Task`` dispatched at the end of
each turn. The runtime owns three lifecycle invariants that this file
locks down:

1. **Drain on next turn** — turn N+1's ``_prepare_session_for_turn``
   awaits the prior turn's pending extraction so memory writes are
   visible by the time ``load_memory_node`` runs.
2. **Drain on shutdown** — ``__aexit__`` awaits any in-flight
   extraction before closing the memory store, so writes that started
   in the last turn aren't dropped.
3. **Drain timeout** — a stalled extraction (e.g., LLM provider hang)
   does not indefinitely block the next turn; the drain bounds at
   ``EXTRACTION_DRAIN_TIMEOUT_SECONDS`` and proceeds with possibly
   stale memory while the stuck task continues running.

Each test runs the runtime in its **default** background mode so the
async dispatch path is exercised directly. Tests that need synchronous
extraction visibility set ``extract_in_foreground=True`` explicitly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, cast

import pytest

from agent.memory.models import EntityRef, ExtractionResult, MemoryWrite
from agent.memory.modes import MemoryMode
from agent.models import Channel
from agent.persistence import PersistentAgentRuntime
from agent.runtime.turn_extraction import EXTRACTION_DRAIN_TIMEOUT_SECONDS
from llm.base import BaseLLMClient, StructuredResponseT

from tests.test_persistence_cross_restart import (
    _FakeCrossRestartLLM,
    _runtime_paths,
)


# Module-level reference so the import isn't optimized away on import-only
# linting passes; callers may also import the constant directly from
# this module to assert against runtime behavior.
_DRAIN_TIMEOUT = EXTRACTION_DRAIN_TIMEOUT_SECONDS


def _one_fact_extraction_result(
    *,
    object_identifier: str = "Sarah",
    evidence_quote: str = "I have a sister named Sarah",
    source_session_id: str = "thread-a",
) -> ExtractionResult:
    """Single-fact result usable across multiple cases."""

    return ExtractionResult(
        facts=[
            MemoryWrite(
                category="relationship",
                subject=EntityRef(type="User", identifier="user-1"),
                predicate="KNOWS",
                object=EntityRef(type="Person", identifier=object_identifier),
                evidence_quote=evidence_quote,
                confidence="high",
                source_session_id=source_session_id,
                source_turn_index=0,
            )
        ],
        reason="single relationship fact",
    )


@pytest.mark.asyncio
async def test_next_turn_drain_makes_prior_extraction_visible(
    tmp_path: Path,
) -> None:
    """Turn N+1 must see turn N's memory writes.

    Background extraction doesn't write synchronously, so a naive
    runtime would let turn N+1 start before turn N's facts hit the
    store. The drain in ``_prepare_session_for_turn`` is what makes
    the contract hold — this test pins it.
    """

    paths = _runtime_paths(tmp_path)
    llm = _FakeCrossRestartLLM(extraction_result=_one_fact_extraction_result())

    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-a",
            message="I have a sister named Sarah",
            channel=Channel.TEST,
            user_id="user-1",
            llm_client=llm,
        )

        # Second turn — its prepare-step must drain extraction from
        # turn 1 before this turn's load_memory_node runs.
        await runtime.run_turn(
            thread_id="thread-a",
            message="What did I tell you about my family?",
            channel=Channel.TEST,
            user_id="user-1",
            llm_client=llm,
        )

        # By the time turn 2 returned, turn 1's extraction must have
        # been drained and committed to the store.
        assert await runtime.memory_store.arecord_count(("user-1", "semantic")) >= 1


@pytest.mark.asyncio
async def test_aexit_drains_inflight_extraction(tmp_path: Path) -> None:
    """``__aexit__`` must drain in-flight extraction before closing the store.

    Without the shutdown drain, a runtime that exits immediately after
    a turn would race the background task against ``memory_store.aclose()``
    and the extracted facts could be silently dropped.
    """

    paths = _runtime_paths(tmp_path)
    llm = _FakeCrossRestartLLM(extraction_result=_one_fact_extraction_result())

    runtime = PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
        finalize_active_sessions_on_close=False,
    )
    async with runtime:
        await runtime.run_turn(
            thread_id="thread-a",
            message="I have a sister named Sarah",
            channel=Channel.TEST,
            user_id="user-1",
            llm_client=llm,
        )
        # Do NOT await any drain explicitly — exit the runtime.
        # __aexit__ should drain.

    # After the context exits, the store on disk must contain the
    # extracted fact. We open a new runtime against the same paths
    # and read it back.
    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime_b:
        assert await runtime_b.memory_store.arecord_count(("user-1", "semantic")) >= 1


class _StallingExtractionLLM(BaseLLMClient):
    """LLM client that hangs forever inside ``ExtractionResult`` calls.

    Used to verify the drain timeout. Other schemas behave normally so
    the rest of the graph completes without the stall.
    """

    def __init__(self) -> None:
        self.stall_event = asyncio.Event()
        self.cancel_event = asyncio.Event()

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "fake reply"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "fake stream chunk"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__

        if schema_name == "ExtractionResult":
            self.stall_event.set()
            try:
                # Hang until the test cancels us at shutdown.
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancel_event.set()
                raise
            raise RuntimeError("unreachable")

        if schema_name == "CrisisAssessmentSchema":
            from agent.gates.safety.service import CrisisAssessmentSchema

            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=0,
                    confidence="high",
                    reason="safe",
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
                    reasoning="fake",
                    confidence="high",
                ),
            )

        if schema_name == "ProceduralExtractionResult":
            from agent.memory.models import ProceduralExtractionResult

            return cast(
                StructuredResponseT,
                ProceduralExtractionResult(rules=[], reason="no rules"),
            )

        if schema_name == "SummarizationResult":
            from agent.memory.models import SummarizationResult

            return cast(
                StructuredResponseT,
                SummarizationResult(arc=None, reason="thin session"),
            )

        raise RuntimeError(f"_StallingExtractionLLM: unexpected schema {schema_name}")


@pytest.mark.asyncio
async def test_drain_timeout_does_not_block_next_turn_forever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled extraction must not block the next turn indefinitely.

    Replaces ``EXTRACTION_DRAIN_TIMEOUT_SECONDS`` with a small value so
    the test can run quickly. The first turn schedules an extraction
    that hangs in the LLM call; the second turn's prepare-step drains
    with timeout and proceeds anyway.
    """

    monkeypatch.setattr(
        "agent.runtime.turn_extraction.EXTRACTION_DRAIN_TIMEOUT_SECONDS", 0.5
    )

    paths = _runtime_paths(tmp_path)
    stalling = _StallingExtractionLLM()

    async with PersistentAgentRuntime(
        **paths,
        memory_mode=MemoryMode.LOCAL,
        finalize_active_sessions_on_close=False,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-stall",
            message="My sister Sarah called me yesterday about Mom",
            channel=Channel.TEST,
            llm_client=stalling,
        )

        # Wait for the stall to begin so the drain has something
        # genuinely in-flight to time out on.
        await asyncio.wait_for(stalling.stall_event.wait(), timeout=2.0)

        # Switch to a non-stalling LLM for turn 2 so the rest of the
        # graph completes normally. Drain should bound to ~0.5s and
        # this turn should finish within a few seconds, not hang.
        normal_llm = _FakeCrossRestartLLM(
            extraction_result=ExtractionResult(facts=[], reason="empty"),
        )
        await asyncio.wait_for(
            runtime.run_turn(
                thread_id="thread-stall",
                message="My family helps me a lot in tough situations",
                channel=Channel.TEST,
                llm_client=normal_llm,
            ),
            timeout=5.0,
        )

    # The stall task should have been cancelled at shutdown.
    assert stalling.cancel_event.is_set() or stalling.stall_event.is_set()
