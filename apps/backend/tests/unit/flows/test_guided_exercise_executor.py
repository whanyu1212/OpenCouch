"""Tests for the guided-exercise streaming executor's task lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agent.flows.guided_exercise import executor as executor_module
from agent.flows.guided_exercise.executor import run_guided_exercise_turn_stream
from agent.runtime.types import TextRuntimeStatusEvent


class _BlockingSkillService:
    """Skill service that streams one chunk then blocks until cancelled.

    The chunk lets the executor's drain loop emit one event (advancing the
    generator past task creation, into the loop); the subsequent block keeps the
    producer task alive so consumer abandonment must cancel it.
    """

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self.cancelled = asyncio.Event()

    async def run_turn(self, state: Any) -> dict[str, Any]:
        self._writer({"type": "chunk", "text": "first chunk"})
        try:
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return {}


class _StubResponseLLM:
    run_context = object()
    used_skill_tool_fallback = False
    last_duration_ms = 0.0


@pytest.mark.asyncio
async def test_stream_cancels_producer_task_on_consumer_abandonment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression for #163: if the consumer abandons the stream early (aclose()),
    # the drain loop exits before `await task`. The producer (skill_service.run_turn)
    # must be cancelled in the finally, not orphaned with an unretrieved exception.
    captured: dict[str, Any] = {}

    def _fake_skill_service(*_a: Any, stream_writer_factory: Any, **_k: Any) -> Any:
        # The executor passes a writer_factory; build the writer and hand it to a
        # blocking skill service that streams one chunk then waits to be cancelled.
        writer = stream_writer_factory()
        service = _BlockingSkillService(writer)
        captured["service"] = service
        return service

    monkeypatch.setattr(
        executor_module,
        "guided_exercise_response_llm",
        lambda *a, **k: _StubResponseLLM(),
    )
    monkeypatch.setattr(
        executor_module, "guided_exercise_skill_service", _fake_skill_service
    )

    gen = run_guided_exercise_turn_stream(
        cast(Any, object()),  # services — unused once factories are stubbed
        {"message": "start breathing"},
        config={},
        context=cast(Any, object()),
    )

    # First event is the status; second is the chunk (proves we entered the drain
    # loop, i.e. the producer task was created). Then abandon the stream.
    first = await gen.__anext__()
    assert first == TextRuntimeStatusEvent(stage="guided_exercise")
    second = await gen.__anext__()  # the "first chunk" emitted by the producer
    assert getattr(second, "text", None) == "first chunk"

    await gen.aclose()

    # The producer task must have been cancelled by the finally, not left pending.
    skill_service = captured["service"]
    await asyncio.wait_for(skill_service.cancelled.wait(), timeout=1.0)
    assert skill_service.cancelled.is_set()
