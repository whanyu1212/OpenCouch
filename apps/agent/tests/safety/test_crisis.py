"""The crisis responder writes a checked reply, or sends the fallback (HY-230)."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import timedelta

import pytest
from ag_ui.core import BaseEvent, RunStartedEvent
from pydantic_ai import CancellationToken
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from opencouch_agent.agent import CompanionDeps
from opencouch_agent.safety import crisis as crisis_module
from opencouch_agent.safety.crisis import (
    CRISIS_FALLBACK,
    IMMINENT_INSTRUCTIONS,
    CrisisResponder,
    build_crisis_agent,
    crisis_events,
    crisis_reply,
)
from opencouch_agent.safety.crisis_reply_check import allowed_numbers, reply_problem
from opencouch_agent.safety.crisis_resources import load_shipped_catalog

CATALOG = load_shipped_catalog()
HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart("i'm going to take all my pills tonight")])
]


def _responder(model: FunctionModel) -> CrisisResponder:
    return CrisisResponder.from_catalog(build_crisis_agent(model), CATALOG)


def _replying(text: str, seen: list[AgentInfo] | None = None) -> FunctionModel:
    def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append(info)
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(reply)


async def _reply(model: FunctionModel, *, imminent: bool = False) -> str:
    return await crisis_reply(
        _responder(model),
        HISTORY,
        deps=CompanionDeps(),
        imminent=imminent,
        cancellation=CancellationToken(),
    )


async def test_the_crisis_responder_offers_the_model_no_tools() -> None:
    seen: list[AgentInfo] = []

    await _reply(_replying("Are you safe right now?", seen))

    assert seen[0].function_tools == []
    assert seen[0].output_tools == []


async def test_a_reply_that_passes_the_check_is_sent_as_written() -> None:
    text = "I'm worried about you. Please call SOS on 1767. Are you safe right now?"

    assert await _reply(_replying(text)) == text


@pytest.mark.parametrize(
    "text",
    ["Call or text 988 now.", "Message them at wa.me/6591511767.", ""],
)
async def test_a_reply_that_fails_the_check_is_replaced_by_the_fallback(
    text: str,
) -> None:
    assert await _reply(_replying(text)) == CRISIS_FALLBACK


async def test_a_model_error_sends_the_fallback() -> None:
    def fail(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        raise RuntimeError("model unavailable")

    assert await _reply(FunctionModel(fail)) == CRISIS_FALLBACK


async def test_a_slow_model_sends_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crisis_module, "REPLY_TIMEOUT", timedelta(seconds=0.05))

    async def stall(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(parts=[TextPart("too late")])

    assert await _reply(FunctionModel(stall)) == CRISIS_FALLBACK


async def test_a_reply_needing_a_second_model_call_sends_the_fallback() -> None:
    calls = 0

    def call_a_tool(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("lookup", "{}", "call-1")])

    assert await _reply(FunctionModel(call_a_tool)) == CRISIS_FALLBACK
    assert calls == 1


async def test_a_cancelled_turn_sends_nothing() -> None:
    cancellation = CancellationToken()
    cancellation.cancel()

    with pytest.raises(RunCancelled):
        await crisis_reply(
            _responder(_replying("Are you safe right now?")),
            HISTORY,
            deps=CompanionDeps(),
            imminent=False,
            cancellation=cancellation,
        )


async def test_the_model_is_given_the_vetted_services_and_their_numbers() -> None:
    seen: list[AgentInfo] = []

    await _reply(_replying("Are you safe right now?", seen))

    instructions = seen[0].instructions or ""
    assert "Samaritans of Singapore (SOS)" in instructions
    assert "1767" in instructions
    assert "995" in instructions
    assert IMMINENT_INSTRUCTIONS.strip() not in instructions


async def test_imminent_danger_puts_the_emergency_number_first() -> None:
    seen: list[AgentInfo] = []

    await _reply(_replying("Call 995 now. Are you safe?", seen), imminent=True)

    instructions = seen[0].instructions or ""
    assert IMMINENT_INSTRUCTIONS.strip() in instructions
    assert instructions.index("Emergency services") < instructions.index("SOS")


def test_the_fallback_passes_the_check() -> None:
    assert reply_problem(CRISIS_FALLBACK, allowed_numbers(CATALOG)) is None


async def test_the_reply_is_sent_whole_and_saved_before_the_run_finishes() -> None:
    order: list[str] = []

    async def write() -> str:
        return "Are you safe right now?"

    async def save(text: str) -> None:
        order.append(f"saved: {text}")

    async for event in crisis_events(
        write, thread_id="thread-1", run_id="run-1", save_reply=save
    ):
        order.append(event.type)

    assert order == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "saved: Are you safe right now?",
        "RUN_FINISHED",
    ]


async def test_nothing_is_saved_if_the_turn_ends_before_the_reply_is_written() -> None:
    saved: list[str] = []

    async def never_finishes() -> str:
        await asyncio.sleep(5)
        return "unreachable"

    async def save(text: str) -> None:
        saved.append(text)

    events: AsyncGenerator[BaseEvent] = crisis_events(
        never_finishes, thread_id="thread-1", run_id="run-1", save_reply=save
    )
    first = await anext(events)
    await events.aclose()

    assert isinstance(first, RunStartedEvent)
    assert saved == []


async def test_a_turn_cancelled_while_writing_ends_quietly_with_nothing_saved() -> None:
    saved: list[str] = []

    async def cancelled() -> str:
        raise RunCancelled("the client went away")

    async def save(text: str) -> None:
        saved.append(text)

    events = [
        event.type
        async for event in crisis_events(
            cancelled, thread_id="thread-1", run_id="run-1", save_reply=save
        )
    ]

    assert events == ["RUN_STARTED"]
    assert saved == []
