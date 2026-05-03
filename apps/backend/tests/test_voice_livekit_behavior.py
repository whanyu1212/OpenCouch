"""LiveKit behavior evals for the OpenCouch voice agent.

These tests use LiveKit's text-only ``AgentSession.run`` testing path.
They do not exercise a LiveKit room, STT, VAD, or audio output. The
goal is to verify the voice agent's LiveKit-native event behavior:
tool calls, tool outputs, multi-turn userdata continuity, and handoffs.
"""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

pytest.importorskip(
    "livekit.agents",
    reason="LiveKit behavior tests require the optional voice extra.",
)

from livekit.agents import AgentSession
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FunctionToolCall,
    LLM,
    LLMStream,
    Tool,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from agent.memory.procedural import (
    aadd_procedural_rule,
    aget_procedural_profile,
    aset_proactive_recall,
    build_procedural_rule,
)
from agent.memory.store import OpenCouchMemoryStore
from voice.livekit.agent import CrisisAgent, TherapeuticAgent
from voice.livekit.session_data import SessionData
from voice.realtime import build_voice_system_prompt


class FakeLLMResponse(BaseModel):
    """Scripted response for one LiveKit test input."""

    type: Literal["llm"] = "llm"
    input: str
    content: str = ""
    ttft: float = 0.0
    duration: float = 0.0
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)

    def speed_up(self, factor: float) -> FakeLLMResponse:
        """Return a faster copy of this response.

        Args:
            factor: Speed multiplier.

        Returns:
            A copied response with shorter timing.
        """

        obj = copy.deepcopy(self)
        obj.ttft /= factor
        obj.duration /= factor
        return obj


class FakeLLM(LLM):
    """Minimal deterministic LLM compatible with ``AgentSession.run``."""

    def __init__(self, *, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__()
        self.fake_response_map = {
            response.input: response for response in fake_responses
        }

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        """Return a scripted stream for the latest chat context item.

        Args:
            chat_ctx: LiveKit chat context.
            tools: Tools available to the model.
            conn_options: Connection options required by the LLM interface.
            parallel_tool_calls: Unused tool-call mode.
            tool_choice: Unused tool-choice hint.
            extra_kwargs: Unused provider options.

        Returns:
            A deterministic fake LLM stream.
        """

        return FakeLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class FakeLLMStream(LLMStream):
    """Stream scripted chunks into LiveKit's run-result recorder."""

    def __init__(
        self,
        llm: FakeLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm = llm

    async def _run(self) -> None:
        start_time = time.perf_counter()
        response = self._llm.fake_response_map.get(self._get_index_text())
        if response is None:
            return

        await asyncio.sleep(response.ttft)
        if response.content:
            self._send_chunk(delta=response.content)

        remaining = response.duration - (time.perf_counter() - start_time)
        if remaining > 0:
            await asyncio.sleep(remaining)

        if response.tool_calls:
            self._send_chunk(tool_calls=response.tool_calls)

    def _send_chunk(
        self,
        *,
        delta: str | None = None,
        tool_calls: list[FunctionToolCall] | None = None,
    ) -> None:
        """Emit one fake chat chunk.

        Args:
            delta: Optional assistant text.
            tool_calls: Optional tool calls.

        Returns:
            None: Sends the chunk into LiveKit's event channel.
        """

        self._event_ch.send_nowait(
            ChatChunk(
                id=str(id(self)),
                delta=ChoiceDelta(
                    role="assistant",
                    content=delta,
                    tool_calls=tool_calls or [],
                ),
            )
        )

    def _get_index_text(self) -> str:
        """Return the lookup key for the current fake response.

        Returns:
            The latest user/system message text or function-call output.
        """

        if not self.chat_ctx.items:
            return ""

        item = self.chat_ctx.items[-1]
        if item.type == "message" and item.role in {"user", "system"}:
            return item.text_content or ""
        if item.type == "function_call_output":
            return item.output or ""
        return ""


class FakeLookupLLM:
    """Deterministic control LLM for search-backed voice tools."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        """Return the next scripted lookup response.

        Args:
            prompt: Prompt sent by the lookup helper.
            system_instruction: Optional system instruction.
            use_search: Whether provider-native search was requested.

        Returns:
            Scripted text response.
        """

        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _tool_call(name: str, arguments: str = "{}") -> FunctionToolCall:
    """Build a deterministic tool call.

    Args:
        name: Tool name.
        arguments: JSON argument string.

    Returns:
        LiveKit function-tool call object.
    """

    return FunctionToolCall(name=name, arguments=arguments, call_id=f"call-{name}")


async def _seed_voice_memory(store: OpenCouchMemoryStore) -> None:
    """Seed saved memory for behavior evals.

    Args:
        store: Memory store to seed.

    Returns:
        None: Mutates the supplied store.
    """

    await store.aput(
        ("voice-user-1", "semantic"),
        "fact-presentations",
        {
            "evidence_quote": "Presentations make me anxious.",
            "source": "voice_behavior_eval",
            "thread_id": "voice-behavior-eval",
        },
    )
    await aadd_procedural_rule(
        store,
        user_id="voice-user-1",
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep replies short."],
        ),
    )


def _agent() -> TherapeuticAgent:
    """Create the voice therapeutic agent under test.

    Returns:
        Configured LiveKit therapeutic agent.
    """

    return TherapeuticAgent(instructions=build_voice_system_prompt())


def _userdata(store: OpenCouchMemoryStore) -> SessionData:
    """Create shared LiveKit session userdata for behavior evals.

    Args:
        store: Memory store for the session.

    Returns:
        Session data wired to the supplied memory store.
    """

    return SessionData(
        user_id="voice-user-1",
        thread_id="voice-behavior-eval",
        memory_store=store,
    )


def _outputs(result) -> list[str]:
    """Collect function output strings from a LiveKit run result.

    Args:
        result: LiveKit ``RunResult``.

    Returns:
        Function-call output strings.
    """

    return [
        event.item.output
        for event in result.events
        if event.type == "function_call_output"
    ]


@pytest.mark.asyncio
async def test_livekit_behavior_lists_saved_memory_with_tool_call() -> None:
    """A memory question should call the voice memory-list tool."""

    store = OpenCouchMemoryStore()
    await _seed_voice_memory(store)
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="What do you remember about me?",
                tool_calls=[_tool_call("show_saved_memory")],
            )
        ]
    )

    async with AgentSession(llm=fake_llm, userdata=_userdata(store)) as session:
        await session.start(_agent())
        result = await session.run(user_input="What do you remember about me?")

    result.expect.contains_function_call(name="show_saved_memory")
    result.expect.contains_function_call_output(is_error=False)
    outputs = "\n".join(_outputs(result))
    assert "Presentations make me anxious." in outputs
    assert "You prefer shorter responses." in outputs


@pytest.mark.asyncio
async def test_livekit_behavior_turns_proactive_recall_off() -> None:
    """A recall preference should call the recall-toggle tool."""

    store = OpenCouchMemoryStore()
    await aset_proactive_recall(store, user_id="voice-user-1", enabled=True)
    userdata = _userdata(store)
    userdata.proactive_recall_enabled = True
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="Don't bring up past sessions unless I ask.",
                tool_calls=[
                    _tool_call(
                        "set_proactive_memory_recall",
                        '{"enabled": false}',
                    )
                ],
            )
        ]
    )

    async with AgentSession(llm=fake_llm, userdata=userdata) as session:
        await session.start(_agent())
        result = await session.run(
            user_input="Don't bring up past sessions unless I ask."
        )

    profile = await aget_procedural_profile(store, user_id="voice-user-1")
    result.expect.contains_function_call(
        name="set_proactive_memory_recall",
        arguments={"enabled": False},
    )
    assert profile.proactive_recall_enabled is False
    assert userdata.proactive_recall_enabled is False


@pytest.mark.asyncio
async def test_livekit_behavior_answers_grounded_factual_lookup_with_tool() -> None:
    """An explicit factual lookup should call the grounded lookup voice tool."""

    store = OpenCouchMemoryStore()
    lookup_llm = FakeLookupLLM(
        [
            "Singapore's SOS crisis hotline is 1767.\nSources: sos.org.sg",
        ]
    )
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="Can you look up Singapore crisis hotline numbers?",
                tool_calls=[
                    _tool_call(
                        "answer_grounded_factual_lookup",
                        '{"query": "Can you look up Singapore crisis hotline numbers?"}',
                    )
                ],
            )
        ]
    )
    userdata = _userdata(store)
    userdata.llm_client = lookup_llm

    async with AgentSession(llm=fake_llm, userdata=userdata) as session:
        await session.start(_agent())
        result = await session.run(
            user_input="Can you look up Singapore crisis hotline numbers?"
        )

    result.expect.contains_function_call(name="answer_grounded_factual_lookup")
    outputs = "\n".join(_outputs(result))
    assert "1767" in outputs
    assert [call["use_search"] for call in lookup_llm.calls] == [True]


@pytest.mark.asyncio
async def test_livekit_behavior_delete_memory_requires_second_turn_confirmation() -> (
    None
):
    """Saved-memory deletion should preserve pending state across turns."""

    store = OpenCouchMemoryStore()
    await _seed_voice_memory(store)
    userdata = _userdata(store)
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="Forget what you remember about presentations.",
                tool_calls=[
                    _tool_call(
                        "prepare_memory_deletion",
                        '{"query": "presentations"}',
                    )
                ],
            ),
            FakeLLMResponse(
                input="yes delete it",
                tool_calls=[_tool_call("confirm_memory_deletion")],
            ),
        ]
    )

    async with AgentSession(llm=fake_llm, userdata=userdata) as session:
        await session.start(_agent())
        prepare_result = await session.run(
            user_input="Forget what you remember about presentations."
        )
        assert (
            await store.aget(("voice-user-1", "semantic"), "fact-presentations")
            is not None
        )
        assert userdata.pending_memory_delete is not None

        confirm_result = await session.run(user_input="yes delete it")

    prepare_result.expect.contains_function_call(name="prepare_memory_deletion")
    confirm_result.expect.contains_function_call(name="confirm_memory_deletion")
    assert await store.aget(("voice-user-1", "semantic"), "fact-presentations") is None
    assert userdata.pending_memory_delete is None


@pytest.mark.asyncio
async def test_livekit_behavior_generic_anxiety_does_not_call_exercise_tool() -> None:
    """Generic anxiety should stay conversational unless the user consents."""

    store = OpenCouchMemoryStore()
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="I feel anxious and overwhelmed.",
                content="That sounds really heavy to sit with.",
            )
        ]
    )

    async with AgentSession(llm=fake_llm, userdata=_userdata(store)) as session:
        await session.start(_agent())
        result = await session.run(user_input="I feel anxious and overwhelmed.")

    assert not any(
        event.type == "function_call" and event.item.name == "start_grounding_exercise"
        for event in result.events
    )
    result.expect.contains_message(role="assistant")


@pytest.mark.asyncio
async def test_livekit_behavior_crisis_tool_handoffs_to_crisis_agent() -> None:
    """The crisis tool should produce a LiveKit handoff event."""

    store = OpenCouchMemoryStore()
    crisis_on_enter_instruction = (
        "The user may be in crisis. Acknowledge what they "
        "said with empathy and provide crisis resources (988 for US/Canada). "
        "Do not lecture. Just be present. Keep it to two or three sentences."
    )
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="I want to die.",
                tool_calls=[
                    _tool_call("crisis_check", '{"concern": "I want to die."}')
                ],
            ),
            FakeLLMResponse(
                input="Transferring to crisis support",
                content="I'm going to stay with you as we focus on immediate safety.",
            ),
            FakeLLMResponse(
                input=crisis_on_enter_instruction,
                content=(
                    "I'm really glad you told me. If you're in immediate danger, "
                    "call emergency services now; if you're in the US or Canada, "
                    "call or text 988."
                ),
            ),
        ]
    )
    userdata = _userdata(store)

    async with AgentSession(llm=fake_llm, userdata=userdata) as session:
        await session.start(_agent())
        result = await session.run(user_input="I want to die.")

    result.expect.contains_function_call(name="crisis_check")
    result.expect.contains_agent_handoff(new_agent_type=CrisisAgent)
    assert userdata.crisis_level == 2
    assert userdata.max_crisis_level == 2


@pytest.mark.asyncio
async def test_livekit_behavior_crisis_agent_looks_up_local_resources() -> None:
    """The crisis agent should expose verified local resource lookup."""

    store = OpenCouchMemoryStore()
    lookup_llm = FakeLookupLLM(
        [
            "Singapore",
            "Samaritans of Singapore | 1767 | https://www.sos.org.sg",
        ]
    )
    crisis_on_enter_instruction = (
        "The user may be in crisis. Acknowledge what they "
        "said with empathy and provide crisis resources (988 for US/Canada). "
        "Do not lecture. Just be present. Keep it to two or three sentences."
    )
    fake_llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input=crisis_on_enter_instruction,
                content="I'm here with you. If you're in immediate danger, call emergency services now.",
            ),
            FakeLLMResponse(
                input="I'm in Singapore and need a crisis hotline.",
                tool_calls=[
                    _tool_call(
                        "provide_crisis_resources",
                        '{"location_context": "I\\u0027m in Singapore and need a crisis hotline."}',
                    )
                ],
            ),
        ]
    )
    userdata = _userdata(store)
    userdata.llm_client = lookup_llm

    async with AgentSession(llm=fake_llm, userdata=userdata) as session:
        await session.start(CrisisAgent())
        result = await session.run(
            user_input="I'm in Singapore and need a crisis hotline."
        )

    result.expect.contains_function_call(name="provide_crisis_resources")
    outputs = "\n".join(_outputs(result))
    assert "Samaritans of Singapore" in outputs
    assert "1767" in outputs
    assert [call["use_search"] for call in lookup_llm.calls] == [False, True]
