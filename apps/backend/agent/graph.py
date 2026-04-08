"""Minimal agent graph helpers for the MVP kernel.

The initial execution path should stay small and explicit:
- build initial state
- run crisis gate
- choose normal or crisis response
- return a normalized output
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass

from agent.context import (
    build_session_summary,
    extract_active_concerns,
    extract_open_loops,
    infer_current_goal,
    trim_history,
    update_session_intent,
)
from agent.models import (
    AgentInput,
    AgentOutput,
    Channel,
    ChunkEvent,
    CrisisAssessment,
    DoneEvent,
    ModeType,
    ResponseKind,
    StatusEvent,
    StreamEvent,
)
from pydantic import BaseModel

from agent.nodes.crisis_gate import run_crisis_gate
from agent.nodes.session_stage import update_session_stage
from agent.state import AgentState
from agent.subgraphs import run_crisis_subgraph, run_therapeutic_subgraph
from agent.subgraphs.therapeutic import run_selected_therapeutic_mode
from services.llm.base import BaseLLMClient


def prepare_turn_state(state: AgentState) -> AgentState:
    """Refresh turn-scoped fields from the current message and persisted transcript.

    Args:
        state: The shared state before crisis routing for the current turn.

    Returns:
        The updated state with derived session-context fields refreshed.
    """

    transcript = deepcopy(state.get("transcript", []))
    active_concerns = extract_active_concerns(
        transcript,
        current_message=state["message"],
    )
    current_goal = infer_current_goal(
        transcript,
        current_message=state["message"],
    )
    open_loops = extract_open_loops(
        transcript,
        current_message=state["message"],
    )
    session_intent, session_intent_source = update_session_intent(
        transcript,
        current_message=state["message"],
        existing_intent=state.get("session_intent"),
        existing_source=state.get("session_intent_source"),
    )
    session_summary = build_session_summary(
        transcript,
        current_message=state["message"],
        active_concerns=active_concerns,
        current_goal=current_goal,
    )

    state["channel"] = state.get("channel", Channel.TEST)
    state["installed_skills"] = list(state.get("installed_skills", []))
    state["history"] = trim_history(transcript)
    state["transcript"] = transcript
    state["working_memory"] = list(state.get("working_memory", []))
    state["session_summary"] = session_summary
    state["active_concerns"] = active_concerns
    state["open_loops"] = open_loops
    state["current_goal"] = current_goal
    state["session_intent"] = session_intent
    state["session_intent_source"] = session_intent_source
    state["turn_count"] = (
        sum(1 for turn in transcript if turn.get("role") == "user") + 1
    )
    state["crisis"] = CrisisAssessment()
    state["route"] = "therapeutic"
    state["mode"] = "supportive_conversation"
    state["mode_type"] = ModeType.THERAPEUTIC
    state["active_modalities"] = []
    state["semantic_signals"] = {}
    state["response_guidance"] = ""
    state["response_type"] = ResponseKind.THERAPEUTIC
    state["response_text"] = ""
    state["should_persist_memory"] = False
    state["session_stage"] = state.get("session_stage", "opening")
    state["session_stage_source"] = state.get("session_stage_source")
    state["session_stage_reason"] = state.get("session_stage_reason", "")
    return state


def build_initial_state(agent_input: AgentInput) -> AgentState:
    """Convert external input into the internal state dictionary.

    Args:
        agent_input: External user input for the current turn.

    Returns:
        The initialized agent state for the current turn.
    """

    state = AgentState(
        # Raw user input for the current turn.
        message=agent_input.message,
        # Normalized channel value so nodes do not care how the message arrived.
        channel=agent_input.channel,
        # Reserved for later once auth and persistence wrap the agent kernel.
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        # Skills are passed as simple names first; later the loader can resolve them.
        installed_skills=list(agent_input.installed_skills),
        # Transcript keeps the full prior session so persistent threads can resume cleanly.
        transcript=[message.model_dump(mode="json") for message in agent_input.history],
        # Retrieved profile/graph memory for the current turn.
        working_memory=list(agent_input.working_memory),
        # History/session fields are filled by `prepare_turn_state`.
        history=[],
        session_summary="",
        active_concerns=[],
        open_loops=[],
        current_goal=None,
        session_intent=None,
        session_intent_source=None,
        session_stage="opening",
        session_stage_source=None,
        session_stage_reason="",
        turn_count=1,
        # Turn-scoped fields are reset by `prepare_turn_state`.
        crisis=CrisisAssessment(),
        route="therapeutic",
        mode="supportive_conversation",
        mode_type=ModeType.THERAPEUTIC,
        active_modalities=[],
        semantic_signals={},
        response_guidance="",
        should_persist_memory=False,
    )
    return prepare_turn_state(state)


def finalize_turn_state(state: AgentState) -> AgentState:
    """Append the current turn to the durable transcript and prompt window.

    Args:
        state: The shared agent state after the selected response path has completed.

    Returns:
        The updated state with transcript/history including the current turn.
    """

    transcript = deepcopy(state.get("transcript", state["history"]))
    transcript.append({"role": "user", "content": state["message"]})
    transcript.append({"role": "assistant", "content": state.get("response_text", "")})
    state["transcript"] = transcript
    state["history"] = trim_history(transcript)
    return state


def state_to_output(state: AgentState) -> AgentOutput:
    """Normalize graph state into the public agent output contract.

    Args:
        state: Shared agent state after graph execution completes.

    Returns:
        The normalized public agent output.
    """

    return AgentOutput(
        # Response text is filled by whichever node wins the route decision.
        response_text=state.get("response_text", ""),
        # Falls back to therapeutic to keep the output shape valid during early wiring.
        response_type=state.get("response_type", ResponseKind.THERAPEUTIC),
        # Crisis assessment is always returned so callers can inspect safety behavior.
        crisis=state["crisis"],
        mode=state.get("mode"),
        mode_type=state.get("mode_type"),
        mode_source=state.get("mode_source"),
        should_persist_memory=state.get("should_persist_memory", False),
    )


async def run_agent(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentOutput:
    """Minimal MVP graph entrypoint.

    Args:
        agent_input: External input for the current turn.
        llm_client: Optional provider-backed client for classification and generation.

    Returns:
        The normalized public output for the completed turn.
    """

    final_states: list[AgentState] = []
    async for event in _run_turn_events(
        build_initial_state(agent_input),
        llm_client=llm_client,
        state_sink=final_states,
    ):
        if isinstance(event, DoneEvent):
            return event.output
    raise RuntimeError("Turn runner completed without emitting a DoneEvent.")


# ── Streaming infrastructure ──────────────────────────────────────────────────


@dataclass
class _CapturedCall:
    """Records the arguments of a generate_text call for deferred streaming."""

    prompt: str = ""
    system_instruction: str | None = None
    temperature: float = 0.0
    was_called: bool = False


class _CapturingLLMClient(BaseLLMClient):
    """Proxy that captures generate_text args instead of calling the provider.

    For generate_structured calls, delegates to the real client. For
    generate_text calls, records the arguments and returns an empty string
    so the response node completes normally without making an API call.
    """

    def __init__(self, real_client: BaseLLMClient) -> None:
        self._real = real_client
        self.captured = _CapturedCall()

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        self.captured = _CapturedCall(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            was_called=True,
        )
        return ""

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        yield ""

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> BaseModel:
        return await self._real.generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            temperature=temperature,
        )


async def _run_turn_events(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
    prepare_state: bool = False,
    state_sink: list[AgentState] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run one full turn and yield status/chunk/done events."""

    if prepare_state:
        state = prepare_turn_state(state)

    yield StatusEvent(stage="crisis_gate")
    state = await run_crisis_gate(state, llm_client=llm_client)

    yield StatusEvent(stage="session_stage")
    state = await update_session_stage(state, llm_client=llm_client)

    route = state["route"]
    yield StatusEvent(stage="response_generation", detail=f"route={route}")

    if llm_client is not None:
        capturing = _CapturingLLMClient(llm_client)
        if route == "crisis":
            state = await run_crisis_subgraph(state, llm_client=capturing)
        else:
            state = await run_therapeutic_subgraph(state, llm_client=capturing)

        if capturing.captured.was_called:
            chunks: list[str] = []
            try:
                async for chunk in llm_client.generate_text_stream(
                    prompt=capturing.captured.prompt,
                    system_instruction=capturing.captured.system_instruction,
                    temperature=capturing.captured.temperature,
                ):
                    chunks.append(chunk)
                    yield ChunkEvent(text=chunk)
                state["response_text"] = "".join(chunks)
            except Exception:
                if route == "crisis":
                    state = await run_crisis_subgraph(state, llm_client=None)
                else:
                    state = await run_selected_therapeutic_mode(state, llm_client=None)
    else:
        if route == "crisis":
            state = await run_crisis_subgraph(state, llm_client=None)
        else:
            state = await run_therapeutic_subgraph(state, llm_client=None)

    state = finalize_turn_state(state)
    if state_sink is not None:
        state_sink.append(deepcopy(state))
    yield DoneEvent(output=state_to_output(state))


async def run_agent_stream(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
    state_sink: list[AgentState] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Streaming variant of run_agent that yields events during execution."""

    async for event in _run_turn_events(
        build_initial_state(agent_input),
        llm_client=llm_client,
        state_sink=state_sink,
    ):
        yield event
