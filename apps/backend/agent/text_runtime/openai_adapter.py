"""OpenAI Agents SDK implementation of the text-agent adapter contract."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from agents import Runner
from langchain_core.runnables import RunnableConfig

from agent.gates.safety.turn_gate import assess_crisis_gate
from agent.graph_constants import FINALIZE_TURN_NODE
from agent.models import Channel, MessageRole
from agent.nodes.load_memory import build_load_memory_delta
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentState
from agent.text_runtime.langgraph_adapter import LangGraphTextAgentAdapter
from agent.text_runtime.openai_agents import (
    THERAPEUTIC_AGENT_NAME,
    OpenAITextRunContext,
    build_therapeutic_agent,
)
from agent.text_runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.therapeutic.prompts import (
    build_supportive_system_prompt,
    build_therapeutic_response_prompt,
)
from agent.turn_dispatch import TurnDispatchPlan, plan_turn_route
from llm.openai_client import DEFAULT_OPENAI_MODEL


_DICT_REDUCER_KEYS = {
    "session_memory",
    "procedural_profile",
    "session_progress",
    "exercise_state",
    "memory_control",
    "grounded_lookup",
    "diagnostics",
}

_RUNTIME_THERAPEUTIC_INSTRUCTIONS = """\
You are the OpenCouch therapeutic text agent for an already-classified safe
turn. The application runtime owns crisis assessment, memory mutation,
grounded lookup, guided-exercise state, persistence, and audit logging.
No SDK tools are attached in this migration slice; answer only the current
safe therapeutic turn using the provided prompt context.
"""


@dataclass(frozen=True)
class _PreparedTurn:
    state: AgentState
    eligible: bool
    fallback_reason: str = ""
    dispatch_plan: TurnDispatchPlan | None = None


class OpenAIAgentsSDKRunner:
    """Small wrapper around the Agents SDK runner for test injection."""

    async def run(
        self,
        *,
        agent: Any,
        input_text: str,
        context: OpenAITextRunContext,
    ) -> Any:
        return await Runner.run(
            agent,
            input_text,
            context=context,
            max_turns=3,
        )

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: OpenAITextRunContext,
    ) -> Any:
        return Runner.run_streamed(
            agent,
            input_text,
            context=context,
            max_turns=3,
        )


_DEFAULT_OPENAI_RUNNER = OpenAIAgentsSDKRunner()


class OpenAITextAgentAdapter:
    """Hybrid OpenAI text adapter with LangGraph fallback for unsupported turns."""

    def __init__(
        self,
        *,
        fallback: LangGraphTextAgentAdapter,
        runner: OpenAIAgentsSDKRunner | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        self._fallback = fallback
        self._runner = runner or _DEFAULT_OPENAI_RUNNER
        self._model = model

    @property
    def checkpoint_workflow(self) -> Any:
        """Return the LangGraph workflow used only for checkpoint persistence."""

        return self._fallback.workflow

    async def get_state(self, config: RunnableConfig) -> AgentState | None:
        """Return the latest checkpointed text state for a thread."""

        return await self._fallback.get_state(config)

    async def update_state(
        self,
        config: RunnableConfig,
        values: Mapping[str, Any],
        *,
        as_node: str | None = None,
    ) -> None:
        """Persist a state update through the fallback checkpointer."""

        await self._fallback.update_state(config, values, as_node=as_node)

    async def run_turn(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
    ) -> Mapping[str, Any]:
        """Run one turn through OpenAI when safe, otherwise delegate to LangGraph."""

        prepared = await self._prepare_turn(
            initial_state, config=config, context=context
        )
        if not prepared.eligible:
            return await self._fallback.run_turn(
                initial_state,
                config=config,
                context=context,
            )

        state = await self._load_turn_memory(prepared.state, context)
        run_context = self._run_context_for_state(state, config, context)
        response_text = await self._run_openai_agent(state, run_context)
        return await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            sdk_duration_ms=None,
            streamed=False,
        )

    async def run_turn_stream(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        """Run one streaming turn through OpenAI when safe, else delegate."""

        prepared = await self._prepare_turn(
            initial_state, config=config, context=context
        )
        if not prepared.eligible:
            async for event in self._fallback.run_turn_stream(
                initial_state,
                config=config,
                context=context,
            ):
                yield event
            return

        yield TextRuntimeStatusEvent(stage="load_memory")
        state = await self._load_turn_memory(prepared.state, context)
        run_context = self._run_context_for_state(state, config, context)
        agent = self._build_agent(state)
        input_text = self._input_text_for_state(state)

        yield TextRuntimeStatusEvent(stage="therapeutic")
        run_start = time.monotonic()
        stream = self._runner.run_streamed(
            agent=agent,
            input_text=input_text,
            context=run_context,
        )
        chunks: list[str] = []
        async for sdk_event in stream.stream_events():
            chunk = _chunk_from_sdk_event(sdk_event)
            if chunk:
                chunks.append(chunk)
                yield TextRuntimeChunkEvent(text=chunk)

        response_text = _final_output_text(
            getattr(stream, "final_output", None),
            fallback="".join(chunks),
        )
        final_state = await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            sdk_duration_ms=elapsed_ms(run_start),
            streamed=True,
        )
        yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
        yield TextRuntimeStateEvent(state=final_state)

    async def _prepare_turn(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
    ) -> _PreparedTurn:
        prior_state = await self.get_state(config)
        state = _effective_turn_state(prior_state, initial_state)

        crisis_result = await assess_crisis_gate(
            state,
            llm_client=context.llm_client,
        )
        _apply_delta(state, crisis_result.delta)
        assessment = crisis_result.assessment
        if (
            assessment.level != 0
            or assessment.needs_crisis_response
            or assessment.needs_clarification
        ):
            return _PreparedTurn(
                state=state,
                eligible=False,
                fallback_reason="crisis_or_safety_clarification",
            )

        dispatch_plan = await plan_turn_route(
            state,
            llm_client=context.llm_client,
        )
        _apply_delta(
            state,
            {
                "route": dispatch_plan.route,
                "turn_lifecycle": {
                    "active_flow": dispatch_plan.active_flow,
                    "action": dispatch_plan.active_flow_action,
                },
                "memory_reference": {"mode": dispatch_plan.memory_reference_mode},
                "diagnostics": {
                    "openai_text_dispatch_route": dispatch_plan.route,
                    "openai_text_dispatch_confidence": dispatch_plan.confidence,
                    "openai_text_dispatch_reason": dispatch_plan.reason,
                },
            },
        )

        fallback_reason = _fallback_reason(dispatch_plan)
        if fallback_reason:
            return _PreparedTurn(
                state=state,
                eligible=False,
                fallback_reason=fallback_reason,
                dispatch_plan=dispatch_plan,
            )

        return _PreparedTurn(
            state=state,
            eligible=True,
            dispatch_plan=dispatch_plan,
        )

    async def _load_turn_memory(
        self,
        state: AgentState,
        context: WorkflowContext,
    ) -> AgentState:
        load_delta = await build_load_memory_delta(state, context)
        _apply_delta(state, load_delta)
        return state

    async def _run_openai_agent(
        self,
        state: AgentState,
        run_context: OpenAITextRunContext,
    ) -> str:
        run_start = time.monotonic()
        result = await self._runner.run(
            agent=self._build_agent(state),
            input_text=self._input_text_for_state(state),
            context=run_context,
        )
        text = _final_output_text(getattr(result, "final_output", None))
        diagnostics = dict(state.get("diagnostics", {}))
        diagnostics["openai_sdk_ms"] = round(elapsed_ms(run_start), 2)
        state["diagnostics"] = diagnostics
        return text

    async def _finalize_openai_turn(
        self,
        state: AgentState,
        *,
        response_text: str,
        config: RunnableConfig,
        sdk_duration_ms: float | None,
        streamed: bool,
    ) -> AgentState:
        assistant_turn = {
            "role": MessageRole.ASSISTANT.value,
            "content": response_text,
            "response_style": "supportive",
        }
        diagnostics = {
            **dict(state.get("diagnostics", {})),
            "text_agent_runtime": "openai",
            "openai_text_runtime_mode": "safe_therapeutic",
            "openai_selected_agent": THERAPEUTIC_AGENT_NAME,
            "openai_streamed": streamed,
        }
        if sdk_duration_ms is not None:
            diagnostics["openai_sdk_ms"] = round(sdk_duration_ms, 2)

        final_state = cast(
            AgentState,
            {
                **dict(state),
                "response_text": response_text,
                "response_style": "supportive",
                "therapeutic_approach": "none",
                "session_action": "none",
                "should_persist_memory": False,
                "diagnostics": diagnostics,
                "transcript": [*list(state.get("transcript", [])), assistant_turn],
            },
        )

        checkpoint_delta = dict(final_state)
        checkpoint_delta["transcript"] = [
            {
                "role": MessageRole.USER.value,
                "content": state.get("message", ""),
            },
            assistant_turn,
        ]
        await self.update_state(
            config,
            checkpoint_delta,
            as_node=FINALIZE_TURN_NODE,
        )
        persisted = await self.get_state(config)
        return persisted or final_state

    def _build_agent(self, state: AgentState) -> Any:
        instructions = (
            f"{_RUNTIME_THERAPEUTIC_INSTRUCTIONS}\n\n"
            f"{build_supportive_system_prompt(state)}"
        )
        return build_therapeutic_agent(
            model=self._model,
            tools=[],
            instructions=instructions,
        )

    def _input_text_for_state(self, state: AgentState) -> str:
        return build_therapeutic_response_prompt(
            state,
            response_style="supportive",
        )

    def _run_context_for_state(
        self,
        state: AgentState,
        config: RunnableConfig,
        context: WorkflowContext,
    ) -> OpenAITextRunContext:
        memory_control = state.get("memory_control", {}) or {}
        session_progress = state.get("session_progress", {}) or {}
        return OpenAITextRunContext(
            thread_id=_thread_id_from_config(config, state),
            workflow_context=context,
            current_user_message=state.get("message", ""),
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
            channel=_channel_from_state(state),
            pending_memory_action=memory_control.get("pending_action"),
            installed_skills=list(state.get("installed_skills", [])),
            turn_count=int(session_progress.get("turn_count", 0) or 0),
        )


def _effective_turn_state(
    prior_state: AgentState | None,
    initial_state: AgentGraphInputState,
) -> AgentState:
    if prior_state is None:
        return cast(AgentState, dict(initial_state))

    state: dict[str, Any] = dict(prior_state)
    for key, value in dict(initial_state).items():
        if key == "transcript":
            state[key] = [
                *list(prior_state.get("transcript", [])),
                *list(value or []),
            ]
        elif key in _DICT_REDUCER_KEYS:
            state[key] = {
                **dict(prior_state.get(key, {}) or {}),
                **dict(value or {}),
            }
        else:
            state[key] = value
    return cast(AgentState, state)


def _apply_delta(state: AgentState, delta: Mapping[str, Any]) -> None:
    for key, value in delta.items():
        if key in _DICT_REDUCER_KEYS:
            state[key] = cast(
                Any,
                {
                    **dict(state.get(key, {}) or {}),
                    **dict(value or {}),
                },
            )
        else:
            state[key] = cast(Any, value)


def _fallback_reason(plan: TurnDispatchPlan) -> str:
    if plan.route != "therapeutic":
        return f"unsupported_route:{plan.route}"
    if plan.active_flow != "none" or plan.active_flow_action != "none":
        return f"active_flow:{plan.active_flow}:{plan.active_flow_action}"
    if plan.memory_reference_mode != "none":
        return f"memory_reference:{plan.memory_reference_mode}"
    return ""


def _thread_id_from_config(config: RunnableConfig, state: Mapping[str, Any]) -> str:
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    thread_id = (
        configurable.get("thread_id") if isinstance(configurable, dict) else None
    )
    return str(thread_id or state.get("session_id") or "openai-text-thread")


def _channel_from_state(state: Mapping[str, Any]) -> Channel:
    channel = state.get("channel")
    if isinstance(channel, Channel):
        return channel
    try:
        return Channel(str(channel))
    except ValueError:
        return Channel.TEST


def _final_output_text(output: Any, *, fallback: str = "") -> str:
    text = output if isinstance(output, str) and output else str(output or fallback)
    if not text:
        raise ValueError("OpenAI Agents SDK returned an empty text response.")
    return text


def _chunk_from_sdk_event(event: Any) -> str | None:
    if getattr(event, "type", None) != "raw_response_event":
        return None

    data = getattr(event, "data", None)
    event_type = (
        data.get("type") if isinstance(data, dict) else getattr(data, "type", None)
    )
    if event_type != "response.output_text.delta":
        return None

    delta = (
        data.get("delta") if isinstance(data, dict) else getattr(data, "delta", None)
    )
    return str(delta or "") or None
