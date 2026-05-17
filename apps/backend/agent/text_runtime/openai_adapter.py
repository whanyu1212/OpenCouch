"""OpenAI Agents SDK implementation of the text-agent adapter contract."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from agents import Runner
from langchain_core.runnables import RunnableConfig

from agent.active_flow import detect_active_flow
from agent.crisis_branch import (
    build_crisis_resource_lookup_delta,
    crisis_response_delta,
    write_crisis_log,
)
from agent.gates.memory_control.service import execute_memory_control_action
from agent.gates.safety.prompts import (
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)
from agent.gates.safety.turn_gate import assess_crisis_gate
from agent.graph_constants import FINALIZE_TURN_NODE
from agent.models import Channel, MessageRole
from agent.nodes.load_memory import build_load_memory_delta
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentState
from agent.text_runtime.langgraph_adapter import LangGraphTextAgentAdapter
from agent.text_runtime.openai_agents import (
    CRISIS_AGENT_NAME,
    GUIDED_EXERCISE_AGENT_NAME,
    THERAPEUTIC_AGENT_NAME,
    MemoryActionType,
    OpenAITextRunContext,
    build_crisis_response_agent,
    build_guided_exercise_agent,
    build_therapeutic_agent,
)
from agent.text_runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeShadowResult,
    TextRuntimeShadowStatus,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.therapeutic.prompts import (
    build_clarifying_system_prompt,
    build_supportive_system_prompt,
    build_therapeutic_response_prompt,
)
from agent.therapeutic.dispatch import (
    build_therapeutic_dispatch_update,
    plan_therapeutic_route,
)
from agent.therapeutic.exercises.runner import ExerciseRunner
from agent.turn_branches import build_grounded_lookup_delta
from agent.turn_dispatch import (
    TurnDispatchPlan,
    build_turn_dispatch_update,
    plan_turn_route,
)
from llm.base import BaseLLMClient, StructuredResponseT
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
guided-exercise state, persistence, and audit logging.

Operational tools may be attached:
- Call show_saved_memory only when the prompt explicitly requires it or the
  user asks what saved memory contains.
- Call show_memory_status only when the prompt explicitly requires it or the
  user asks whether memory is enabled, how many memories exist, or whether
  proactive recall is on.
- Call mutating memory tools only when the prompt explicitly requires the
  matching action or the user clearly asks to change saved memory.
- Preserve deletion confirmation semantics: prepare deletion first, then
  confirm or cancel only when a pending deletion exists.
- Call answer_grounded_lookup only when the prompt explicitly requires it or
  the user asks for external, source-backed, current, official, factual, or
  resource information.
- Never invent tool results or claim a side effect happened without the tool.
"""

_RUNTIME_CRISIS_INSTRUCTIONS = """\
You are the OpenCouch crisis text specialist for a turn already classified by
the application runtime. The runtime owns crisis assessment, resource lookup,
audit logging, persistence, memory mutation, and guided-exercise state.
Do not reclassify the user or invent crisis resources. Follow the provided
prompt context exactly for either level-1 safety clarification or level-2/3
crisis response.
"""

_RUNTIME_GUIDED_EXERCISE_INSTRUCTIONS = """\
You are the OpenCouch guided exercise text specialist for a turn already
selected by the application runtime. The runtime owns consent, exercise
selection, step state, step classification, exit handling, completion, memory
side effects, and persistence.

Use the runtime-provided exercise skill block and step directive as the source
of truth. Do not offer a menu, start a different exercise, skip steps, add
unsupported steps, or continue an exercise after the runtime says to exit or
complete it.
"""


@dataclass(frozen=True)
class _PreparedTurn:
    state: AgentState
    eligible: bool
    fallback_reason: str = ""
    dispatch_plan: TurnDispatchPlan | None = None


@dataclass(frozen=True)
class _SafeAgentResult:
    response_text: str
    runtime_mode: str
    response_style: str
    sdk_duration_ms: float


class OpenAIAgentsSDKRunner:
    """Small wrapper around the Agents SDK runner for test injection."""

    async def run(
        self,
        *,
        agent: Any,
        input_text: str,
        context: OpenAITextRunContext,
        session: Any | None = None,
    ) -> Any:
        return await Runner.run(
            agent,
            input_text,
            context=context,
            max_turns=3,
            session=session,
        )

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: OpenAITextRunContext,
        session: Any | None = None,
    ) -> Any:
        return Runner.run_streamed(
            agent,
            input_text,
            context=context,
            max_turns=3,
            session=session,
        )


class _OpenAIGuidedExerciseResponseLLM(BaseLLMClient):
    """Response-LLM adapter that routes exercise prose through Agents SDK."""

    def __init__(
        self,
        *,
        runner: OpenAIAgentsSDKRunner,
        model: str,
        run_context: OpenAITextRunContext,
        session: Any | None = None,
    ) -> None:
        self._runner = runner
        self._model = model
        self._run_context = run_context
        self._session = session
        self.last_duration_ms: float | None = None

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        del use_search
        if self._session is not None:
            prompt = _strip_recent_history_from_prompt(prompt)
        run_start = time.monotonic()
        result = await self._runner.run(
            agent=self._build_agent(system_instruction),
            input_text=prompt,
            context=self._run_context,
            session=self._session,
        )
        self.last_duration_ms = elapsed_ms(run_start)
        return _final_output_text(getattr(result, "final_output", None))

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        if self._session is not None:
            prompt = _strip_recent_history_from_prompt(prompt)
        run_start = time.monotonic()
        stream = self._runner.run_streamed(
            agent=self._build_agent(system_instruction),
            input_text=prompt,
            context=self._run_context,
            session=self._session,
        )
        chunks: list[str] = []
        async for sdk_event in stream.stream_events():
            chunk = _chunk_from_sdk_event(sdk_event)
            if chunk:
                chunks.append(chunk)
                yield chunk

        self.last_duration_ms = elapsed_ms(run_start)
        final_text = _final_output_text(
            getattr(stream, "final_output", None),
            fallback="".join(chunks),
        )
        if final_text and not chunks:
            yield final_text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        del prompt, response_schema, system_instruction, use_search
        raise RuntimeError("Guided exercise response adapter does not classify.")

    def _build_agent(self, system_instruction: str | None) -> Any:
        instructions = _RUNTIME_GUIDED_EXERCISE_INSTRUCTIONS
        if system_instruction:
            instructions = f"{instructions}\n\n{system_instruction}"
        return build_guided_exercise_agent(
            model=self._model,
            instructions=instructions,
        )


_DEFAULT_OPENAI_RUNNER = OpenAIAgentsSDKRunner()
_PRIOR_STATE_NOT_PROVIDED = object()


class OpenAITextAgentAdapter:
    """OpenAI text adapter with LangGraph-backed checkpoint persistence."""

    def __init__(
        self,
        *,
        checkpoint_adapter: LangGraphTextAgentAdapter | None = None,
        fallback: LangGraphTextAgentAdapter | None = None,
        runner: OpenAIAgentsSDKRunner | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        if checkpoint_adapter is None:
            if fallback is None:
                raise TypeError("OpenAITextAgentAdapter requires checkpoint_adapter.")
            checkpoint_adapter = fallback
        self._checkpoint_adapter = checkpoint_adapter
        self._runner = runner or _DEFAULT_OPENAI_RUNNER
        self._model = model

    @property
    def checkpoint_workflow(self) -> Any:
        """Return the LangGraph workflow used only for checkpoint persistence."""

        return self._checkpoint_adapter.workflow

    async def get_state(self, config: RunnableConfig) -> AgentState | None:
        """Return the latest checkpointed text state for a thread."""

        return await self._checkpoint_adapter.get_state(config)

    async def update_state(
        self,
        config: RunnableConfig,
        values: Mapping[str, Any],
        *,
        as_node: str | None = None,
    ) -> None:
        """Persist a state update through the checkpoint adapter."""

        await self._checkpoint_adapter.update_state(config, values, as_node=as_node)

    async def run_turn(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> Mapping[str, Any]:
        """Run one turn through the OpenAI text runtime."""

        prepared = await self._prepare_turn(
            initial_state, config=config, context=context
        )
        if not prepared.eligible:
            raise RuntimeError("OpenAI text runtime produced an ineligible turn.")

        crisis_mode = _crisis_runtime_mode(prepared)
        if crisis_mode is not None:
            return await self._run_crisis_turn(
                prepared.state,
                config=config,
                context=context,
                runtime_mode=crisis_mode,
                streamed=False,
                session=session,
            )

        memory_action = _memory_action_type(prepared)
        if memory_action is not None:
            return await self._run_memory_tool_turn(
                prepared.state,
                action_type=memory_action,
                config=config,
                context=context,
                streamed=False,
                session=session,
            )

        grounded_query = _grounded_lookup_query(prepared)
        if grounded_query is not None:
            return await self._run_grounded_lookup_tool_turn(
                prepared.state,
                query=grounded_query,
                config=config,
                context=context,
                streamed=False,
                session=session,
            )

        state, guided_exercise = await self._load_and_prepare_guided_exercise(
            prepared.state,
            context,
        )
        if guided_exercise:
            return await self._run_guided_exercise_turn(
                state,
                config=config,
                context=context,
                streamed=False,
                session=session,
            )

        safe_result = await self._run_safe_agent_turn(
            state,
            config=config,
            context=context,
            session=session,
        )
        return await self._finalize_openai_turn(
            state,
            response_text=safe_result.response_text,
            config=config,
            runtime_mode=safe_result.runtime_mode,
            response_style=safe_result.response_style,
            selected_agent=THERAPEUTIC_AGENT_NAME,
            sdk_duration_ms=safe_result.sdk_duration_ms,
            streamed=False,
        )

    async def run_turn_stream(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        """Run one streaming turn through the OpenAI text runtime."""

        prepared = await self._prepare_turn(
            initial_state, config=config, context=context
        )
        if not prepared.eligible:
            raise RuntimeError("OpenAI text runtime produced an ineligible turn.")

        crisis_mode = _crisis_runtime_mode(prepared)
        if crisis_mode is not None:
            if crisis_mode == "crisis_response":
                yield TextRuntimeStatusEvent(stage="crisis_resource_lookup")
            elif crisis_mode == "crisis_clarification":
                yield TextRuntimeStatusEvent(stage="load_memory")
            async for event in self._run_crisis_turn_stream(
                prepared.state,
                config=config,
                context=context,
                runtime_mode=crisis_mode,
                session=session,
            ):
                yield event
            return

        memory_action = _memory_action_type(prepared)
        if memory_action is not None:
            yield TextRuntimeStatusEvent(stage="memory_control")
            final_state = await self._run_memory_tool_turn(
                prepared.state,
                action_type=memory_action,
                config=config,
                context=context,
                streamed=True,
                session=session,
            )
            yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
            yield TextRuntimeStateEvent(state=final_state)
            return

        grounded_query = _grounded_lookup_query(prepared)
        if grounded_query is not None:
            yield TextRuntimeStatusEvent(stage="grounded_lookup")
            final_state = await self._run_grounded_lookup_tool_turn(
                prepared.state,
                query=grounded_query,
                config=config,
                context=context,
                streamed=True,
                session=session,
            )
            yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
            yield TextRuntimeStateEvent(state=final_state)
            return

        yield TextRuntimeStatusEvent(stage="load_memory")
        state, guided_exercise = await self._load_and_prepare_guided_exercise(
            prepared.state,
            context,
        )
        if guided_exercise:
            async for event in self._run_guided_exercise_turn_stream(
                state,
                config=config,
                context=context,
                session=session,
            ):
                yield event
            return

        run_context = self._run_context_for_state(state, config, context)
        agent = self._build_agent(state)
        input_text = self._input_text_for_state(
            state,
            include_recent_history=session is None,
        )

        yield TextRuntimeStatusEvent(stage="therapeutic")
        run_start = time.monotonic()
        stream = self._runner.run_streamed(
            agent=agent,
            input_text=input_text,
            context=run_context,
            session=session,
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
        safe_result = self._resolve_safe_agent_result(
            state,
            run_context=run_context,
            response_text=response_text,
            sdk_duration_ms=elapsed_ms(run_start),
        )
        final_state = await self._finalize_openai_turn(
            state,
            response_text=safe_result.response_text,
            config=config,
            runtime_mode=safe_result.runtime_mode,
            response_style=safe_result.response_style,
            selected_agent=THERAPEUTIC_AGENT_NAME,
            sdk_duration_ms=safe_result.sdk_duration_ms,
            streamed=True,
        )
        yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
        yield TextRuntimeStateEvent(state=final_state)

    async def run_shadow_turn(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        prior_state: AgentState | None = None,
    ) -> TextRuntimeShadowResult:
        """Evaluate the OpenAI path without serving output or writing state."""

        shadow_start = time.monotonic()
        try:
            prepared = await self._prepare_turn(
                initial_state,
                config=config,
                context=context,
                prior_state=prior_state,
            )
            if not prepared.eligible:
                return _shadow_result(
                    prepared,
                    status="fallback",
                    shadow_duration_ms=elapsed_ms(shadow_start),
                )

            crisis_mode = _crisis_runtime_mode(prepared)
            if crisis_mode is not None:
                return _shadow_result(
                    prepared,
                    status="eligible",
                    selected_agent=CRISIS_AGENT_NAME,
                    shadow_duration_ms=elapsed_ms(shadow_start),
                )

            if _memory_action_type(prepared) is not None:
                return _shadow_result(
                    prepared,
                    status="eligible",
                    selected_agent=THERAPEUTIC_AGENT_NAME,
                    shadow_duration_ms=elapsed_ms(shadow_start),
                )

            if _grounded_lookup_query(prepared) is not None:
                return _shadow_result(
                    prepared,
                    status="eligible",
                    selected_agent=THERAPEUTIC_AGENT_NAME,
                    shadow_duration_ms=elapsed_ms(shadow_start),
                )

            state, guided_exercise = await self._load_and_prepare_guided_exercise(
                prepared.state,
                context,
            )
            if guided_exercise:
                return _shadow_result(
                    _PreparedTurn(
                        state=state,
                        eligible=True,
                        dispatch_plan=prepared.dispatch_plan,
                    ),
                    status="eligible",
                    selected_agent=GUIDED_EXERCISE_AGENT_NAME,
                    shadow_duration_ms=elapsed_ms(shadow_start),
                )

            run_context = self._run_context_for_state(state, config, context)
            agent = self._build_shadow_agent(state)
            input_text = self._input_text_for_state(state)

            run_start = time.monotonic()
            result = await self._runner.run(
                agent=agent,
                input_text=input_text,
                context=run_context,
            )
            response_text = _final_output_text(getattr(result, "final_output", None))
            return _shadow_result(
                prepared,
                status="eligible",
                selected_agent=THERAPEUTIC_AGENT_NAME,
                sdk_duration_ms=elapsed_ms(run_start),
                shadow_duration_ms=elapsed_ms(shadow_start),
                response_text=response_text,
            )
        except Exception as exc:  # noqa: BLE001 - shadow must not break serving
            return TextRuntimeShadowResult(
                runtime="openai",
                status="error",
                eligible=False,
                shadow_duration_ms=elapsed_ms(shadow_start),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    async def _prepare_turn(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        prior_state: AgentState | None | object = _PRIOR_STATE_NOT_PROVIDED,
    ) -> _PreparedTurn:
        if prior_state is _PRIOR_STATE_NOT_PROVIDED:
            prior_state = await self.get_state(config)
        state = _effective_turn_state(prior_state, initial_state)

        crisis_result = await assess_crisis_gate(
            state,
            llm_client=context.llm_client,
        )
        _apply_delta(state, crisis_result.delta)
        assessment = crisis_result.assessment
        if assessment.level != 0:
            return _PreparedTurn(
                state=state,
                eligible=True,
                dispatch_plan=None,
            )

        if detect_active_flow(state) == "none":
            _apply_agent_primary_safe_turn_update(state)
            return _PreparedTurn(
                state=state,
                eligible=True,
                dispatch_plan=None,
            )

        dispatch_start = time.monotonic()
        dispatch_plan = await plan_turn_route(
            state,
            llm_client=context.llm_client,
        )
        dispatch_update = build_turn_dispatch_update(
            state,
            dispatch_plan,
            duration_ms=elapsed_ms(dispatch_start),
        )
        diagnostics = dict(dispatch_update.get("diagnostics", {}) or {})
        diagnostics.update(
            {
                "openai_text_dispatch_route": dispatch_plan.route,
                "openai_text_dispatch_confidence": dispatch_plan.confidence,
                "openai_text_dispatch_reason": dispatch_plan.reason,
            }
        )
        dispatch_update["diagnostics"] = diagnostics
        _apply_delta(state, dispatch_update)

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

    async def _load_and_prepare_guided_exercise(
        self,
        state: AgentState,
        context: WorkflowContext,
    ) -> tuple[AgentState, bool]:
        state = await self._load_turn_memory(state, context)
        dispatch_plan = await plan_therapeutic_route(
            state,
            context.llm_client,
        )
        if dispatch_plan.response_style != "guided_exercise":
            return state, False

        dispatch_update = build_therapeutic_dispatch_update(state, dispatch_plan)
        _apply_delta(state, dispatch_update)
        return state, True

    async def _run_memory_tool_turn(
        self,
        state: AgentState,
        *,
        action_type: MemoryActionType,
        config: RunnableConfig,
        context: WorkflowContext,
        streamed: bool,
        session: Any | None = None,
    ) -> AgentState:
        action = _memory_action_payload_from_state(state)
        run_context = self._run_context_for_state(state, config, context)
        agent = self._build_agent(state)
        _, sdk_duration_ms = await self._run_openai_agent_with(
            state,
            agent=agent,
            input_text=self._memory_tool_input_text_for_state(state, action),
            run_context=run_context,
            session=session,
        )
        tool_result = run_context.latest_memory_tool_result(action_type)
        diagnostics: dict[str, Any] = {
            **dict(state.get("diagnostics", {}) or {}),
            "openai_memory_tool_expected": _memory_tool_name(action_type),
            "openai_memory_tool_calls": [
                call.tool_name for call in run_context.memory_tool_calls
            ],
            "openai_memory_tool_side_effects": [
                call.side_effect for call in run_context.memory_tool_calls
            ],
        }

        if tool_result is None:
            fallback_result = await execute_memory_control_action(
                run_context.agent_state_for_memory_action(action),
                context,
            )
            response_text = fallback_result.response_text
            memory_control = fallback_result.memory_control
            procedural_profile = fallback_result.procedural_profile
            diagnostics["openai_memory_tool_fallback"] = True
        else:
            response_text = tool_result.response_text
            memory_control = tool_result.memory_control
            procedural_profile = tool_result.procedural_profile
            diagnostics["openai_memory_tool_fallback"] = False

        delta: dict[str, Any] = {
            "memory_control": memory_control,
            "diagnostics": diagnostics,
        }
        if procedural_profile is not None:
            delta["procedural_profile"] = procedural_profile
        _apply_delta(state, delta)
        return await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            runtime_mode="memory_control",
            response_style="memory_control",
            selected_agent=THERAPEUTIC_AGENT_NAME,
            sdk_duration_ms=sdk_duration_ms,
            streamed=streamed,
        )

    async def _run_grounded_lookup_tool_turn(
        self,
        state: AgentState,
        *,
        query: str,
        config: RunnableConfig,
        context: WorkflowContext,
        streamed: bool,
        session: Any | None = None,
    ) -> AgentState:
        run_context = self._run_context_for_state(state, config, context)
        agent = self._build_agent(state)
        _, sdk_duration_ms = await self._run_openai_agent_with(
            state,
            agent=agent,
            input_text=self._grounded_lookup_input_text_for_state(state, query),
            run_context=run_context,
            session=session,
        )
        tool_result = run_context.latest_grounded_tool_result()
        diagnostics: dict[str, Any] = {
            **dict(state.get("diagnostics", {}) or {}),
            "openai_grounded_tool_expected": "answer_grounded_lookup",
            "openai_grounded_tool_calls": [
                call.tool_name for call in run_context.grounded_tool_calls
            ],
        }

        if tool_result is None:
            fallback_delta = await build_grounded_lookup_delta(state, context)
            _apply_delta(state, fallback_delta)
            response_text = str(state.get("response_text") or "")
            if not response_text:
                raise ValueError("grounded_lookup returned an empty response.")
            diagnostics["openai_grounded_tool_fallback"] = True
            diagnostics.update(dict(state.get("diagnostics", {}) or {}))
            diagnostics["openai_grounded_tool_fallback"] = True
            _apply_delta(state, {"diagnostics": diagnostics})
        else:
            response_text = tool_result.response_text
            diagnostics["openai_grounded_tool_fallback"] = False
            _apply_delta(
                state,
                {
                    "grounded_lookup": tool_result.grounded_lookup,
                    "diagnostics": diagnostics,
                },
            )

        return await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            runtime_mode="grounded_lookup",
            response_style="grounded_lookup",
            selected_agent=THERAPEUTIC_AGENT_NAME,
            sdk_duration_ms=sdk_duration_ms,
            streamed=streamed,
        )

    async def _run_guided_exercise_turn(
        self,
        state: AgentState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        streamed: bool,
        session: Any | None = None,
    ) -> AgentState:
        response_llm = self._guided_exercise_response_llm(
            state,
            config,
            context,
            session=session,
        )
        runner = self._guided_exercise_runner(
            context,
            response_llm=response_llm,
        )
        delta = await runner.run(state)
        _apply_delta(state, delta)
        response_text = str(state.get("response_text") or "")
        if not response_text:
            raise ValueError("guided_exercise returned an empty response.")
        return await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            runtime_mode="guided_exercise",
            response_style=str(state.get("response_style") or "guided_exercise"),
            selected_agent=GUIDED_EXERCISE_AGENT_NAME,
            sdk_duration_ms=response_llm.last_duration_ms,
            streamed=streamed,
        )

    async def _run_guided_exercise_turn_stream(
        self,
        state: AgentState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        yield TextRuntimeStatusEvent(stage="guided_exercise")
        queue: asyncio.Queue[str] = asyncio.Queue()

        def writer_factory() -> Any:
            def writer(payload: dict[str, str]) -> None:
                if payload.get("type") == "chunk":
                    queue.put_nowait(str(payload.get("text") or ""))

            return writer

        response_llm = self._guided_exercise_response_llm(
            state,
            config,
            context,
            session=session,
        )
        runner = self._guided_exercise_runner(
            context,
            response_llm=response_llm,
            stream_writer_factory=writer_factory,
        )
        task = asyncio.create_task(runner.run(state))
        while not task.done() or not queue.empty():
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if chunk:
                yield TextRuntimeChunkEvent(text=chunk)

        delta = await task
        _apply_delta(state, delta)
        response_text = str(state.get("response_text") or "")
        if not response_text:
            raise ValueError("guided_exercise returned an empty response.")
        final_state = await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            runtime_mode="guided_exercise",
            response_style=str(state.get("response_style") or "guided_exercise"),
            selected_agent=GUIDED_EXERCISE_AGENT_NAME,
            sdk_duration_ms=response_llm.last_duration_ms,
            streamed=True,
        )
        yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
        yield TextRuntimeStateEvent(state=final_state)

    def _guided_exercise_response_llm(
        self,
        state: AgentState,
        config: RunnableConfig,
        context: WorkflowContext,
        *,
        session: Any | None = None,
    ) -> _OpenAIGuidedExerciseResponseLLM:
        return _OpenAIGuidedExerciseResponseLLM(
            runner=self._runner,
            model=self._model,
            run_context=self._run_context_for_state(state, config, context),
            session=session,
        )

    @staticmethod
    def _guided_exercise_runner(
        context: WorkflowContext,
        *,
        response_llm: BaseLLMClient,
        stream_writer_factory: Any | None = None,
    ) -> ExerciseRunner:
        kwargs: dict[str, Any] = {}
        if stream_writer_factory is not None:
            kwargs["stream_writer_factory"] = stream_writer_factory
        return ExerciseRunner(
            classifier_llm=context.llm_client,
            response_llm=response_llm,
            memory_store=context.memory_store,
            memory_mode=context.memory_mode,
            **kwargs,
        )

    async def _run_crisis_turn(
        self,
        state: AgentState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        runtime_mode: str,
        streamed: bool,
        session: Any | None = None,
    ) -> AgentState:
        if runtime_mode == "crisis_response":
            lookup_delta = await build_crisis_resource_lookup_delta(state, context)
            _apply_delta(state, lookup_delta)
        elif runtime_mode == "crisis_clarification":
            state = await self._load_turn_memory(state, context)
        else:
            raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

        run_context = self._run_context_for_state(state, config, context)
        agent = self._build_crisis_agent(state, runtime_mode=runtime_mode)
        input_text = self._crisis_input_text_for_state(
            state,
            runtime_mode=runtime_mode,
            include_recent_history=session is None,
        )
        response_text, sdk_duration_ms = await self._run_openai_agent_with(
            state,
            agent=agent,
            input_text=input_text,
            run_context=run_context,
            session=session,
        )

        response_style = _response_style_for_crisis_mode(runtime_mode)
        if runtime_mode == "crisis_response":
            _apply_delta(state, crisis_response_delta(response_text))
            await write_crisis_log(state, context)
        else:
            _apply_delta(
                state,
                {
                    "route": "therapeutic",
                    "response_style": response_style,
                    "response_text": response_text,
                },
            )

        return await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            runtime_mode=runtime_mode,
            response_style=response_style,
            selected_agent=CRISIS_AGENT_NAME,
            sdk_duration_ms=sdk_duration_ms,
            streamed=streamed,
        )

    async def _run_crisis_turn_stream(
        self,
        state: AgentState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        runtime_mode: str,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        if runtime_mode == "crisis_response":
            lookup_delta = await build_crisis_resource_lookup_delta(state, context)
            _apply_delta(state, lookup_delta)
        elif runtime_mode == "crisis_clarification":
            state = await self._load_turn_memory(state, context)
        else:
            raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

        run_context = self._run_context_for_state(state, config, context)
        agent = self._build_crisis_agent(state, runtime_mode=runtime_mode)
        input_text = self._crisis_input_text_for_state(
            state,
            runtime_mode=runtime_mode,
            include_recent_history=session is None,
        )

        yield TextRuntimeStatusEvent(stage=runtime_mode)
        run_start = time.monotonic()
        stream = self._runner.run_streamed(
            agent=agent,
            input_text=input_text,
            context=run_context,
            session=session,
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
        sdk_duration_ms = elapsed_ms(run_start)
        response_style = _response_style_for_crisis_mode(runtime_mode)
        if runtime_mode == "crisis_response":
            _apply_delta(state, crisis_response_delta(response_text))
            yield TextRuntimeStatusEvent(stage="crisis_log")
            await write_crisis_log(state, context)
        else:
            _apply_delta(
                state,
                {
                    "route": "therapeutic",
                    "response_style": response_style,
                    "response_text": response_text,
                },
            )

        final_state = await self._finalize_openai_turn(
            state,
            response_text=response_text,
            config=config,
            runtime_mode=runtime_mode,
            response_style=response_style,
            selected_agent=CRISIS_AGENT_NAME,
            sdk_duration_ms=sdk_duration_ms,
            streamed=True,
        )
        yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
        yield TextRuntimeStateEvent(state=final_state)

    async def _run_openai_agent(
        self,
        state: AgentState,
        run_context: OpenAITextRunContext,
        *,
        session: Any | None = None,
    ) -> str:
        text, _ = await self._run_openai_agent_with(
            state,
            agent=self._build_agent(state),
            input_text=self._input_text_for_state(
                state,
                include_recent_history=session is None,
            ),
            run_context=run_context,
            session=session,
        )
        return text

    async def _run_safe_agent_turn(
        self,
        state: AgentState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> _SafeAgentResult:
        run_context = self._run_context_for_state(state, config, context)
        agent = self._build_agent(state)
        input_text = self._input_text_for_state(
            state,
            include_recent_history=session is None,
        )
        response_text, sdk_duration_ms = await self._run_openai_agent_with(
            state,
            agent=agent,
            input_text=input_text,
            run_context=run_context,
            session=session,
        )
        return self._resolve_safe_agent_result(
            state,
            run_context=run_context,
            response_text=response_text,
            sdk_duration_ms=sdk_duration_ms,
        )

    def _resolve_safe_agent_result(
        self,
        state: AgentState,
        *,
        run_context: OpenAITextRunContext,
        response_text: str,
        sdk_duration_ms: float,
    ) -> _SafeAgentResult:
        runtime_mode, response_style, resolved_text = _merge_safe_agent_tool_results(
            state,
            run_context=run_context,
            response_text=response_text,
        )
        return _SafeAgentResult(
            response_text=resolved_text,
            runtime_mode=runtime_mode,
            response_style=response_style,
            sdk_duration_ms=sdk_duration_ms,
        )

    async def _run_openai_agent_with(
        self,
        state: AgentState,
        *,
        agent: Any,
        input_text: str,
        run_context: OpenAITextRunContext,
        session: Any | None = None,
    ) -> tuple[str, float]:
        run_start = time.monotonic()
        result = await self._runner.run(
            agent=agent,
            input_text=input_text,
            context=run_context,
            session=session,
        )
        text = _final_output_text(getattr(result, "final_output", None))
        sdk_duration_ms = elapsed_ms(run_start)
        diagnostics = dict(state.get("diagnostics", {}))
        diagnostics["openai_sdk_ms"] = round(sdk_duration_ms, 2)
        state["diagnostics"] = diagnostics
        return text, sdk_duration_ms

    async def _finalize_openai_turn(
        self,
        state: AgentState,
        *,
        response_text: str,
        config: RunnableConfig,
        runtime_mode: str,
        response_style: str,
        selected_agent: str | None,
        sdk_duration_ms: float | None,
        streamed: bool,
    ) -> AgentState:
        assistant_turn = {
            "role": MessageRole.ASSISTANT.value,
            "content": response_text,
            "response_style": response_style,
        }
        diagnostics = {
            **dict(state.get("diagnostics", {})),
            "text_agent_runtime": "openai",
            "openai_text_runtime_mode": runtime_mode,
            "openai_selected_agent": selected_agent,
            "openai_streamed": streamed,
        }
        if sdk_duration_ms is not None:
            diagnostics["openai_sdk_ms"] = round(sdk_duration_ms, 2)

        route = _route_for_runtime_mode(runtime_mode)
        final_values: dict[str, Any] = {
            **dict(state),
            "response_text": response_text,
            "response_style": response_style,
            "diagnostics": diagnostics,
            "transcript": [*list(state.get("transcript", [])), assistant_turn],
        }
        if route is not None:
            final_values["route"] = route
        if runtime_mode in {"safe_therapeutic", "crisis_clarification"}:
            final_values.update(
                {
                    "therapeutic_approach": "none",
                    "session_action": "none",
                    "should_persist_memory": False,
                }
            )
        final_state = cast(AgentState, final_values)

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
            instructions=instructions,
        )

    def _build_shadow_agent(self, state: AgentState) -> Any:
        instructions = (
            f"{_RUNTIME_THERAPEUTIC_INSTRUCTIONS}\n\n"
            "Shadow runs must not call tools or create side effects. Produce a "
            "best-effort safe therapeutic reply from the visible prompt only.\n\n"
            f"{build_supportive_system_prompt(state)}"
        )
        return build_therapeutic_agent(
            model=self._model,
            instructions=instructions,
            tools=[],
        )

    def _build_crisis_agent(self, state: AgentState, *, runtime_mode: str) -> Any:
        if runtime_mode == "crisis_response":
            system_prompt = build_crisis_response_system_prompt()
        elif runtime_mode == "crisis_clarification":
            system_prompt = build_clarifying_system_prompt(state)
        else:
            raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

        instructions = f"{_RUNTIME_CRISIS_INSTRUCTIONS}\n\n{system_prompt}"
        return build_crisis_response_agent(
            model=self._model,
            instructions=instructions,
        )

    def _input_text_for_state(
        self,
        state: AgentState,
        *,
        include_recent_history: bool = True,
    ) -> str:
        prompt_state = (
            state if include_recent_history else _state_without_prompt_history(state)
        )
        prompt = build_therapeutic_response_prompt(
            prompt_state,
            response_style="supportive",
        )
        operational_context = _operational_context_for_prompt(state)
        if not operational_context:
            return prompt
        return f"{prompt}\n\n{operational_context}"

    def _memory_tool_input_text_for_state(
        self,
        state: AgentState,
        action: Mapping[str, Any],
    ) -> str:
        action_type = cast(MemoryActionType, str(action.get("type") or ""))
        tool_name = _memory_tool_name(action_type)
        arguments = _memory_tool_arguments(action)
        return (
            "The current user turn is an explicit saved-memory management "
            "request selected by the OpenCouch runtime.\n\n"
            f"Required tool: {tool_name}\n"
            f"Required tool arguments: {json.dumps(arguments, sort_keys=True)}\n"
            "Call the required tool exactly once before answering. Then answer "
            "using only the tool result's response_text. Do not call a different "
            "memory tool. Do not infer or invent memory.\n\n"
            f'Current user message: "{state.get("message", "")}"'
        )

    def _grounded_lookup_input_text_for_state(
        self,
        state: AgentState,
        query: str,
    ) -> str:
        return (
            "The current user turn is an explicit grounded lookup request "
            "selected by the OpenCouch runtime.\n\n"
            "Required tool: answer_grounded_lookup\n"
            f"Required tool arguments: {json.dumps({'query': query}, sort_keys=True)}\n"
            "Call the required tool exactly once before answering. Then answer "
            "using only the tool result's response_text. Do not provide "
            "ungrounded factual claims.\n\n"
            f'Current user message: "{state.get("message", "")}"'
        )

    def _crisis_input_text_for_state(
        self,
        state: AgentState,
        *,
        runtime_mode: str,
        include_recent_history: bool = True,
    ) -> str:
        prompt_state = (
            state if include_recent_history else _state_without_prompt_history(state)
        )
        if runtime_mode == "crisis_response":
            return build_crisis_response_prompt(prompt_state)
        if runtime_mode == "crisis_clarification":
            return build_therapeutic_response_prompt(
                prompt_state,
                response_style="clarifying",
            )
        raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

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


def _state_without_prompt_history(state: AgentState) -> AgentState:
    prompt_state = dict(state)
    prompt_state["transcript"] = []
    prompt_state["history"] = []
    return cast(AgentState, prompt_state)


def _strip_recent_history_from_prompt(prompt: str) -> str:
    marker = "Recent conversation:\n"
    current_marker = "\nCurrent user message:\n"
    start = prompt.find(marker)
    if start == -1:
        return prompt
    history_start = start + len(marker)
    end = prompt.find(current_marker, history_start)
    if end == -1:
        return prompt

    middle = prompt[history_start:end]
    preserved = ""
    for context_marker in (
        "\nRelevant context requested by the user:",
        "\nRelevant context from past sessions:",
        "\nPrivate memory context is available",
    ):
        context_start = middle.find(context_marker)
        if context_start != -1:
            preserved = middle[context_start:]
            break

    replacement = "(conversation history is provided by the SDK session)"
    return f"{prompt[:history_start]}{replacement}{preserved}{prompt[end:]}"


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


def _apply_agent_primary_safe_turn_update(state: AgentState) -> None:
    """Set app-owned safe-turn context without invoking the graph router."""

    _apply_delta(
        state,
        {
            "route": "therapeutic",
            "turn_lifecycle": {"active_flow": "none", "action": "none"},
            "memory_reference": {
                "mode": _memory_reference_mode_for_message(
                    str(state.get("message") or "")
                )
            },
            "diagnostics": {"openai_agent_primary_routing": True},
        },
    )


def _memory_reference_mode_for_message(message: str) -> str:
    text = " ".join(message.lower().split())
    explicit_phrases = (
        "what did we work out",
        "what did we decide",
        "where did we leave off",
        "where we left off",
        "last time we talked",
        "last time we spoke",
        "last session",
        "previous session",
        "what helped last time",
        "continue from last time",
        "continue where we left",
    )
    if any(phrase in text for phrase in explicit_phrases):
        return "explicit"
    return "none"


def _merge_safe_agent_tool_results(
    state: AgentState,
    *,
    run_context: OpenAITextRunContext,
    response_text: str,
) -> tuple[str, str, str]:
    memory_calls = list(run_context.memory_tool_calls)
    grounded_calls = list(run_context.grounded_tool_calls)
    diagnostics: dict[str, Any] = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_agent_primary_routing": True,
    }

    for call in memory_calls:
        delta: dict[str, Any] = {"memory_control": call.memory_control}
        if call.procedural_profile is not None:
            delta["procedural_profile"] = call.procedural_profile
        _apply_delta(state, delta)

    if memory_calls:
        latest_memory_call = memory_calls[-1]
        diagnostics.update(
            {
                "openai_memory_tool_expected": latest_memory_call.tool_name,
                "openai_memory_tool_selected": latest_memory_call.tool_name,
                "openai_memory_tool_calls": [call.tool_name for call in memory_calls],
                "openai_memory_tool_side_effects": [
                    call.side_effect for call in memory_calls
                ],
                "openai_memory_tool_fallback": False,
            }
        )

    for call in grounded_calls:
        _apply_delta(state, {"grounded_lookup": call.grounded_lookup})

    if grounded_calls:
        latest_grounded_call = grounded_calls[-1]
        diagnostics.update(
            {
                "openai_grounded_tool_expected": latest_grounded_call.tool_name,
                "openai_grounded_tool_selected": latest_grounded_call.tool_name,
                "openai_grounded_tool_calls": [
                    call.tool_name for call in grounded_calls
                ],
                "openai_grounded_tool_fallback": False,
            }
        )

    _apply_delta(state, {"diagnostics": diagnostics})

    if grounded_calls:
        _apply_delta(state, {"route": "grounded_lookup"})
        return "grounded_lookup", "grounded_lookup", grounded_calls[-1].response_text
    if memory_calls:
        _apply_delta(state, {"route": "memory_control"})
        return "memory_control", "memory_control", memory_calls[-1].response_text

    _apply_delta(state, {"route": "therapeutic"})
    return "safe_therapeutic", "supportive", response_text


def _route_for_runtime_mode(runtime_mode: str) -> str | None:
    if runtime_mode in {"safe_therapeutic", "crisis_clarification"}:
        return "therapeutic"
    if runtime_mode == "memory_control":
        return "memory_control"
    if runtime_mode == "grounded_lookup":
        return "grounded_lookup"
    if runtime_mode == "crisis_response":
        return "crisis"
    return None


def _operational_context_for_prompt(state: AgentState) -> str:
    lines = [
        "Operational context:",
        "- The current turn has already passed the app-owned crisis gate.",
        "- Decide whether to answer directly or call one attached tool when the "
        "user explicitly asks for saved-memory management or grounded lookup.",
    ]
    memory_control = state.get("memory_control", {}) or {}
    pending_action = (
        memory_control.get("pending_action")
        if isinstance(memory_control, Mapping)
        else None
    )
    if isinstance(pending_action, Mapping):
        preview = ""
        target = pending_action.get("target")
        if isinstance(target, Mapping):
            preview = str(target.get("preview") or "").strip()
        pending_line = (
            "- Pending memory deletion exists. Call confirm_memory_deletion only "
            "if the user clearly confirms; call cancel_memory_deletion only if "
            "the user clearly declines."
        )
        if preview:
            pending_line = f"{pending_line} Target preview: {preview}"
        lines.append(pending_line)

    memory_reference = state.get("memory_reference", {}) or {}
    if (
        isinstance(memory_reference, Mapping)
        and memory_reference.get("mode") == "explicit"
    ):
        lines.append(
            "- The user explicitly asked to use prior conversation context; use "
            "retrieved memory context when it is available."
        )

    return "\n".join(lines)


def _memory_action_type_from_plan(
    plan: TurnDispatchPlan | None,
) -> MemoryActionType | None:
    if plan is None or plan.memory_action is None:
        return None
    action = plan.memory_action.to_state_action()
    action_type = action.get("type")
    if action_type in {
        "list",
        "status",
        "set_recall",
        "save_preference",
        "forget_by_index",
        "forget_by_query",
        "confirm_pending",
        "cancel_pending",
    }:
        return cast(MemoryActionType, action_type)
    return None


def _memory_action_type(
    prepared: _PreparedTurn,
) -> MemoryActionType | None:
    plan = prepared.dispatch_plan
    if plan is None or plan.route != "memory_control":
        return None
    return _memory_action_type_from_plan(plan)


def _grounded_lookup_query(prepared: _PreparedTurn) -> str | None:
    plan = prepared.dispatch_plan
    if plan is None or plan.route != "grounded_lookup":
        return None
    query = (plan.grounded_lookup_query or "").strip()
    return query or None


def _memory_action_payload_from_state(state: AgentState) -> dict[str, Any]:
    memory_control = state.get("memory_control", {}) or {}
    action = (
        memory_control.get("action", {}) if isinstance(memory_control, dict) else {}
    )
    if not isinstance(action, dict) or "type" not in action:
        raise ValueError("memory_control.action requires a type.")
    return dict(action)


def _memory_tool_name(action_type: MemoryActionType) -> str:
    return {
        "list": "show_saved_memory",
        "status": "show_memory_status",
        "set_recall": "set_proactive_memory_recall",
        "save_preference": "save_response_preference",
        "forget_by_index": "prepare_memory_deletion_by_index",
        "forget_by_query": "prepare_memory_deletion_by_query",
        "confirm_pending": "confirm_memory_deletion",
        "cancel_pending": "cancel_memory_deletion",
    }[action_type]


def _memory_tool_arguments(action: Mapping[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("type") or "")
    if action_type in {"list", "status", "confirm_pending", "cancel_pending"}:
        return {}
    if action_type == "set_recall":
        return {"enabled": bool(action.get("enabled"))}
    if action_type == "save_preference":
        return {"preference_text": str(action.get("preference_text") or "")}
    if action_type == "forget_by_index":
        return {
            "target_kind": str(action.get("target_kind") or ""),
            "target_index": int(action.get("target_index") or 0),
        }
    if action_type == "forget_by_query":
        return {"query": str(action.get("query") or "")}
    raise ValueError(f"Unsupported memory tool action: {action_type}")


def _crisis_runtime_mode(prepared: _PreparedTurn) -> str | None:
    crisis = prepared.state.get("crisis")
    if crisis is None:
        return None
    if (
        getattr(crisis, "needs_crisis_response", False)
        or getattr(
            crisis,
            "level",
            0,
        )
        >= 2
    ):
        return "crisis_response"
    if (
        getattr(crisis, "needs_clarification", False)
        or getattr(crisis, "level", 0) == 1
    ):
        return "crisis_clarification"
    return None


def _response_style_for_crisis_mode(runtime_mode: str) -> str:
    if runtime_mode == "crisis_response":
        return "crisis_response"
    if runtime_mode == "crisis_clarification":
        return "clarifying"
    raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")


def _shadow_result(
    prepared: _PreparedTurn,
    *,
    status: TextRuntimeShadowStatus,
    selected_agent: str | None = None,
    sdk_duration_ms: float | None = None,
    shadow_duration_ms: float | None = None,
    response_text: str | None = None,
) -> TextRuntimeShadowResult:
    assessment = prepared.state.get("crisis")
    plan = prepared.dispatch_plan
    memory_action_type = None
    if plan is not None and plan.memory_action is not None:
        memory_action_type = str(plan.memory_action.payload.get("type") or "")
    summary = _response_text_summary(response_text)
    return TextRuntimeShadowResult(
        runtime="openai",
        status=status,
        eligible=prepared.eligible,
        fallback_reason=prepared.fallback_reason or None,
        route=plan.route if plan is not None else prepared.state.get("route"),
        active_flow=plan.active_flow if plan is not None else None,
        active_flow_action=plan.active_flow_action if plan is not None else None,
        memory_reference_mode=(
            plan.memory_reference_mode if plan is not None else None
        ),
        memory_action_type=memory_action_type or None,
        grounded_lookup_query=(
            plan.grounded_lookup_query if plan is not None else None
        ),
        crisis_level=getattr(assessment, "level", None),
        needs_crisis_response=getattr(assessment, "needs_crisis_response", None),
        needs_crisis_clarification=getattr(assessment, "needs_clarification", None),
        selected_agent=selected_agent,
        sdk_duration_ms=sdk_duration_ms,
        shadow_duration_ms=shadow_duration_ms,
        **summary,
    )


def _response_text_summary(text: str | None) -> dict[str, Any]:
    if not text:
        return {
            "response_text_length": None,
            "response_text_preview": None,
            "response_text_sha256": None,
        }
    return {
        "response_text_length": len(text),
        "response_text_preview": text[:160],
        "response_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
    }


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
