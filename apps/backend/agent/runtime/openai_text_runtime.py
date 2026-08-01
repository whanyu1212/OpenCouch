"""OpenAI Agents SDK implementation of the text-agent runtime."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

from agents import Runner
from agent.guardrails.prompts import build_crisis_response_prompt
from agent.models import Channel, CrisisAssessment, MessageRole
from agent.observability.decorators import trace_event, trace_span
from agent.observability.events import (
    RUNTIME_TEXT_TURN,
    SDK_OPENAI_CALL,
    SDK_OPENAI_CALL_COMPLETED,
)
from agent.observability.timing import elapsed_ms
from agent.specialists.roster import build_openai_text_agent_roster
from agent.runtime.context import OpenAITextRunContext
from agent.flows.crisis import (
    crisis_resource_tool_input_text_for_state as crisis_resource_prompt_for_state_path,
)
from agent.flows.guided_exercise import prepare_guided_exercise_route
from agent.guardrails import run_crisis_input_guardrail
from agent.runtime.memory_context import build_turn_memory_delta
from agent.runtime.state_ops import (
    apply_state_delta,
    build_effective_turn_state,
    finalize_openai_turn,
)
from agent.runtime.prompt_utils import final_output_text
from agent.runtime.session.history import state_without_prompt_history
from agent.runtime.route_registry import (
    TextRouteRegistryFactory,
    build_default_text_route_registry,
)
from agent.runtime.services import TextRuntimeServices
from agent.runtime.text_turn_graph import (
    PreparedTurn,
    TextRoutePlan,
    TextTurnGraph,
    TextTurnGraphResult,
)
from agent.runtime.triage_dispatch import apply_triage_turn_dispatch
from agent.runtime.types import (
    TextRuntimeConfig,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState, AgentTurnInputState
from agent.specialists.therapeutic_response.prompts import (
    build_therapeutic_response_prompt,
)
from agent.specialists.therapeutic_response.runtime_prompts import (
    build_therapeutic_agent_input,
)
from llm.openai_client import DEFAULT_OPENAI_MODEL


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

    async def run_triage(
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
            max_turns=1,
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


_DEFAULT_OPENAI_RUNNER = OpenAIAgentsSDKRunner()
_PRIOR_STATE_NOT_PROVIDED = object()


def _drain_prefetched_memory(context: WorkflowContext) -> None:
    """Cancel/retrieve any speculative memory prefetch at turn end.

    The prefetch task is scheduled per-turn but only consumed by routes that
    call ``load_turn_memory`` (therapeutic). Routes that skip memory load —
    crisis_response and grounded_lookup — would otherwise orphan the task,
    leaking a held memory-store connection and an unretrieved exception. Draining
    here makes "prefetch is always settled by turn end" an invariant independent
    of route. ``cancel_if_pending`` is idempotent, so this is a safe no-op when
    the prefetch was already consumed or never scheduled. Runs at teardown, off
    the response path, so it adds no turn latency.
    """

    pre_fetched = context.pre_fetched_memory
    if pre_fetched is not None:
        pre_fetched.cancel_if_pending()


class OpenAITextRuntime:
    """OpenAI Agents SDK text runtime."""

    def __init__(
        self,
        *,
        runner: OpenAIAgentsSDKRunner | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        route_registry_factory: TextRouteRegistryFactory = (
            build_default_text_route_registry
        ),
    ) -> None:
        self._runner = runner or _DEFAULT_OPENAI_RUNNER
        self._model = model
        self._roster = build_openai_text_agent_roster(model=model)
        self._route_registry = route_registry_factory(self._services)
        self._turn_graph = TextTurnGraph(
            prepare_turn=self._prepare_turn,
            load_and_prepare_guided_exercise=self._load_and_prepare_guided_exercise,
        )

    def _services(self) -> TextRuntimeServices:
        return TextRuntimeServices(
            runner=self._runner,
            roster=self._roster,
            build_run_context=self._run_context_for_state,
            build_agent=self._build_agent,
            input_text_for_state=self._input_text_for_state,
            crisis_input_text_for_state=self._crisis_input_text_for_state,
            run_openai_agent_with=self._run_openai_agent_with,
            finalize_turn=self._finalize_openai_turn,
            load_turn_memory=self._load_turn_memory,
        )

    @trace_span(RUNTIME_TEXT_TURN, attrs={"runtime_mode": "text", "streamed": False})
    async def run_turn(
        self,
        initial_state: AgentTurnInputState,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
        prior_state: AgentState | None = None,
    ) -> Mapping[str, Any]:
        """Run one turn through the OpenAI text runtime."""

        # Wraps the whole body — including the no-LLM smoke return — so the
        # prefetch is drained on every exit path. The runtime schedules the
        # prefetch independently of llm_client, so a deterministic (no-LLM) turn
        # can still carry a live prefetch task that must not be orphaned.
        try:
            if context.llm_client is None:
                return _deterministic_smoke_state(
                    initial_state,
                    prior_state=prior_state,
                    streamed=False,
                )

            route_result = await self._turn_graph.resolve(
                initial_state,
                config=config,
                context=context,
                prior_state=prior_state,
            )
            plan = self._require_route_plan(route_result)
            self._apply_route_plan_diagnostics(plan)
            return await self._execute_route_plan(
                plan,
                config=config,
                context=context,
                session=session,
            )
        finally:
            _drain_prefetched_memory(context)

    @trace_span(RUNTIME_TEXT_TURN, attrs={"runtime_mode": "text", "streamed": True})
    async def run_turn_stream(
        self,
        initial_state: AgentTurnInputState,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
        prior_state: AgentState | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        """Run one streaming turn through the OpenAI text runtime."""

        try:
            if context.llm_client is None:
                yield TextRuntimeStatusEvent(stage="deterministic")
                final_state = _deterministic_smoke_state(
                    initial_state,
                    prior_state=prior_state,
                    streamed=True,
                )
                yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
                yield TextRuntimeStateEvent(state=final_state)
                return

            route_result = await self._turn_graph.resolve(
                initial_state,
                config=config,
                context=context,
                prior_state=prior_state,
            )
            plan = self._require_route_plan(route_result)
            self._apply_route_plan_diagnostics(plan)
            for stage in plan.stream_status_stages:
                yield TextRuntimeStatusEvent(stage=stage)
            async for event in self._stream_route_plan(
                plan,
                config=config,
                context=context,
                session=session,
            ):
                yield event
        finally:
            # Fires on normal completion, the no-LLM smoke return, AND early
            # consumer abandonment (aclose()/GeneratorExit), so the prefetch is
            # drained on every exit path.
            _drain_prefetched_memory(context)

    def _require_route_plan(self, result: TextTurnGraphResult) -> TextRoutePlan:
        if result.plan is None:
            raise RuntimeError("OpenAI text runtime produced an ineligible turn.")
        return result.plan

    def _apply_route_plan_diagnostics(self, plan: TextRoutePlan) -> None:
        apply_state_delta(
            plan.state,
            {
                "diagnostics": {
                    "openai_text_route_plan_kind": plan.kind,
                    "openai_text_route_plan_runtime_mode": plan.runtime_mode,
                    "openai_text_route_plan_selected_agent": plan.selected_agent,
                }
            },
        )

    async def _execute_route_plan(
        self,
        plan: TextRoutePlan,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AgentState:
        return await self._route_registry.handler_for(plan.kind).execute(
            plan,
            config=config,
            context=context,
            session=session,
        )

    async def _stream_route_plan(
        self,
        plan: TextRoutePlan,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        async for event in self._route_registry.handler_for(plan.kind).stream(
            plan,
            config=config,
            context=context,
            session=session,
        ):
            yield event

    async def _prepare_turn(
        self,
        initial_state: AgentTurnInputState,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        prior_state: AgentState | None | object = _PRIOR_STATE_NOT_PROVIDED,
    ) -> PreparedTurn:
        effective_prior_state = (
            None
            if prior_state is _PRIOR_STATE_NOT_PROVIDED
            else cast(AgentState | None, prior_state)
        )
        state = _effective_turn_state(effective_prior_state, initial_state)

        run_context = self._run_context_for_state(state, config, context)
        guardrail_output = await run_crisis_input_guardrail(
            agent=self._build_agent(state),
            input_text=str(state.get("message") or ""),
            context=run_context,
        )
        apply_state_delta(state, dict(guardrail_output.delta))
        assessment = guardrail_output.assessment
        if assessment.level != 0:
            return PreparedTurn(
                state=state,
                eligible=True,
            )

        state = await self._apply_triage_turn_dispatch(
            state,
            config=config,
            context=context,
        )
        return PreparedTurn(
            state=state,
            eligible=True,
        )

    async def _apply_triage_turn_dispatch(
        self,
        state: AgentState,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
    ) -> AgentState:
        return await apply_triage_turn_dispatch(
            state,
            config=config,
            context=context,
            run_context_factory=lambda: self._run_context_for_state(
                state,
                config,
                context,
            ),
            runner=self._runner,
            triage_agent=self._roster.triage_agent,
        )

    async def _load_turn_memory(
        self,
        state: AgentState,
        context: WorkflowContext,
    ) -> AgentState:
        load_delta = await build_turn_memory_delta(state, context)
        apply_state_delta(state, dict(load_delta))
        return state

    async def _load_and_prepare_guided_exercise(
        self,
        state: AgentState,
        context: WorkflowContext,
    ) -> tuple[AgentState, bool]:
        return await prepare_guided_exercise_route(
            state,
            context,
            load_turn_memory=self._load_turn_memory,
        )

    @trace_span(
        SDK_OPENAI_CALL,
        attrs=lambda args, kwargs: {
            "model": args[0]._model,
            "agent_name": getattr(kwargs.get("agent"), "name", None),
        },
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
        text = final_output_text(getattr(result, "final_output", None))
        sdk_duration_ms = elapsed_ms(run_start)
        trace_event(
            SDK_OPENAI_CALL_COMPLETED,
            {
                "duration_ms": sdk_duration_ms,
                "response_text_length": len(text),
            },
        )
        diagnostics = dict(state.get("diagnostics", {}))
        diagnostics["openai_sdk_ms"] = round(sdk_duration_ms, 2)
        state["diagnostics"] = diagnostics
        return text, sdk_duration_ms

    async def _finalize_openai_turn(
        self,
        state: AgentState,
        *,
        response_text: str,
        config: TextRuntimeConfig,
        runtime_mode: str,
        response_style: str,
        selected_agent: str | None,
        sdk_duration_ms: float | None,
        streamed: bool,
    ) -> AgentState:
        del config
        return finalize_openai_turn(
            state,
            response_text=response_text,
            runtime_mode=runtime_mode,
            response_style=response_style,
            selected_agent=selected_agent,
            sdk_duration_ms=sdk_duration_ms,
            streamed=streamed,
        )

    def _build_agent(self, state: AgentState) -> Any:
        del state
        return self._roster.therapeutic_agent

    def _input_text_for_state(
        self,
        state: AgentState,
        *,
        include_recent_history: bool = True,
        prompt_appendix: str | None = None,
    ) -> str:
        return build_therapeutic_agent_input(
            state,
            include_recent_history=include_recent_history,
            prompt_appendix=prompt_appendix,
        )

    def _crisis_input_text_for_state(
        self,
        state: AgentState,
        *,
        runtime_mode: str,
        include_recent_history: bool = True,
        require_resource_tool: bool = False,
    ) -> str:
        prompt_state = (
            state if include_recent_history else state_without_prompt_history(state)
        )
        if runtime_mode == "crisis_response":
            if require_resource_tool:
                return crisis_resource_prompt_for_state_path(prompt_state)
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
        config: TextRuntimeConfig,
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
            agent_state=state,
            installed_skills=list(state.get("installed_skills", [])),
            transcript=[
                dict(turn)
                for turn in list(state.get("transcript", []) or [])
                if isinstance(turn, Mapping)
            ],
            turn_count=int(session_progress.get("turn_count", 0) or 0),
        )


def _effective_turn_state(
    prior_state: AgentState | None,
    initial_state: AgentTurnInputState,
) -> AgentState:
    return build_effective_turn_state(prior_state, initial_state)


def _deterministic_smoke_state(
    initial_state: AgentTurnInputState,
    *,
    prior_state: AgentState | None,
    streamed: bool,
) -> AgentState:
    """Return a local-only final state for deterministic CLI smoke runs."""

    state = _effective_turn_state(prior_state, initial_state)
    response_text = _deterministic_smoke_response_text(state)
    response_style = "deterministic_smoke"
    assistant_turn = {
        "role": MessageRole.ASSISTANT.value,
        "content": response_text,
        "response_style": response_style,
    }
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "text_agent_runtime": "deterministic_smoke",
        "openai_text_runtime_mode": "deterministic_smoke",
        "openai_selected_agent": None,
        "openai_streamed": streamed,
        "deterministic_smoke": True,
        "finalize_done_at_monotonic": time.monotonic(),
        "routing_trace": [
            {
                "stage": "runtime",
                "decision": "deterministic_smoke",
                "source": "no_llm_client",
                "reason": "No LLM client is configured for this turn.",
            }
        ],
    }
    return cast(
        AgentState,
        {
            **dict(state),
            "response_text": response_text,
            "response_style": response_style,
            "therapeutic_approach": "none",
            "session_action": "none",
            "should_persist_memory": False,
            "route": "therapeutic",
            "crisis": CrisisAssessment(
                reason=("Deterministic smoke mode did not run crisis classification."),
            ),
            "diagnostics": diagnostics,
            "turn_lifecycle": {
                "active_flow": "none",
                "action": "none",
            },
            "grounded_lookup": {
                "query": "",
                "status": "not_attempted",
            },
            "memory_reference": {
                "mode": "none",
            },
            "transcript": [
                *list(state.get("transcript", [])),
                assistant_turn,
            ],
        },
    )


def _deterministic_smoke_response_text(state: Mapping[str, Any]) -> str:
    """Return the deterministic user-facing smoke response text."""

    message = str(state.get("message") or "").strip()
    suffix = f" Received message: {message}" if message else ""
    return (
        "Deterministic smoke mode: kept this turn local because no LLM client "
        "is configured. Crisis classification, therapeutic generation, and "
        f"memory extraction did not run.{suffix}"
    )


def _thread_id_from_config(config: TextRuntimeConfig, state: Mapping[str, Any]) -> str:
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
