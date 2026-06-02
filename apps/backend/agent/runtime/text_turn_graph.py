"""App-owned route planning for OpenAI text-runtime turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from agent.runtime.types import TextRuntimeConfig
from agent.runtime.workflow_context import WorkflowContext
from agent.specialists.crisis import CRISIS_AGENT_NAME
from agent.specialists.guided_exercise import GUIDED_EXERCISE_AGENT_NAME
from agent.specialists.therapeutic import THERAPEUTIC_AGENT_NAME
from agent.state import AgentState, AgentTurnInputState

TextRouteKind = Literal[
    "crisis_response",
    "crisis_clarification",
    "grounded_lookup",
    "guided_exercise",
    "memory_control",
    "therapeutic",
]


@dataclass(frozen=True)
class PreparedTurn:
    """State prepared by deterministic app-owned pre-routing gates."""

    state: AgentState
    eligible: bool
    fallback_reason: str = ""


@dataclass(frozen=True)
class TextRoutePlan:
    """One resolved app-owned branch for a text runtime turn."""

    kind: TextRouteKind
    state: AgentState
    prepared: PreparedTurn
    runtime_mode: str
    response_style: str
    selected_agent: str
    query: str = ""
    stream_status_stages: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextTurnGraphResult:
    """Prepared turn plus optional route plan."""

    prepared: PreparedTurn
    plan: TextRoutePlan | None


LoadAndPrepareGuidedExercise = Callable[
    [AgentState, WorkflowContext],
    Awaitable[tuple[AgentState, bool]],
]


class TextTurnGraph:
    """Resolve app-owned text turn routing before SDK specialist execution."""

    def __init__(
        self,
        *,
        prepare_turn: Callable[..., Awaitable[PreparedTurn]],
        load_and_prepare_guided_exercise: LoadAndPrepareGuidedExercise,
    ) -> None:
        self._prepare_turn = prepare_turn
        self._load_and_prepare_guided_exercise = load_and_prepare_guided_exercise

    async def resolve(
        self,
        initial_state: AgentTurnInputState,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        prior_state: AgentState | None = None,
    ) -> TextTurnGraphResult:
        """Resolve one turn into a deterministic route plan."""

        prepared = await self._prepare_turn(
            initial_state,
            config=config,
            context=context,
            prior_state=prior_state,
        )
        if not prepared.eligible:
            return TextTurnGraphResult(prepared=prepared, plan=None)

        crisis_mode = crisis_runtime_mode_for_state(prepared.state)
        if crisis_mode is not None:
            return TextTurnGraphResult(
                prepared=prepared,
                plan=_crisis_plan(prepared, crisis_mode),
            )

        state = prepared.state
        if state.get("route") == "grounded_lookup":
            return TextTurnGraphResult(
                prepared=prepared,
                plan=TextRoutePlan(
                    kind="grounded_lookup",
                    state=state,
                    prepared=prepared,
                    runtime_mode="grounded_lookup",
                    response_style="grounded_lookup",
                    selected_agent=THERAPEUTIC_AGENT_NAME,
                    query=grounded_lookup_query_for_state(state),
                    stream_status_stages=("grounded_lookup",),
                ),
            )

        state, guided_exercise = await self._load_and_prepare_guided_exercise(
            state,
            context,
        )
        routed_prepared = PreparedTurn(
            state=state,
            eligible=True,
            fallback_reason=prepared.fallback_reason,
        )
        if guided_exercise:
            return TextTurnGraphResult(
                prepared=prepared,
                plan=TextRoutePlan(
                    kind="guided_exercise",
                    state=state,
                    prepared=routed_prepared,
                    runtime_mode="guided_exercise",
                    response_style="guided_exercise",
                    selected_agent=GUIDED_EXERCISE_AGENT_NAME,
                    stream_status_stages=("load_memory",),
                ),
            )

        if state.get("route") == "memory_control":
            return TextTurnGraphResult(
                prepared=prepared,
                plan=TextRoutePlan(
                    kind="memory_control",
                    state=state,
                    prepared=routed_prepared,
                    runtime_mode="memory_control",
                    response_style="memory_control",
                    selected_agent=THERAPEUTIC_AGENT_NAME,
                    stream_status_stages=("load_memory",),
                ),
            )

        return TextTurnGraphResult(
            prepared=prepared,
            plan=TextRoutePlan(
                kind="therapeutic",
                state=state,
                prepared=routed_prepared,
                runtime_mode="safe_therapeutic",
                response_style=str(state.get("response_style") or "supportive"),
                selected_agent=THERAPEUTIC_AGENT_NAME,
                stream_status_stages=("load_memory",),
            ),
        )


def crisis_runtime_mode_for_state(state: AgentState) -> str | None:
    """Return the crisis runtime mode implied by prepared state."""

    crisis = state.get("crisis")
    if crisis is None:
        return None
    if (
        getattr(crisis, "needs_crisis_response", False)
        or getattr(crisis, "level", 0) >= 2
    ):
        return "crisis_response"
    if (
        getattr(crisis, "needs_clarification", False)
        or getattr(crisis, "level", 0) == 1
    ):
        return "crisis_clarification"
    return None


def grounded_lookup_query_for_state(state: AgentState) -> str:
    """Return the grounded lookup query requested by routing state."""

    return str(
        (state.get("grounded_lookup", {}) or {}).get("query")
        or state.get("message")
        or ""
    ).strip()


def _crisis_plan(prepared: PreparedTurn, runtime_mode: str) -> TextRoutePlan:
    if runtime_mode == "crisis_response":
        return TextRoutePlan(
            kind="crisis_response",
            state=prepared.state,
            prepared=prepared,
            runtime_mode=runtime_mode,
            response_style="crisis_response",
            selected_agent=CRISIS_AGENT_NAME,
            stream_status_stages=("crisis_resource_lookup",),
        )
    if runtime_mode == "crisis_clarification":
        return TextRoutePlan(
            kind="crisis_clarification",
            state=prepared.state,
            prepared=prepared,
            runtime_mode=runtime_mode,
            response_style="clarifying",
            selected_agent=CRISIS_AGENT_NAME,
            stream_status_stages=("load_memory",),
        )
    raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")


__all__ = [
    "PreparedTurn",
    "TextRouteKind",
    "TextRoutePlan",
    "TextTurnGraph",
    "TextTurnGraphResult",
    "crisis_runtime_mode_for_state",
    "grounded_lookup_query_for_state",
]
