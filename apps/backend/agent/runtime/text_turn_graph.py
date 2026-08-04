"""App-owned route planning for OpenAI text-runtime turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
CrisisTextRouteKind = Literal["crisis_response", "crisis_clarification"]


@dataclass(frozen=True)
class _TextRouteSpec:
    runtime_mode: str
    response_style: str | None
    selected_agent: str
    stream_status_stages: tuple[str, ...]


_TEXT_ROUTE_SPECS: dict[TextRouteKind, _TextRouteSpec] = {
    "crisis_response": _TextRouteSpec(
        runtime_mode="crisis_response",
        response_style="crisis_response",
        selected_agent=CRISIS_AGENT_NAME,
        stream_status_stages=("crisis_resource_lookup",),
    ),
    "crisis_clarification": _TextRouteSpec(
        runtime_mode="crisis_clarification",
        response_style="clarifying",
        selected_agent=CRISIS_AGENT_NAME,
        stream_status_stages=("load_memory",),
    ),
    "grounded_lookup": _TextRouteSpec(
        runtime_mode="grounded_lookup",
        response_style="grounded_lookup",
        selected_agent=THERAPEUTIC_AGENT_NAME,
        stream_status_stages=("grounded_lookup",),
    ),
    "guided_exercise": _TextRouteSpec(
        runtime_mode="guided_exercise",
        response_style="guided_exercise",
        selected_agent=GUIDED_EXERCISE_AGENT_NAME,
        stream_status_stages=("load_memory",),
    ),
    "memory_control": _TextRouteSpec(
        runtime_mode="memory_control",
        response_style="memory_control",
        selected_agent=THERAPEUTIC_AGENT_NAME,
        stream_status_stages=("load_memory",),
    ),
    "therapeutic": _TextRouteSpec(
        runtime_mode="safe_therapeutic",
        response_style=None,
        selected_agent=THERAPEUTIC_AGENT_NAME,
        stream_status_stages=("load_memory",),
    ),
}


@dataclass(frozen=True)
class PreparedTurn:
    """State prepared by deterministic app-owned pre-routing gates."""

    state: AgentState
    eligible: bool
    fallback_reason: str = ""


@dataclass(frozen=True)
class TextRoutePlan:
    """One resolved app-owned branch with metadata derived from its route kind."""

    kind: TextRouteKind
    prepared: PreparedTurn
    state: AgentState = field(init=False)
    runtime_mode: str = field(init=False)
    response_style: str = field(init=False)
    selected_agent: str = field(init=False)
    query: str = field(init=False)
    stream_status_stages: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        state = self.prepared.state
        spec = _TEXT_ROUTE_SPECS[self.kind]
        response_style = spec.response_style or str(
            state.get("response_style") or "supportive"
        )

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "runtime_mode", spec.runtime_mode)
        object.__setattr__(self, "response_style", response_style)
        object.__setattr__(self, "selected_agent", spec.selected_agent)
        object.__setattr__(
            self,
            "query",
            grounded_lookup_query_for_state(state)
            if self.kind == "grounded_lookup"
            else "",
        )
        object.__setattr__(
            self,
            "stream_status_stages",
            spec.stream_status_stages,
        )


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

        crisis_kind = crisis_runtime_mode_for_state(prepared.state)
        if crisis_kind is not None:
            return TextTurnGraphResult(
                prepared=prepared,
                plan=TextRoutePlan(kind=crisis_kind, prepared=prepared),
            )

        state = prepared.state
        if state.get("route") == "grounded_lookup":
            return TextTurnGraphResult(
                prepared=prepared,
                plan=TextRoutePlan(kind="grounded_lookup", prepared=prepared),
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
                plan=TextRoutePlan(kind="guided_exercise", prepared=routed_prepared),
            )

        if state.get("route") == "memory_control":
            return TextTurnGraphResult(
                prepared=prepared,
                plan=TextRoutePlan(kind="memory_control", prepared=routed_prepared),
            )

        return TextTurnGraphResult(
            prepared=prepared,
            plan=TextRoutePlan(kind="therapeutic", prepared=routed_prepared),
        )


def crisis_runtime_mode_for_state(
    state: AgentState,
) -> CrisisTextRouteKind | None:
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


__all__ = [
    "PreparedTurn",
    "TextRouteKind",
    "TextRoutePlan",
    "TextTurnGraph",
    "TextTurnGraphResult",
    "crisis_runtime_mode_for_state",
    "grounded_lookup_query_for_state",
]
