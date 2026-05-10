"""Framework-agnostic therapeutic dispatch planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agent.active_flow import current_turn_active_flow
from agent.memory.models import (
    ConfidenceLevel,
    DispatchDecision,
    ExerciseStartBasis,
    TherapeuticApproach,
    TherapeuticResponseStyle,
)
from agent.state import AgentState
from agent.therapeutic.dispatch.prompt import (
    build_therapeutic_dispatch_prompt,
    build_therapeutic_dispatch_system_prompt,
)

# Response styles that preserve active exercise state across a side turn.
# Keep this narrow: explanatory or reflective turns can mean the user has
# switched away from the exercise and should clear exercise continuity.
_EXERCISE_PRESERVING_STYLES: frozenset[TherapeuticResponseStyle] = frozenset(
    {"clarifying", "guided_exercise"}
)
_AUTHORIZED_EXERCISE_START_BASES: frozenset[ExerciseStartBasis] = frozenset(
    {"explicit_user_request", "accepted_assistant_offer"}
)
_CONSENT_GATE_RESPONSE_STYLE: TherapeuticResponseStyle = "supportive"


@dataclass(frozen=True)
class DispatchPlan:
    """Routing plan produced before LangGraph command construction."""

    response_style: TherapeuticResponseStyle
    therapeutic_approach: TherapeuticApproach
    clear_exercise: bool = False
    source: str = "llm_primary"
    reason: str = "LLM classifier selected this response."
    confidence: ConfidenceLevel | None = None
    exercise_start_basis: ExerciseStartBasis | None = None


async def plan_therapeutic_route(
    state: AgentState,
    llm_client: Any | None,
) -> DispatchPlan:
    """Plan the therapeutic route for a turn.

    The LLM classifier decides response style and therapeutic approach. The
    only additional logic is exercise-state bookkeeping: when an exercise is
    active and the LLM routes to a style that should not preserve exercise
    continuity, the exercise state is cleared. When an exercise is active and
    the LLM keeps the user inside it or asks a clarifying side question, the
    exercise's pinned therapeutic approach is reused for continuity.

    Args:
        state: The current agent state.
        llm_client: Control-plane LLM client for structured routing.

    Returns:
        A dispatch plan describing response style, therapeutic approach, and
        whether active guided-exercise state should be cleared.
    """

    # Exercise-state bookkeeping.
    exercise_state = state.get("exercise_state", {}) or {}
    exercise_active = (
        exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )
    pinned_approach = (
        exercise_state.get("exercise_therapeutic_approach") if exercise_active else None
    )
    active_flow = current_turn_active_flow(state)

    if (
        exercise_active
        and active_flow.active_flow == "guided_exercise"
        and active_flow.action in {"continue", "resume"}
    ):
        approach = cast(
            TherapeuticApproach,
            pinned_approach or state.get("therapeutic_approach") or "none",
        )
        return DispatchPlan(
            response_style="guided_exercise",
            therapeutic_approach=approach,
            clear_exercise=False,
            source="active_flow",
            reason=f"Turn dispatch marked the active exercise as {active_flow.action}.",
            confidence="high",
        )

    if llm_client is None:
        raise RuntimeError("Therapeutic dispatch requires a classifier LLM.")

    decision: DispatchDecision = await llm_client.generate_structured(
        prompt=build_therapeutic_dispatch_prompt(state),
        response_schema=DispatchDecision,
        system_instruction=build_therapeutic_dispatch_system_prompt(),
    )

    response_style = decision.response_style
    approach = decision.therapeutic_approach
    source = "llm_primary"
    reason = decision.reasoning
    exercise_start_basis = decision.exercise_start_basis

    if (
        response_style == "guided_exercise"
        and exercise_start_basis not in _AUTHORIZED_EXERCISE_START_BASES
    ):
        response_style = _CONSENT_GATE_RESPONSE_STYLE
        source = "exercise_consent_gate"
        reason = (
            "Guided exercise requires an explicit user request or acceptance "
            "of a specific assistant offer."
        )

    clear_exercise = (
        exercise_active
        and active_flow.action != "preserve"
        and response_style not in _EXERCISE_PRESERVING_STYLES
    )

    if exercise_active and response_style == "guided_exercise" and pinned_approach:
        # Continuing the exercise: preserve the pinned approach.
        approach = pinned_approach
    elif exercise_active and response_style == "clarifying":
        # Clarifying side-turn: prefer the pinned exercise approach, fall
        # back to the prior turn's therapeutic_approach for continuity.
        approach = pinned_approach or state.get("therapeutic_approach") or approach

    return DispatchPlan(
        response_style=response_style,
        therapeutic_approach=approach,
        clear_exercise=clear_exercise,
        source=source,
        reason=reason,
        confidence=decision.confidence,
        exercise_start_basis=exercise_start_basis,
    )
