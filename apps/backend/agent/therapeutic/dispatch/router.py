"""Framework-agnostic therapeutic routing policy.

Simplified LLM-primary dispatch: the LLM classifier decides response style
and therapeutic approach for every turn. The only non-LLM logic is exercise
state bookkeeping (clearing exercise_state when the LLM routes away from an
active exercise).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent.state import AgentState
from agent.therapeutic.dispatch.classifier import (
    _pick_response_style_and_approach_with_llm,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchPlan:
    """Routing plan produced before LangGraph command construction."""

    response_style: str
    therapeutic_approach: str
    clear_exercise: bool = False
    source: str = "llm_primary"
    reason: str = "LLM classifier selected this response."
    confidence: str | None = None


async def plan_therapeutic_route(
    state: AgentState,
    llm_client: Any | None,
) -> DispatchPlan:
    """Plan the therapeutic route for a turn.

    The LLM classifier decides response style and therapeutic approach.
    The only additional logic is exercise-state bookkeeping: when an
    exercise is active and the LLM routes to a non-exercise style, the
    exercise state is cleared.

    Args:
        state: The current agent state.
        llm_client: Control-plane LLM client for structured routing.

    Returns:
        A dispatch plan describing response style, therapeutic approach, and
        whether active guided-exercise state should be cleared.
    """

    if llm_client is None:
        logger.warning("therapeutic_dispatch: no LLM client, defaulting to supportive")
        return DispatchPlan(
            response_style="supportive",
            therapeutic_approach="none",
            clear_exercise=False,
            source="fallback",
            reason="No LLM client available; defaulting to supportive.",
        )

    decision = await _pick_response_style_and_approach_with_llm(state, llm_client)

    response_style = decision.response_style
    approach = decision.therapeutic_approach
    reason = decision.reasoning
    confidence = decision.confidence

    logger.debug(
        "therapeutic_dispatch: LLM picked response_style=%s approach=%s",
        response_style,
        approach,
    )

    # Exercise-state bookkeeping.
    exercise_state = state.get("exercise_state", {}) or {}
    exercise_active = (
        exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )

    # Side-turn styles (clarifying, psychoeducation) preserve exercise state
    # so the user can resume the exercise after a brief detour.
    _SIDE_TURN_STYLES = {"clarifying", "psychoeducation", "guided_exercise"}
    clear_exercise = exercise_active and response_style not in _SIDE_TURN_STYLES

    if exercise_active and response_style == "guided_exercise":
        # Continuing the exercise: preserve the pinned approach.
        pinned_approach = exercise_state.get("exercise_therapeutic_approach")
        if pinned_approach:
            approach = pinned_approach
    elif exercise_active and response_style == "clarifying":
        # Clarifying side-turn: use the pinned exercise approach for continuity.
        pinned_approach = exercise_state.get("exercise_therapeutic_approach")
        if pinned_approach:
            approach = pinned_approach
        elif state.get("therapeutic_approach"):
            approach = state["therapeutic_approach"]

    return DispatchPlan(
        response_style=response_style,
        therapeutic_approach=approach,
        clear_exercise=clear_exercise,
        source="llm_primary",
        reason=reason,
        confidence=confidence,
    )
