"""Framework-agnostic therapeutic routing policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent.state import AgentState
from agent.therapeutic.dispatch.classifier import (
    _pick_response_style_and_approach_with_llm,
)
from agent.therapeutic.dispatch.fallback import pick_therapeutic_response_style
from agent.therapeutic.dispatch.guards import (
    _active_exercise_therapeutic_approach,
    _blocks_unconsented_exercise_start,
    _exercise_lifecycle,
    _is_active_exercise_clarification,
    _is_bare_ack_to_open_question,
    _looks_like_pending_exercise_choice,
    _matches_any,
)
from agent.therapeutic.dispatch.regex_catalog import EXERCISE_EXIT_PATTERNS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchPlan:
    """Routing plan produced before LangGraph command construction."""

    response_style: str
    therapeutic_approach: str
    clear_exercise: bool = False


async def plan_therapeutic_route(
    state: AgentState,
    llm_client: Any | None,
) -> DispatchPlan:
    """Plan the therapeutic route for a turn without depending on LangGraph.

    Args:
        state: The current agent state.
        llm_client: Optional control-plane LLM client for structured routing.

    Returns:
        A dispatch plan describing response style, therapeutic approach, and
        whether active guided-exercise state should be cleared.
    """

    message = state.get("message", "")
    lowered = message.lower()
    exercise_lifecycle = _exercise_lifecycle(state)

    # Honor explicit exercise opt-outs without waiting for the LLM.
    if exercise_lifecycle == "active" and _matches_any(lowered, EXERCISE_EXIT_PATTERNS):
        logger.debug("therapeutic_dispatch: active-exercise exit override")
        return DispatchPlan("supportive", "none", clear_exercise=True)

    if exercise_lifecycle == "pending_choice" and _looks_like_pending_exercise_choice(
        message
    ):
        logger.debug("therapeutic_dispatch: pending exercise selection choice")
        existing_approach = state.get("therapeutic_approach") or "none"
        return DispatchPlan("guided_exercise", existing_approach)

    if exercise_lifecycle == "active" and _is_active_exercise_clarification(message):
        logger.debug("therapeutic_dispatch: active-exercise clarification override")
        existing_approach = _active_exercise_therapeutic_approach(state) or "none"
        return DispatchPlan("clarifying", existing_approach)

    if exercise_lifecycle == "inactive" and _is_bare_ack_to_open_question(
        state, message
    ):
        logger.debug("therapeutic_dispatch: bare acknowledgment needs clarification")
        return DispatchPlan("clarifying", "none")

    if llm_client is not None:
        try:
            response_style, approach = await _pick_response_style_and_approach_with_llm(
                state,
                llm_client,
            )
            logger.debug(
                "therapeutic_dispatch: LLM picked response_style=%s approach=%s",
                response_style,
                approach,
            )

            if exercise_lifecycle == "active":
                if response_style == "guided_exercise":
                    existing_approach = (
                        _active_exercise_therapeutic_approach(state) or approach
                    )
                    return DispatchPlan("guided_exercise", existing_approach)

                if response_style == "clarifying":
                    existing_approach = (
                        _active_exercise_therapeutic_approach(state) or approach
                    )
                    logger.debug(
                        "therapeutic_dispatch: mid-exercise clarifying "
                        "(exercise state preserved, approach=%s)",
                        existing_approach,
                    )
                    return DispatchPlan("clarifying", existing_approach)

                if response_style == "psychoeducation":
                    logger.debug(
                        "therapeutic_dispatch: mid-exercise psychoeducation "
                        "(exercise state preserved, approach=%s)",
                        approach,
                    )
                    return DispatchPlan("psychoeducation", approach)

                logger.debug(
                    "therapeutic_dispatch: LLM exit from active exercise -> %s",
                    response_style,
                )
                return DispatchPlan(response_style, approach, clear_exercise=True)

            if (
                response_style == "guided_exercise"
                and _blocks_unconsented_exercise_start(state, message)
            ):
                logger.debug(
                    "therapeutic_dispatch: unconsented exercise guard -> "
                    "psychoeducation"
                )
                return DispatchPlan("psychoeducation", approach)

            return DispatchPlan(response_style, approach)
        except Exception:
            logger.warning(
                "therapeutic_dispatch LLM classifier failed; falling back to regex.",
                exc_info=True,
            )

    # Without an LLM, active exercises continue unless a deterministic exit fired.
    if exercise_lifecycle == "active":
        logger.debug("therapeutic_dispatch: regex fallback - continuing exercise")
        existing_approach = _active_exercise_therapeutic_approach(state) or "none"
        return DispatchPlan("guided_exercise", existing_approach)

    response_style = pick_therapeutic_response_style(message)
    logger.debug(
        "therapeutic_dispatch: regex fallback picked response_style=%s",
        response_style,
    )
    fallback_approach = (
        "motivational_interviewing" if response_style == "supportive" else "none"
    )
    return DispatchPlan(response_style, fallback_approach)
