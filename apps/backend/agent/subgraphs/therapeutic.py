"""Therapeutic subgraph entrypoint."""

from __future__ import annotations

import re

from agent.modality_selector import select_modalities_for_mode
from agent.nodes.guided_exercise import run_guided_exercise_response
from agent.nodes.orientation import run_orientation_response
from agent.nodes.out_of_scope import run_out_of_scope_response
from agent.nodes.reflection import run_reflection_response
from agent.nodes.realignment import run_realignment_response
from agent.nodes.therapeutic import run_therapeutic_response
from agent.state import AgentState
from services.llm.base import BaseLLMClient

BOUNDARY_PATTERNS = (
    r"\bdiagnose\b",
    r"\bdiagnosis\b",
    r"\bmedication\b",
    r"\bwhat meds\b",
    r"\bshould i take\b",
    r"\blawyer\b",
    r"\blegal advice\b",
    r"\bmedical advice\b",
)

REPAIR_PATTERNS = (
    r"\bthat missed the point\b",
    r"\byou missed the point\b",
    r"\bthat did not help\b",
    r"\byou are not listening\b",
    r"\byou don't understand\b",
    r"\bthat's not what i meant\b",
    r"\byou misunderstood\b",
)

INTAKE_PATTERNS = (
    r"\bhow does this work\b",
    r"\bwhat can you help with\b",
    r"\bi'?m new here\b",
    r"\bfirst time here\b",
    r"\bwhat are you\b",
)

EXERCISE_PATTERNS = (
    r"\bexercise\b",
    r"\bgrounding\b",
    r"\bbreathing\b",
    r"\bjournal(?:ing)? prompt\b",
    r"\bthought record\b",
    r"\bhelp me calm down\b",
    r"\bwhat can i do right now\b",
)

REFLECTION_PATTERNS = (
    r"\bwhat patterns do you notice\b",
    r"\bwhat do you notice\b",
    r"\bsummarize\b",
    r"\breflect on\b",
    r"\bhelp me understand\b",
    r"\bwhy do i keep\b",
    r"\bmake sense of\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any pattern in the provided tuple."""

    return any(re.search(pattern, text) for pattern in patterns)


def select_therapeutic_mode(state: AgentState) -> str:
    """Select the active non-crisis response mode for the therapeutic subgraph.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The selected therapeutic mode for downstream response generation.
    """

    if state["crisis"].needs_clarification:
        return "safety_check"

    message = state["message"].lower()
    has_history = bool(state["history"])

    if _matches_any(message, BOUNDARY_PATTERNS):
        return "out_of_scope"
    if _matches_any(message, REPAIR_PATTERNS):
        return "realignment"
    if not has_history and _matches_any(message, INTAKE_PATTERNS):
        return "orientation"
    if _matches_any(message, EXERCISE_PATTERNS):
        return "guided_exercise"
    if _matches_any(message, REFLECTION_PATTERNS):
        return "reflection"

    session_intent = state.get("session_intent")
    if session_intent in {"guided_cbt_work", "grounding_or_calm_down"}:
        return "guided_exercise"
    if session_intent == "reflection_and_pattern_finding":
        return "reflection"
    if session_intent in {"supportive_conversation", "just_need_to_vent"}:
        return "support"
    if state.get("session_stage") == "closing":
        return "support"
    return "support"


async def run_therapeutic_subgraph(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Run the therapeutic response path for the current turn.

    Args:
        state: The shared agent state after crisis-gate routing.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated shared agent state after the therapeutic path completes.
    """

    mode = select_therapeutic_mode(state)
    state["mode"] = mode
    state["active_modalities"] = list(select_modalities_for_mode(state, mode))

    if mode in {"support", "safety_check"}:
        return await run_therapeutic_response(state, llm_client=llm_client)
    if mode == "orientation":
        return await run_orientation_response(state, llm_client=llm_client)
    if mode == "guided_exercise":
        return await run_guided_exercise_response(state, llm_client=llm_client)
    if mode == "reflection":
        return await run_reflection_response(state, llm_client=llm_client)
    if mode == "out_of_scope":
        return await run_out_of_scope_response(state, llm_client=llm_client)
    if mode == "realignment":
        return await run_realignment_response(state, llm_client=llm_client)

    return await run_therapeutic_response(state, llm_client=llm_client)
