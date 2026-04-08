"""Therapeutic subgraph entrypoint."""

from __future__ import annotations

import re

from pydantic import BaseModel

from agent.models import ModeType
from agent.modality_selector import select_modalities_for_mode
from agent.nodes.therapeutic import run_therapeutic_response
from agent.nodes.therapeutic_mode_registry import (
    run_registered_therapeutic_mode_response,
)
from agent.prompts.builders import (
    build_therapeutic_classifier_prompt,
    build_therapeutic_classifier_system_prompt,
)
from agent.response_shaping import build_response_guidance
from agent.semantic_signals import get_semantic_signals
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
    r"\bthat'?s not helpful\b",
    r"\byou'?re not getting it\b",
    r"\bwrong direction\b",
)

INTAKE_PATTERNS = (
    r"\bhow does this work\b",
    r"\bwhat can you help with\b",
    r"\bwhat can you do(?: for me)?\b",
    r"\bhow can you help(?: me)?\b",
    r"\bi'?m new here\b",
    r"\bfirst time here\b",
    r"\bwhat are you\b",
)

SUPPORT_PATTERNS = (
    r"\bi(?:'m| am) feeling\b",
    r"\bi feel\b",
    r"\bi've been\b",
    r"\bi have difficulty\b",
    r"\bi keep\b",
    r"\bit'?s like\b",
    r"\bi'm struggling\b",
    r"\bthis has been hard\b",
    r"\bi feel so\b",
)

ACTION_REQUEST_PATTERNS = (
    r"\bwhat can i do\b",
    r"\bwhat should i do\b",
    r"\bhow do i\b",
    r"\bhow can i\b",
    r"\bnavigate\b",
    r"\bcope\b",
    r"\bhandle\b",
    r"\bmanage\b",
    r"\bpractical step\b",
    r"\bsomething concrete\b",
    r"\bwhat helps\b",
)

EXERCISE_PATTERNS = (
    r"\bexercise\b",
    r"\bgrounding\b",
    r"\bbreathing\b",
    r"\bbreathe\b",
    r"\bjournal(?:ing)? prompt\b",
    r"\bthought record\b",
    r"\bhelp me calm down\b",
    r"\bwhat can i do right now\b",
    r"\brealistic step\b",
    r"\bdoable first move\b",
    r"\bget moving again\b",
    r"\bpanic(?:king)?\b",
    r"\boverwhelmed\b",
    r"\bwalk me through something\b",
    r"\bi need to do something\b",
    r"\bcan you walk me through\b",
)

PSYCHOEDUCATION_PATTERNS = (
    r"\bwhat is anxiety\b",
    r"\bexplain anxiety\b",
    r"\bwhy does my body\b",
    r"\bwhy do i react like this\b",
    r"\bnervous system\b",
    r"\bstress response\b",
    r"\bwhat'?s happening in my body\b",
    r"\bhow does anxiety work\b",
    r"\bhow does stress work\b",
    r"\bis it normal\b.*\b(anxiety|panic|stress|body|shake|shaking)\b",
    r"\bwhat is burnout\b",
    r"\bexplain.+\b(burned? out|stress|exhaustion)\b",
)

REFLECTION_PATTERNS = (
    r"\bwhat patterns do you notice\b",
    r"\bwhat do you notice\b",
    r"\bsummarize\b",
    r"\breflect on\b",
    r"\bhelp me understand\b",
    r"\bhelp understanding why i keep\b",
    r"\bunderstanding why i keep\b",
    r"\bwhy do i keep\b",
    r"\bmake sense of\b",
    r"\bis there a theme\b",
    r"\bdo you see a connection\b",
    r"\bwhat keeps happening\b",
)

OPERATIONAL_MODES = frozenset(
    {"safety_check", "out_of_scope", "realignment", "orientation"}
)

LLM_CLASSIFIABLE_MODES = frozenset(
    {
        "supportive_conversation",
        "guided_exercise",
        "psychoeducation",
        "pattern_reflection",
    }
)


class TherapeuticModeClassification(BaseModel):
    """Structured schema for LLM-based therapeutic mode classification."""

    mode: str
    confidence: str
    reason: str


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any pattern in the provided tuple."""

    return any(re.search(pattern, text) for pattern in patterns)


async def classify_therapeutic_mode_with_llm(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> str:
    """Classify the therapeutic mode using an LLM when keyword routing misses.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Provider-backed client for structured generation.

    Returns:
        A validated therapeutic mode string.
    """

    raw = await llm_client.generate_structured(
        prompt=build_therapeutic_classifier_prompt(state),
        response_schema=TherapeuticModeClassification,
        system_instruction=build_therapeutic_classifier_system_prompt(),
        temperature=0,
    )

    if raw.mode in LLM_CLASSIFIABLE_MODES:
        return raw.mode
    return "supportive_conversation"


async def select_therapeutic_mode(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> tuple[str, str]:
    """Select the active non-crisis response mode for the therapeutic subgraph.

    Uses a three-layer routing architecture:
    1. Deterministic keyword patterns (fast, predictable)
    2. Session intent fallback (sticky context from prior turns)
    3. LLM classifier fallback (only when layers 1+2 produce no match)

    Safety-critical modes (safety_check, out_of_scope, realignment, orientation)
    are exclusively deterministic and never touched by the LLM.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for fallback classification.

    Returns:
        A tuple of (mode, source) where source is ``keyword``, ``session_intent``,
        ``llm``, or ``default``.
    """

    # Layer 1: Deterministic keyword routing.
    if state["crisis"].needs_clarification:
        return "safety_check", "keyword"

    message = state["message"].lower()
    has_history = bool(state["history"])
    signals = get_semantic_signals(state)

    if _matches_any(message, BOUNDARY_PATTERNS):
        return "out_of_scope", "keyword"
    if _matches_any(message, REPAIR_PATTERNS):
        return "realignment", "keyword"
    if not has_history and _matches_any(message, INTAKE_PATTERNS):
        return "orientation", "keyword"
    if _matches_any(message, PSYCHOEDUCATION_PATTERNS):
        return "psychoeducation", "keyword"
    if _matches_any(message, EXERCISE_PATTERNS):
        return "guided_exercise", "keyword"
    if _matches_any(message, REFLECTION_PATTERNS):
        return "pattern_reflection", "keyword"
    if _matches_any(message, ACTION_REQUEST_PATTERNS) and (
        signals["wants_grounding"]
        or signals["wants_cbt"]
        or signals["wants_behavioral_activation"]
        or signals["has_anxiety_theme"]
        or signals["has_stress_theme"]
    ):
        return "guided_exercise", "keyword"
    if _matches_any(message, SUPPORT_PATTERNS) and (
        signals["has_anxiety_theme"]
        or signals["has_stress_theme"]
        or signals["has_grief_theme"]
        or signals["has_relational_theme"]
        or signals["is_venting"]
    ):
        return "supportive_conversation", "keyword"

    # Layer 2: Session intent fallback.
    session_intent = state.get("session_intent")
    if session_intent in {"guided_cbt_work", "grounding_or_calm_down"}:
        return "guided_exercise", "session_intent"
    if session_intent == "psychoeducation":
        return "psychoeducation", "session_intent"
    if session_intent == "reflection_and_pattern_finding":
        return "pattern_reflection", "session_intent"
    if session_intent in {"supportive_conversation", "just_need_to_vent"}:
        return "supportive_conversation", "session_intent"
    if state.get("session_stage") == "closing":
        return "supportive_conversation", "session_intent"

    # Layer 3: LLM classifier fallback.
    if llm_client is not None:
        try:
            mode = await classify_therapeutic_mode_with_llm(
                state, llm_client=llm_client
            )
            return mode, "llm"
        except Exception:
            return "supportive_conversation", "default"

    # Layer 4: Deterministic default.
    return "supportive_conversation", "default"


async def run_selected_therapeutic_mode(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Run the response node for an already-selected therapeutic mode.

    Args:
        state: Shared agent state with `mode` already populated.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated shared agent state after the selected node completes.
    """

    mode = state["mode"]

    if mode == "safety_check":
        return await run_therapeutic_response(state, llm_client=llm_client)
    return await run_registered_therapeutic_mode_response(
        state,
        mode=mode,
        llm_client=llm_client,
    )


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

    mode, mode_source = await select_therapeutic_mode(state, llm_client=llm_client)

    state["mode"] = mode
    state["mode_source"] = mode_source
    state["mode_type"] = (
        ModeType.OPERATIONAL if mode in OPERATIONAL_MODES else ModeType.THERAPEUTIC
    )
    state["semantic_signals"] = dict(get_semantic_signals(state))
    state["active_modalities"] = list(select_modalities_for_mode(state, mode))
    state["response_guidance"] = build_response_guidance(state, mode=mode)

    return await run_selected_therapeutic_mode(state, llm_client=llm_client)
