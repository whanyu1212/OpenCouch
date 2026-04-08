"""Shared turn-level semantic signals used across routing and prompt shaping."""

from __future__ import annotations

from typing import TypedDict, cast

from agent.state import AgentState

RELATIONAL_TERMS = (
    "relationship",
    "partner",
    "boyfriend",
    "girlfriend",
    "wife",
    "husband",
    "friend",
    "friends",
    "family",
    "sister",
    "brother",
    "mother",
    "father",
    "parent",
    "coworker",
    "boss",
    "roommate",
    "breakup",
    "divorce",
    "lonely",
    "alone",
    "conflict",
)

GRIEF_TERMS = (
    "grief",
    "loss",
    "died",
    "death",
    "funeral",
    "bereavement",
)

ANXIETY_TERMS = (
    "anxious",
    "anxiety",
    "ruminating",
    "rumination",
    "spiral",
    "spiraling",
    "panic",
    "nervous system",
    "body",
)

STRESS_TERMS = (
    "stress",
    "overwhelmed",
    "burned out",
    "burnt out",
    "exhausted",
    "drained",
    "depleted",
)

GROUNDING_TERMS = (
    "grounding",
    "breathe",
    "breathing",
    "panic",
    "overwhelmed",
    "flooded",
    "calm down",
    "regulate",
    "regulation",
    "distress",
)

CBT_TERMS = (
    "cbt",
    "thought record",
    "reframe",
    "cognitive distortion",
    "behavioral activation",
    "problem solving",
)

EXPLANATION_TERMS = (
    "what is anxiety",
    "explain anxiety",
    "why does my body",
    "why do i react like this",
    "nervous system",
    "stress response",
    "what's happening in my body",
    "how does anxiety work",
    "how does stress work",
    "what is burnout",
)

PATTERN_REVIEW_TERMS = (
    "pattern",
    "reflect",
    "make sense",
    "understand why i keep",
    "understanding why i keep",
    "what keeps happening",
    "is there a theme",
    "do you see a connection",
)

VENTING_TERMS = (
    "just want to vent",
    "just let me vent",
    "do not want advice",
    "don't want advice",
    "do not need advice",
    "don't need advice",
)

PROGRESS_TERMS = (
    "handled it well",
    "handled it better",
    "better this time",
    "i did it",
    "it went better",
    "i'm proud",
    "i am proud",
    "i actually managed",
    "i actually did",
    "i got through it",
    "i stayed calmer",
    "i stood up for myself",
    "i followed through",
)

BEHAVIORAL_ACTIVATION_TERMS = (
    "stuck",
    "avoid",
    "avoiding",
    "avoidance",
    "can't start",
    "cannot start",
    "depleted",
    "drained",
    "exhausted",
    "low energy",
    "numb",
    "shut down",
    "barely get out of bed",
)

PET_TERMS = (
    "cat",
    "dog",
    "pet",
    "kitten",
    "puppy",
)

PET_CARE_TASK_TERMS = (
    "medication",
    "medications",
    "medicine",
    "pill",
    "pills",
    "dose",
    "doses",
    "feeding it",
    "feed it",
    "give it",
    "grab hold",
    "keep still",
    "hold still",
    "vet",
)


class SemanticSignals(TypedDict):
    """Shared semantic signals for one turn."""

    has_relational_theme: bool
    has_grief_theme: bool
    has_anxiety_theme: bool
    has_stress_theme: bool
    wants_grounding: bool
    wants_cbt: bool
    wants_explanation: bool
    wants_pattern_reflection: bool
    wants_behavioral_activation: bool
    is_venting: bool
    is_progress_update: bool
    needs_supportive_boundary: bool
    safety_sensitive: bool
    is_closing: bool


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether text contains any candidate term."""

    return any(term in text for term in terms)


def _combined_text(state: AgentState) -> str:
    """Combine recent history and structured context for signal derivation."""

    history_text = " ".join(
        turn.get("content", "") for turn in state.get("history", [])
    )
    concerns = " ".join(state.get("active_concerns", []))
    goal = state.get("current_goal") or ""
    return f"{history_text} {concerns} {goal} {state['message']}".lower()


def _recent_user_text(state: AgentState, *, limit: int = 2) -> str:
    """Combine the most recent user turns with the current message."""

    history_user_turns = [
        turn.get("content", "")
        for turn in state.get("history", [])
        if turn.get("role") == "user"
    ]
    parts = history_user_turns[-limit:] + [state["message"]]
    return " ".join(part for part in parts if part).lower()


def derive_semantic_signals(state: AgentState) -> SemanticSignals:
    """Derive shared semantic signals for the current turn."""

    text = _combined_text(state)
    recent_user_text = _recent_user_text(state)
    concerns = {concern.lower() for concern in state.get("active_concerns", [])}
    current_goal = (state.get("current_goal") or "").lower()
    session_intent = state.get("session_intent")
    session_stage = state.get("session_stage")

    has_relational_theme = "relationship strain" in concerns or _contains_any(
        text, RELATIONAL_TERMS
    )
    has_grief_theme = "grief or loss" in concerns or _contains_any(text, GRIEF_TERMS)
    has_anxiety_theme = "anxiety or rumination" in concerns or _contains_any(
        text, ANXIETY_TERMS
    )
    has_stress_theme = "overwhelm or stress" in concerns or _contains_any(
        text, STRESS_TERMS
    )
    wants_grounding = (
        session_intent == "grounding_or_calm_down"
        or "feel calmer right now" in current_goal
        or _contains_any(text, GROUNDING_TERMS)
    )
    wants_cbt = (
        session_intent == "guided_cbt_work"
        or "structured exercise" in current_goal
        or _contains_any(text, CBT_TERMS)
    )
    wants_explanation = session_intent == "psychoeducation" or _contains_any(
        text, EXPLANATION_TERMS
    )
    wants_pattern_reflection = (
        session_intent == "reflection_and_pattern_finding"
        or _contains_any(text, PATTERN_REVIEW_TERMS)
    )
    wants_behavioral_activation = _contains_any(text, BEHAVIORAL_ACTIVATION_TERMS)
    is_venting = session_intent == "just_need_to_vent" or _contains_any(
        text, VENTING_TERMS
    )
    is_progress_update = _contains_any(text, PROGRESS_TERMS)
    needs_supportive_boundary = _contains_any(
        recent_user_text, PET_TERMS
    ) and _contains_any(recent_user_text, PET_CARE_TASK_TERMS)

    return {
        "has_relational_theme": has_relational_theme,
        "has_grief_theme": has_grief_theme,
        "has_anxiety_theme": has_anxiety_theme,
        "has_stress_theme": has_stress_theme,
        "wants_grounding": wants_grounding,
        "wants_cbt": wants_cbt,
        "wants_explanation": wants_explanation,
        "wants_pattern_reflection": wants_pattern_reflection,
        "wants_behavioral_activation": wants_behavioral_activation,
        "is_venting": is_venting,
        "is_progress_update": is_progress_update,
        "needs_supportive_boundary": needs_supportive_boundary,
        "safety_sensitive": state["crisis"].level >= 1
        or state.get("mode") == "safety_check",
        "is_closing": session_stage == "closing",
    }


def get_semantic_signals(state: AgentState) -> SemanticSignals:
    """Return cached semantic signals for the current turn."""

    cached = state.get("semantic_signals")
    if cached:
        return cast(SemanticSignals, cached)
    signals = derive_semantic_signals(state)
    state["semantic_signals"] = dict(signals)
    return signals
