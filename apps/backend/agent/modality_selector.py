"""Select modality overlays for the current non-crisis response mode."""

from __future__ import annotations

from agent.prompts.catalog import Modality
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

ACT_TERMS = (
    "anxious",
    "anxiety",
    "ruminating",
    "rumination",
    "spiral",
    "spiraling",
    "stuck",
    "avoid",
    "avoiding",
    "avoidance",
    "control",
    "uncertainty",
    "panic",
)

DISTRESS_SKILL_TERMS = (
    "grounding",
    "breathe",
    "breathing",
    "panic",
    "overwhelmed",
    "flooded",
    "regulate",
    "regulation",
    "distress",
    "calm down",
)

CBT_TERMS = (
    "cbt",
    "thought record",
    "reframe",
    "cognitive distortion",
    "behavioral activation",
    "problem solving",
)


def _combined_text(state: AgentState) -> str:
    """Combine recent text and session context for modality selection.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        Lowercased text used for lightweight modality heuristics.
    """

    history_text = " ".join(turn.get("content", "") for turn in state["history"][-6:])
    concerns = " ".join(state.get("active_concerns", []))
    goal = state.get("current_goal") or ""
    return f"{history_text} {concerns} {goal} {state['message']}".lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether the given text contains any of the provided terms.

    Args:
        text: Lowercased text to inspect.
        terms: Candidate substrings to match.

    Returns:
        True when any term is present.
    """

    return any(term in text for term in terms)


def select_modalities_for_mode(state: AgentState, mode: str) -> tuple[Modality, ...]:
    """Select modality overlays for the chosen therapeutic mode.

    Args:
        state: Shared agent state for the current turn.
        mode: Selected non-crisis mode for this response.

    Returns:
        A tuple of modality overlays in priority order.
    """

    text = _combined_text(state)
    session_intent = state.get("session_intent")

    if mode == "support":
        modalities: list[Modality] = ["motivational_interviewing"]
        if _contains_any(text, GRIEF_TERMS):
            modalities.append("grief_support")
        if _contains_any(text, RELATIONAL_TERMS):
            modalities.append("interpersonal_therapy")
        if session_intent == "guided_cbt_work" or _contains_any(text, CBT_TERMS):
            modalities.append("cbt")
        if session_intent == "grounding_or_calm_down" or _contains_any(text, ACT_TERMS):
            modalities.append("act")
        if _contains_any(text, DISTRESS_SKILL_TERMS):
            modalities.append("dbt_skills")
        if state["crisis"].level >= 1:
            modalities.append("pfa")
        return tuple(dict.fromkeys(modalities))

    if mode == "reflection":
        modalities = ["motivational_interviewing"]
        if _contains_any(text, GRIEF_TERMS):
            modalities.append("grief_support")
        if _contains_any(text, RELATIONAL_TERMS):
            modalities.append("interpersonal_therapy")
        if session_intent == "guided_cbt_work" or _contains_any(text, CBT_TERMS):
            modalities.append("cbt")
        if _contains_any(text, ACT_TERMS):
            modalities.append("act")
        return tuple(dict.fromkeys(modalities))

    if mode == "guided_exercise":
        modalities: list[Modality] = []
        if session_intent == "guided_cbt_work" or _contains_any(text, CBT_TERMS):
            modalities.append("cbt")
        if session_intent == "grounding_or_calm_down" or _contains_any(
            text, DISTRESS_SKILL_TERMS
        ):
            modalities.extend(("dbt_skills", "pfa"))
        if _contains_any(text, ACT_TERMS):
            modalities.append("act")
        if not modalities:
            modalities.append("cbt")
        return tuple(dict.fromkeys(modalities))

    if mode == "realignment":
        return ("motivational_interviewing",)

    if mode == "safety_check":
        return ("pfa",)

    return ()
