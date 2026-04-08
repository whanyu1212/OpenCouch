"""Select modality overlays for the current non-crisis response mode."""

from __future__ import annotations

from agent.prompts.catalog import Modality
from agent.semantic_signals import get_semantic_signals
from agent.state import AgentState


def _limit_modalities(
    modalities: list[Modality],
    *,
    max_count: int,
) -> tuple[Modality, ...]:
    """Deduplicate modalities while keeping priority order bounded.

    Args:
        modalities: Ordered list of candidate modalities.
        max_count: Maximum number of modalities to return.

    Returns:
        A bounded tuple of unique modalities in priority order.
    """

    unique = tuple(dict.fromkeys(modalities))
    return unique[:max_count]


def select_modalities_for_mode(state: AgentState, mode: str) -> tuple[Modality, ...]:
    """Select modality overlays for the chosen therapeutic mode.

    Args:
        state: Shared agent state for the current turn.
        mode: Selected non-crisis mode for this response.

    Returns:
        A tuple of modality overlays in priority order.
    """

    signals = get_semantic_signals(state)
    session_stage = state.get("session_stage")

    if mode == "supportive_conversation":
        modalities: list[Modality] = []
        if signals["safety_sensitive"]:
            modalities.append("pfa")
        if signals["has_grief_theme"]:
            modalities.append("grief_support")
        if signals["wants_grounding"] and session_stage in {"opening", "stabilizing"}:
            modalities.append("pfa")
        elif signals["has_anxiety_theme"]:
            modalities.append("act")
        if signals["wants_cbt"] and session_stage != "closing":
            modalities.append("cbt")
        return _limit_modalities(modalities, max_count=3)

    if mode == "pattern_reflection":
        modalities: list[Modality] = []
        if signals["has_grief_theme"]:
            modalities.append("grief_support")
        if signals["wants_cbt"] and session_stage != "closing":
            modalities.append("cbt")
        elif signals["has_anxiety_theme"]:
            modalities.append("act")
        return _limit_modalities(modalities, max_count=3)

    if mode == "guided_exercise":
        modalities: list[Modality] = []
        if signals["wants_grounding"]:
            modalities.append("pfa")
            if signals["has_anxiety_theme"] and session_stage != "closing":
                modalities.append("act")
        elif signals["wants_behavioral_activation"]:
            modalities.append("cbt")
            if signals["has_anxiety_theme"]:
                modalities.append("act")
        elif signals["wants_cbt"]:
            modalities.append("cbt")
        elif signals["has_anxiety_theme"]:
            modalities.append("act")
        if not modalities:
            modalities.append("cbt")
        return _limit_modalities(modalities, max_count=3)

    if mode == "psychoeducation":
        modalities: list[Modality] = []
        if signals["has_anxiety_theme"] or signals["wants_explanation"]:
            modalities.append("act")
        if signals["wants_cbt"]:
            modalities.append("cbt")
        if signals["safety_sensitive"]:
            modalities.append("pfa")
        if signals["has_grief_theme"]:
            modalities.append("grief_support")
        if not modalities:
            modalities.append("cbt")
        return _limit_modalities(modalities, max_count=3)

    if mode == "realignment":
        return ()

    if mode == "safety_check":
        return ("pfa",)

    return ()
