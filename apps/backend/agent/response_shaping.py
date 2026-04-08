"""Helpers for shaping therapeutic responses within a selected mode."""

from __future__ import annotations

from agent.semantic_signals import get_semantic_signals
from agent.state import AgentState


def needs_supportive_boundary(state: AgentState) -> bool:
    """Return whether the turn needs a supportive in-scope boundary."""

    return get_semantic_signals(state)["needs_supportive_boundary"]


def infer_support_strategy(state: AgentState) -> str:
    """Infer how supportive-conversation replies should be shaped.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        One of `hold_space`, `strengths_based`, or `supportive_guidance`.
    """

    signals = get_semantic_signals(state)
    if signals["is_venting"]:
        return "hold_space"
    if signals["is_progress_update"]:
        return "strengths_based"
    return "supportive_guidance"


def infer_guided_exercise_focus(state: AgentState) -> str:
    """Infer which structured intervention best fits the current exercise turn.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        One of `grounding`, `behavioral_activation`, `thought_work`, or `acceptance`.
    """

    signals = get_semantic_signals(state)
    modalities = set(state.get("active_modalities", []))

    if "pfa" in modalities or signals["wants_grounding"]:
        return "grounding"
    if signals["wants_behavioral_activation"]:
        return "behavioral_activation"
    if "cbt" in modalities or signals["wants_cbt"]:
        return "thought_work"
    if "act" in modalities:
        return "acceptance"
    return "thought_work"


def infer_psychoeducation_topic(state: AgentState) -> str:
    """Infer the likely psychoeducation topic for the current turn.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        A short topic label such as `anxiety_response`, `stress_response`, or
        `general_emotional_process`.
    """

    signals = get_semantic_signals(state)

    if signals["has_anxiety_theme"]:
        return "anxiety_response"
    if signals["has_stress_theme"]:
        return "stress_response"
    if signals["has_grief_theme"]:
        return "grief_process"
    return "general_emotional_process"


def build_response_guidance(state: AgentState, *, mode: str) -> str:
    """Build deterministic turn-specific guidance after mode selection.

    Args:
        state: Shared agent state for the current turn.
        mode: Selected response mode for the turn.

    Returns:
        A concise guidance string for prompt shaping.
    """

    if mode == "supportive_conversation":
        strategy = infer_support_strategy(state)
        if needs_supportive_boundary(state):
            return (
                "The user is bringing in a stressful practical task outside OpenCouch's core role. "
                "Acknowledge the strain, avoid procedural advice in that outside domain, and redirect "
                "toward emotional support, pacing, or deciding what kind of help they need."
            )
        if strategy == "hold_space":
            return (
                "User is venting or explicitly does not want advice. Hold space, "
                "reflect the emotional weight, and do not move into fixing or next steps "
                "unless the user asks."
            )
        if strategy == "strengths_based":
            return (
                "User is reporting progress or a win. Name what they handled, noticed, or "
                "followed through on, and reinforce effort or capacity without sounding inflated."
            )
        signals = get_semantic_signals(state)
        if signals["has_relational_theme"]:
            return (
                "User wants support around a relational strain or role transition. Validate first, "
                "then stay close to the concrete relationship, communication gap, or support need "
                "instead of drifting into abstract analysis."
            )
        return (
            "User wants supportive conversation. Validate first, then offer at most one gentle "
            "next step, perspective, or check-in question."
        )

    if mode == "pattern_reflection":
        signals = get_semantic_signals(state)
        if infer_support_strategy(state) == "strengths_based":
            return (
                "Pattern reflection should stay strengths-aware. Name what the user did differently, "
                "reflect back the shift clearly, and avoid making it sound grandiose or cheesy."
            )
        if "grief_support" in state.get("active_modalities", []):
            return (
                "Pattern reflection should stay gentle and grief-aware. Name themes tentatively and "
                "avoid treating grief like a problem to solve."
            )
        if signals["has_relational_theme"]:
            return (
                "Pattern reflection should emphasize relational themes, communication patterns, and "
                "role strain without sounding diagnostic."
            )
        if "act" in state.get("active_modalities", []):
            return (
                "Pattern reflection should notice loops of struggle, avoidance, or fusion with thoughts "
                "without arguing with the user's experience."
            )
        return (
            "Pattern reflection should name one or two recurring themes carefully, stay close to the "
            "user's own framing, and avoid overclaiming."
        )

    if mode == "guided_exercise":
        focus = infer_guided_exercise_focus(state)
        if focus == "grounding":
            return (
                "User is activated or overwhelmed. Use grounding or regulation, not thought work, "
                "and keep the exercise immediately usable."
            )
        if focus == "behavioral_activation":
            return (
                "User sounds stuck, depleted, or avoidant. Use behavioral activation with one tiny, "
                "realistic action rather than a demanding plan."
            )
        if focus == "acceptance":
            return (
                "Use a brief acceptance or defusion exercise. Help the user step back from the thought "
                "or feeling and choose one small values-aligned action."
            )
        return (
            "Use one bounded CBT-style exercise such as a thought check or brief thought record. "
            "Keep it concrete and avoid overwhelming the user."
        )

    if mode == "psychoeducation":
        topic = infer_psychoeducation_topic(state)
        if topic == "anxiety_response":
            return (
                "The user wants a normalizing explanation of anxiety or body activation. Explain one "
                "likely mind-body process in simple, non-diagnostic language."
            )
        if topic == "stress_response":
            return (
                "The user wants to understand a stress response. Explain how ongoing activation can affect "
                "rest, concentration, patience, or energy without sounding like a textbook."
            )
        if topic == "grief_process":
            return (
                "The user wants a grief-oriented explanation. Normalize the nonlinear, shifting quality of grief "
                "without turning it into pathology."
            )
        return (
            "Offer one brief, grounded explanation of what may be happening emotionally or physically, "
            "and tie it directly to the user's experience."
        )

    if mode == "orientation":
        return (
            "Explain what OpenCouch can help with, state the boundary naturally, and ask one focused question "
            "about what the user wants help with today."
        )

    if mode == "safety_check":
        return (
            "Ask one direct safety question only. Do not continue ordinary support, interpretation, or structured work "
            "until the safety ambiguity is clarified."
        )

    if mode == "out_of_scope":
        return "Decline clearly, state the limitation briefly, and redirect toward a safer or more appropriate source of help."

    if mode == "realignment":
        return "Acknowledge the miss directly, avoid defensiveness, and re-attune to what the user actually meant."

    return ""
