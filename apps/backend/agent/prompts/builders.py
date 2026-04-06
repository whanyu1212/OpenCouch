"""Prompt builders for concrete agent nodes."""

from __future__ import annotations

from agent.prompts.catalog import Modality
from agent.prompts.modes import build_system_prompt
from agent.state import AgentState


def format_recent_history(state: AgentState, *, limit: int = 6) -> str:
    """Format recent history entries for prompt injection.

    Args:
        state: Shared agent state for the current turn.
        limit: Maximum number of recent history entries to include.

    Returns:
        A compact formatted history block for prompt injection.
    """

    history = state["history"][-limit:]
    if not history:
        return "(no prior history)"

    return "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
        for turn in history
        if turn.get("content")
    )


def format_session_context(state: AgentState) -> str:
    """Format structured session context for prompt injection.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        A compact multi-line context block.
    """

    active_concerns = ", ".join(state["active_concerns"]) or "(none)"
    open_loops = "; ".join(state["open_loops"]) or "(none)"
    current_goal = state["current_goal"] or "(not yet clear)"
    session_intent = state.get("session_intent") or "(not yet clear)"
    session_intent_source = state.get("session_intent_source") or "(none)"
    session_stage = state.get("session_stage") or "(not yet clear)"
    session_stage_source = state.get("session_stage_source") or "(none)"

    return (
        f"Turn count: {state['turn_count']}\n"
        f"Session summary: {state['session_summary']}\n"
        f"Active concerns: {active_concerns}\n"
        f"Open loops: {open_loops}\n"
        f"Current goal: {current_goal}\n"
        f"Session intent: {session_intent}\n"
        f"Session intent source: {session_intent_source}\n"
        f"Session stage: {session_stage}\n"
        f"Session stage source: {session_stage_source}"
    )


def format_stage_guidance(state: AgentState) -> str:
    """Format stage-specific response guidance for prompt injection.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        A concise instruction block for the active session stage.
    """

    stage = state.get("session_stage")
    if stage == "closing":
        return (
            "Stage guidance:\n"
            "- Treat this as a closing-phase reply.\n"
            "- Briefly summarize the most important takeaway or shift from this session.\n"
            "- Offer at most one concrete next step or thing to hold onto.\n"
            "- Do not open a broad new exploration.\n"
            "- End with a gentle landing rather than a big open-ended question."
        )
    if stage == "stabilizing":
        return (
            "Stage guidance:\n"
            "- Help the user consolidate what is becoming clearer.\n"
            "- Favor grounding, integration, or next-step planning over deeper excavation."
        )
    if stage == "deepening":
        return (
            "Stage guidance:\n"
            "- It is appropriate to stay with emotional material or pattern exploration.\n"
            "- Keep the reply contained and coherent rather than expanding too broadly."
        )
    return (
        "Stage guidance:\n"
        "- Treat this as an opening-phase reply.\n"
        "- Orient, validate, and avoid jumping too fast into heavy structure."
    )


def build_therapeutic_system_prompt(
    *,
    modalities: tuple[Modality, ...] = ("motivational_interviewing",),
) -> str:
    """Build the system prompt for normal support replies.

    Args:
        modalities: Optional modality overlays for the support response.

    Returns:
        The system prompt for a normal support reply.
    """

    return build_system_prompt(
        mode="support",
        modalities=modalities,
    )


def build_therapeutic_response_prompt(state: AgentState) -> str:
    """Build the user prompt for normal support replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for a normal support reply.
    """

    return (
        "Write the next assistant message for a mental health support conversation.\n\n"
        "Requirements:\n"
        "- Validate the user's experience before offering any suggestion.\n"
        "- Reflect the user's emotional state without sounding scripted.\n"
        "- Offer one helpful next step, perspective, or question.\n"
        "- Ask at most one gentle open-ended question.\n"
        "- Keep the reply concise, around 3 to 5 sentences.\n\n"
        f"{format_stage_guidance(state)}\n\n"
        f"Structured session context:\n{format_session_context(state)}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_crisis_response_system_prompt(
    *,
    modalities: tuple[Modality, ...] = ("pfa",),
) -> str:
    """Build the system prompt for crisis replies.

    Args:
        modalities: Optional modality overlays for crisis response behavior.

    Returns:
        The system prompt for a crisis reply.
    """

    return build_system_prompt(
        mode="crisis_response",
        modalities=modalities,
    )


def build_crisis_response_prompt(state: AgentState) -> str:
    """Build the user prompt for crisis replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for a crisis reply.
    """

    crisis = state["crisis"]
    urgency = (
        "The user may be in immediate danger."
        if crisis.level >= 3
        else "The user appears to have self-harm or suicidal ideation without a clear imminent plan."
    )

    return (
        "Write the next assistant message for a mental health support conversation in crisis mode.\n\n"
        "Requirements:\n"
        "- Start with calm, direct acknowledgement.\n"
        "- Prioritize immediate safety and offline human support.\n"
        "- Encourage contacting emergency services or a local crisis hotline if the user may be in immediate danger.\n"
        "- Encourage contacting a trusted person nearby who can stay with them.\n"
        "- Ask at most one immediate action question.\n"
        "- Keep the reply concise, around 3 to 5 sentences.\n\n"
        f"Crisis context: {urgency}\n"
        f"Classifier reason: {crisis.reason}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_crisis_classifier_system_prompt() -> str:
    """Build the system prompt for the crisis classifier.

    Returns:
        The system prompt for structured crisis classification.
    """

    return build_system_prompt(
        mode="crisis_classifier",
    )


def build_crisis_classifier_prompt(state: AgentState) -> str:
    """Build the structured classification prompt for the crisis gate.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for structured crisis classification.
    """

    return (
        "Assess the user's current risk level for self-harm or suicide.\n\n"
        "Return only the structured schema.\n"
        "Use these levels:\n"
        "- 0: no acute risk\n"
        "- 1: high distress or ambiguous concerning language, but no clear self-harm signal\n"
        "- 2: self-harm or suicidal ideation without clear imminent plan\n"
        "- 3: imminent risk with plan, means, or near-term intent\n\n"
        "Mark needs_crisis_response true for levels 2 or 3.\n"
        "Mark needs_clarification true when the message is concerning but ambiguous.\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_orientation_system_prompt() -> str:
    """Build the system prompt for orientation replies.

    Returns:
        The system prompt for orientation replies.
    """

    return build_system_prompt(
        mode="orientation",
        modalities=("motivational_interviewing",),
    )


def build_orientation_response_prompt(state: AgentState) -> str:
    """Build the user prompt for orientation replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for orientation replies.
    """

    return (
        "Write the next assistant message for the orientation phase of a mental health support conversation.\n\n"
        "Requirements:\n"
        "- Briefly explain what OpenCouch can help with.\n"
        "- State the product boundary naturally without sounding legalistic.\n"
        "- Ask one focused question about what the user wants help with today.\n"
        "- Keep the reply concise, around 3 to 5 sentences.\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_reflection_system_prompt(
    *,
    modalities: tuple[Modality, ...] = ("motivational_interviewing",),
) -> str:
    """Build the system prompt for reflection replies.

    Args:
        modalities: Optional modality overlays for reflection behavior.

    Returns:
        The system prompt for reflection replies.
    """

    return build_system_prompt(
        mode="reflection",
        modalities=modalities,
    )


def build_reflection_response_prompt(state: AgentState) -> str:
    """Build the user prompt for reflection replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for reflection replies.
    """

    return (
        "Write the next assistant message for a reflective mental health support conversation.\n\n"
        "Requirements:\n"
        "- Name one or two patterns or themes carefully.\n"
        "- Do not sound diagnostic or overconfident.\n"
        "- Stay close to the user's own framing.\n"
        "- End with one light check-in question if helpful.\n"
        "- Keep the reply concise, around 3 to 5 sentences.\n\n"
        f"{format_stage_guidance(state)}\n\n"
        f"Structured session context:\n{format_session_context(state)}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_guided_exercise_system_prompt() -> str:
    """Build the system prompt for guided exercise replies.

    Returns:
        The system prompt for guided exercise replies.
    """

    return build_system_prompt(
        mode="guided_exercise",
        modalities=("cbt",),
    )


def build_guided_exercise_response_prompt(state: AgentState) -> str:
    """Build the user prompt for guided exercise replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for guided exercise replies.
    """

    return (
        "Write the next assistant message for a guided self-help exercise.\n\n"
        "Requirements:\n"
        "- Offer one concrete exercise only.\n"
        "- Explain the purpose briefly.\n"
        "- Keep the steps simple and usable right away.\n"
        "- Do not overwhelm the user with multiple options.\n"
        "- Keep the reply concise, around 4 to 6 sentences.\n\n"
        f"{format_stage_guidance(state)}\n\n"
        f"Structured session context:\n{format_session_context(state)}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_out_of_scope_system_prompt() -> str:
    """Build the system prompt for out-of-scope boundary replies.

    Returns:
        The system prompt for out-of-scope boundary replies.
    """

    return build_system_prompt(mode="out_of_scope")


def build_out_of_scope_response_prompt(state: AgentState) -> str:
    """Build the user prompt for out-of-scope boundary replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for out-of-scope boundary replies.
    """

    return (
        "Write the next assistant message for a request that is outside scope.\n\n"
        "Requirements:\n"
        "- Decline clearly and briefly.\n"
        "- State the limitation without sounding defensive.\n"
        "- Redirect to a safer next step or a better kind of help.\n"
        "- Keep the reply concise, around 2 to 4 sentences.\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_realignment_system_prompt() -> str:
    """Build the system prompt for realignment replies.

    Returns:
        The system prompt for realignment replies.
    """

    return build_system_prompt(
        mode="realignment",
        modalities=("motivational_interviewing",),
    )


def build_realignment_response_prompt(state: AgentState) -> str:
    """Build the user prompt for realignment replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for realignment replies.
    """

    return (
        "Write the next assistant message after the user indicates the previous reply missed the mark.\n\n"
        "Requirements:\n"
        "- Acknowledge the miss directly.\n"
        "- Avoid defensiveness or explanation of internal reasoning.\n"
        "- Recenter the user's concern.\n"
        "- Ask one brief corrective question if needed.\n"
        "- Keep the reply concise, around 2 to 4 sentences.\n\n"
        f"{format_stage_guidance(state)}\n\n"
        f"Structured session context:\n{format_session_context(state)}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )
