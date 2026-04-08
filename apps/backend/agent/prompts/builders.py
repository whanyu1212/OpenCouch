"""Prompt builders for concrete agent nodes."""

from __future__ import annotations

from agent.response_shaping import (
    infer_guided_exercise_focus,
    infer_psychoeducation_topic,
    infer_support_strategy,
)
from agent.prompts.catalog import Modality
from agent.prompts.loader import compose_knowledge_sections
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
    working_memory = "; ".join(state.get("working_memory", [])) or "(none)"
    current_goal = state["current_goal"] or "(not yet clear)"
    session_intent = state.get("session_intent") or "(not yet clear)"
    session_intent_source = state.get("session_intent_source") or "(none)"
    session_stage = state.get("session_stage") or "(not yet clear)"
    session_stage_source = state.get("session_stage_source") or "(none)"

    return (
        f"Turn count: {state['turn_count']}\n"
        f"Session summary: {state['session_summary']}\n"
        f"Long-term memory: {working_memory}\n"
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
    paths = ["session_stages.md"]
    if stage == "closing":
        paths.append("response_modes/closing_guidance.md")

    return compose_knowledge_sections(*paths)


def format_response_guidance(state: AgentState) -> str:
    """Format turn-specific response guidance for prompt injection.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        A short response-guidance block or a placeholder when absent.
    """

    guidance = (state.get("response_guidance") or "").strip()
    return guidance if guidance else "(no additional turn-specific guidance)"


def _infer_turn_posture(state: AgentState) -> str:
    """Infer a short, turn-aware posture instruction based on conversational context.

    This replaces fixed bullet-point requirements with context-sensitive guidance
    that varies across turns, producing more natural conversation rhythm.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        A concise posture instruction for the current turn.
    """

    turn = state["turn_count"]
    stage = state.get("session_stage", "opening")
    history = state.get("history", [])
    # What did we say last turn?
    last_assistant = ""
    for entry in reversed(history[-4:]):
        if entry.get("role") == "assistant":
            last_assistant = entry.get("content", "").lower()
            break

    asked_question_last = "?" in last_assistant
    user_answering = asked_question_last and turn > 1
    user_asking = "?" in state["message"]
    user_short = len(state["message"].split()) <= 6

    parts: list[str] = []

    # Turn-count awareness
    if turn <= 1:
        parts.append(
            "This is the first turn. Orient to what the user needs before "
            "assuming a direction. Do not lead with a plan or structure."
        )
    elif turn == 2:
        parts.append(
            "This is an early turn. You are still learning what matters to "
            "them. Follow their lead rather than steering."
        )

    # Responding to the user answering your question
    if user_answering and not user_asking:
        parts.append(
            "The user is responding to your previous question. Build on what "
            "they said — do not re-validate what you already acknowledged. "
            "Move the conversation forward."
        )

    # Short user messages
    if user_short and turn > 2:
        parts.append(
            "The user gave a short reply. Match their energy — a brief, "
            "focused response is better than a long one. One or two "
            "sentences can be enough."
        )

    # User is asking a question
    if user_asking:
        parts.append(
            "The user asked a question. Answer it directly before adding anything else."
        )

    # Stage-specific posture
    if stage == "deepening" and turn >= 3:
        parts.append(
            "You are in a deepening phase. You have enough context to be "
            "more direct and specific. Avoid repeating earlier reflections."
        )
    elif stage == "stabilizing":
        parts.append(
            "The conversation is consolidating. Favor clarity and "
            "integration over further exploration."
        )
    elif stage == "closing":
        parts.append(
            "The session is ending. Be brief. Summarize one thread, "
            "offer one takeaway. Do not open new territory."
        )

    # Default — avoid the same shape every turn
    if not parts:
        parts.append(
            "Respond naturally to what the user said. Vary your response "
            "shape — not every turn needs a reflection followed by a question."
        )

    return " ".join(parts)


def build_therapeutic_system_prompt(
    *,
    modalities: tuple[Modality, ...] = (),
) -> str:
    """Build the system prompt for normal support replies.

    Args:
        modalities: Optional modality overlays for the support response.

    Returns:
        The system prompt for a normal support reply.
    """

    return build_system_prompt(
        mode="supportive_conversation",
        modalities=modalities,
    )


def build_therapeutic_response_prompt(state: AgentState) -> str:
    """Build the user prompt for normal support replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for a normal support reply.
    """

    support_strategy = infer_support_strategy(state)
    if support_strategy == "hold_space":
        strategy_note = (
            "Strategy: hold space. The user wants to vent or process — "
            "do not offer advice, fixes, or next steps unless they ask. "
            "Reflect the weight of what they said. Silence and brevity are fine."
        )
    elif support_strategy == "strengths_based":
        strategy_note = (
            "Strategy: strengths-based. The user is reporting something "
            "they handled or a shift they made. Name the specific thing "
            "without inflating it. Let them sit with the win."
        )
    else:
        strategy_note = (
            "Strategy: supportive guidance. Respond to what the user "
            "actually said. If it calls for validation, validate. If it "
            "calls for a perspective, offer one. If it calls for a question, ask one. "
            "Do not do all three every turn."
        )

    posture = _infer_turn_posture(state)

    return (
        "Write the next assistant message for a mental health support conversation.\n\n"
        f"{strategy_note}\n\n"
        f"Turn posture: {posture}\n\n"
        f"Turn-specific guidance:\n{format_response_guidance(state)}\n\n"
        f"Session context:\n{format_session_context(state)}\n\n"
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
        "Acknowledge directly and calmly. Prioritize immediate safety — "
        "surface crisis resources (988 Suicide & Crisis Lifeline) and "
        "encourage contacting a trusted person. Ask at most one safety "
        "question. Be concise and clear.\n\n"
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
        "Briefly explain what OpenCouch can help with. State the boundary "
        "naturally — not a therapist, not an emergency service. Ask one "
        "question about what the user wants today. Keep it warm and short.\n\n"
        f"Turn-specific guidance:\n{format_response_guidance(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_reflection_system_prompt(
    *,
    modalities: tuple[Modality, ...] = (),
) -> str:
    """Build the system prompt for reflection replies.

    Args:
        modalities: Optional modality overlays for reflection behavior.

    Returns:
        The system prompt for reflection replies.
    """

    return build_system_prompt(
        mode="pattern_reflection",
        modalities=modalities,
    )


def build_reflection_response_prompt(state: AgentState) -> str:
    """Build the user prompt for reflection replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for reflection replies.
    """

    posture = _infer_turn_posture(state)

    return (
        "Write the next assistant message for a reflective mental health support conversation.\n\n"
        "Name patterns tentatively — present them as observations to test, not conclusions. "
        "Stay close to the user's own words. Do not sound diagnostic.\n\n"
        f"Turn posture: {posture}\n\n"
        f"Turn-specific guidance:\n{format_response_guidance(state)}\n\n"
        f"Session context:\n{format_session_context(state)}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_guided_exercise_system_prompt(
    *,
    modalities: tuple[Modality, ...] = ("cbt",),
) -> str:
    """Build the system prompt for guided exercise replies.

    Args:
        modalities: Optional modality overlays for guided exercise behavior.

    Returns:
        The system prompt for guided exercise replies.
    """

    return build_system_prompt(
        mode="guided_exercise",
        modalities=modalities,
    )


def build_guided_exercise_response_prompt(state: AgentState) -> str:
    """Build the user prompt for guided exercise replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for guided exercise replies.
    """

    focus = infer_guided_exercise_focus(state)
    if focus == "grounding":
        focus_guidance = (
            "Exercise focus: grounding or regulation.\n"
            "- Prefer a brief sensory, breathing, or distress-tolerance practice.\n"
            "- Help the user settle activation before asking them to analyze anything."
        )
    elif focus == "behavioral_activation":
        focus_guidance = (
            "Exercise focus: behavioral activation.\n"
            "- Prefer one tiny, realistic action tied to routine, movement, connection, or mastery.\n"
            "- Make the first step intentionally easy to start."
        )
    elif focus == "acceptance":
        focus_guidance = (
            "Exercise focus: acceptance and defusion.\n"
            "- Help the user step back from the thought or feeling instead of arguing with it.\n"
            "- End with one small values-aligned step."
        )
    else:
        focus_guidance = (
            "Exercise focus: thought work.\n"
            "- Prefer one simple CBT-style structure such as a thought check or brief thought record.\n"
            "- Keep the exercise bounded and concrete."
        )

    posture = _infer_turn_posture(state)

    return (
        "Write the next assistant message for a guided self-help exercise.\n\n"
        "One exercise only. Explain the purpose in one line, then guide "
        "through the steps. Keep it immediately usable — not a worksheet.\n\n"
        f"{focus_guidance}\n\n"
        f"Turn posture: {posture}\n\n"
        f"Turn-specific guidance:\n{format_response_guidance(state)}\n\n"
        f"Session context:\n{format_session_context(state)}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_psychoeducation_system_prompt(
    *,
    modalities: tuple[Modality, ...] = ("cbt",),
) -> str:
    """Build the system prompt for psychoeducation replies.

    Args:
        modalities: Optional modality overlays for psychoeducation behavior.

    Returns:
        The system prompt for psychoeducation replies.
    """

    return build_system_prompt(
        mode="psychoeducation",
        modalities=modalities,
    )


def build_psychoeducation_response_prompt(state: AgentState) -> str:
    """Build the user prompt for psychoeducation replies.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for psychoeducation replies.
    """

    topic = infer_psychoeducation_topic(state)
    posture = _infer_turn_posture(state)

    return (
        "Write the next assistant message for a psychoeducation turn in a mental health support conversation.\n\n"
        "Explain one process in plain language. Connect it to what the user "
        "is actually experiencing — not a textbook explanation. Check whether "
        "it fits before adding more.\n\n"
        f"Psychoeducation topic: {topic}\n\n"
        f"Turn posture: {posture}\n\n"
        f"Turn-specific guidance:\n{format_response_guidance(state)}\n\n"
        f"Session context:\n{format_session_context(state)}\n\n"
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
        "Decline clearly. State the limit without being defensive. "
        "Redirect toward what you can help with or who can help better. Be brief.\n\n"
        f"Turn-specific guidance:\n{format_response_guidance(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_realignment_system_prompt() -> str:
    """Build the system prompt for realignment replies.

    Returns:
        The system prompt for realignment replies.
    """

    return build_system_prompt(
        mode="realignment",
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
        "Acknowledge the miss directly — no defensiveness, no explanation of your reasoning. "
        "Recenter on what they actually meant. Be brief.\n\n"
        f"Turn-specific guidance:\n{format_response_guidance(state)}\n\n"
        f"Session context:\n{format_session_context(state)}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )


def build_therapeutic_classifier_system_prompt() -> str:
    """Build the system prompt for the therapeutic mode classifier.

    Returns:
        The system prompt for structured mode classification.
    """

    return (
        "You are classifying the therapeutic mode for a mental health support conversation.\n"
        "Choose the single mode that best matches the user's current need.\n"
        "Return only the structured schema.\n"
        "Prefer supportive_conversation when the intent is genuinely unclear."
    )


def build_therapeutic_classifier_prompt(state: AgentState) -> str:
    """Build the user prompt for the therapeutic mode classifier.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        The user/task prompt for structured mode classification.
    """

    session_intent = state.get("session_intent") or "(none)"
    current_goal = state.get("current_goal") or "(none)"
    active_concerns = ", ".join(state.get("active_concerns", [])) or "(none)"

    return (
        "Classify the user's current therapeutic need.\n\n"
        "Allowed modes:\n"
        "- supportive_conversation: general emotional support, venting, processing\n"
        "- guided_exercise: grounding, breathing, thought records, behavioral activation, acceptance exercises\n"
        "- psychoeducation: explaining what is happening in mind or body, normalizing reactions\n"
        "- pattern_reflection: reflecting on recurring themes, connections, or cycles\n\n"
        f"Session context:\n"
        f"Session intent: {session_intent}\n"
        f"Current goal: {current_goal}\n"
        f"Active concerns: {active_concerns}\n\n"
        f"Recent conversation:\n{format_recent_history(state, limit=4)}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )
