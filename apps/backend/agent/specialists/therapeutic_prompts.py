"""System and task prompt builders for therapeutic response styles."""

from __future__ import annotations

from collections.abc import Mapping

from agent.runtime.session.state import current_turn_lifecycle
from agent.prompts import (
    compose_sources as _compose,
    format_recent_history as _format_recent_history,
    load_prompt_source as _load_knowledge_file,
)
from agent.state import AgentState
from agent.specialists.therapeutic_prompt_context import (
    _format_procedural_rules_block,
    _format_recall_toggle_constraint,
    _format_working_memory,
    _has_episodic_context,
)
from agent.specialists.therapeutic_prompt_instructions import (
    _CLARIFYING_INSTRUCTIONS,
    _CLOSING_INSTRUCTIONS,
    _CONTINUITY_FILE,
    _PSYCHOEDUCATION_INSTRUCTIONS,
    _REFLECTIVE_INSTRUCTIONS,
    _SUPPORTIVE_INSTRUCTIONS,
    _TECHNIQUE_INSTRUCTIONS,
)
from agent.specialists.therapeutic_prompt_sources import (
    _knowledge_for_response_style,
)

_SAFETY_CHECK_FILE = "response_styles/safety_check.md"
_SAFETY_CHECK_OVERRIDE = """
Safety-check override:
- The crisis classifier marked this turn as concerning but ambiguous
  (level 1). Include exactly one direct safety question before ordinary
  support continues.
- In clarifying style, these safety-check instructions override the
  ordinary clarifying guidance about open-ended context questions.
- Do not provide hotline, 988, emergency-services, ER, or crisis-line
  guidance unless crisis response is active or the user says they may
  act on self-harm thoughts.
""".strip()


def _crisis_needs_safety_clarification(state: AgentState) -> bool:
    """Return whether the current turn needs a safety clarification.

    Args:
        state: Current agent state.

    Returns:
        Whether the crisis gate marked this turn as level-1 ambiguous risk.
    """

    crisis = state.get("crisis")
    if crisis is None:
        return False
    if isinstance(crisis, dict):
        return bool(crisis.get("needs_clarification", False))
    return bool(getattr(crisis, "needs_clarification", False))


def _compose_system_prompt_with_state(
    knowledge: str,
    instructions: str,
    state: AgentState,
) -> str:
    """Assemble a system prompt from static + dynamic parts.

    The static parts are the knowledge-file composition and the
    response-style instructions block. The dynamic parts are the
    procedural rules block, the recall toggle constraint (both
    read from state), and the cross-session continuity guidance
    (loaded only for returning users with episodic memory).

    Procedural rules are injected as silent constraints; the recall toggle
    only controls explicit references to semantic and episodic memory.

    Args:
        knowledge: Static source-file prompt content.
        instructions: Mode-specific behavioral instructions.
        state: Current runtime state used for dynamic prompt blocks.

    Returns:
        Full system prompt for a therapeutic response style.
    """

    continuity_block = ""
    if _has_episodic_context(state):
        continuity_block = "\n\n" + _load_knowledge_file(_CONTINUITY_FILE)

    safety_block = ""
    if _crisis_needs_safety_clarification(state):
        safety_block = "\n\n" + _SAFETY_CHECK_OVERRIDE

    rules_block = _format_procedural_rules_block(state)
    recall_block = _format_recall_toggle_constraint(state)
    session_arc_block = _format_session_arc_response_block(state)
    active_flow_block = _format_active_flow_response_block(state)
    clarification_block = _format_clarification_response_block(state)
    return (
        f"{knowledge}\n\n{instructions}{safety_block}"
        f"{continuity_block}{rules_block}{recall_block}"
        f"{session_arc_block}{active_flow_block}{clarification_block}"
    )


def _format_clarification_response_block(state: AgentState) -> str:
    """Return private guidance for mixed-intent clarification turns."""

    turn_lifecycle = state.get("turn_lifecycle", {}) or {}
    if not isinstance(turn_lifecycle, Mapping):
        return ""

    clarification_needed = turn_lifecycle.get("clarification_needed") is True
    clarification_kind = str(turn_lifecycle.get("clarification_kind") or "none")
    no_clarification_reason = str(
        turn_lifecycle.get("no_clarification_reason") or "none"
    )
    if not clarification_needed and no_clarification_reason == "none":
        return ""

    lines = [
        "\n\nMixed-intent clarification guidance:",
        "- Treat this as private routing guidance, not text to recite.",
    ]
    intent_summary = str(turn_lifecycle.get("intent_summary") or "").strip()
    if intent_summary:
        lines.append(f"- Intent summary: {intent_summary}")
    clarification_question = str(
        turn_lifecycle.get("clarification_question") or ""
    ).strip()

    if clarification_needed and clarification_kind == "blocking":
        lines.extend(
            [
                "- Ask exactly one concise clarification question before taking route-specific action.",
                "- Acknowledge the plausible user needs in natural language without mentioning internal route names.",
                "- Do not start a guided exercise, perform grounded lookup, or mutate saved memory on this turn.",
            ]
        )
        if clarification_question:
            lines.append(
                f"- Suggested clarification question: {clarification_question}"
            )
    elif clarification_needed and clarification_kind == "soft":
        lines.extend(
            [
                "- Proceed with the selected action while briefly acknowledging the secondary need if it fits.",
                "- Invite correction in one light phrase; do not block the response with a question.",
            ]
        )
    elif no_clarification_reason == "explicit_privacy_control":
        lines.append(
            "- Respect the user's privacy or saved-memory control without asking whether to comply."
        )
    elif no_clarification_reason == "explicit_action_request":
        lines.append(
            "- Follow the explicit safe action request; do not ask the user to choose between support and that action."
        )
    elif no_clarification_reason == "clear_single_intent":
        lines.append(
            "- Continue with the clear primary intent; no clarification is needed."
        )

    return "\n".join(lines)


def _format_session_arc_response_block(state: AgentState) -> str:
    """Return private response guidance from runtime session state.

    Args:
        state: Current runtime state.

    Returns:
        str: Optional system-prompt block for session-arc pacing.
    """

    session_progress = state.get("session_progress", {}) or {}
    session_intent = session_progress.get("session_intent")
    session_stage = session_progress.get("session_stage")
    guidance_permission = session_progress.get("guidance_permission")
    response_guidance = str(state.get("response_guidance") or "").strip()
    if not (
        session_intent or session_stage or guidance_permission or response_guidance
    ):
        return ""

    lines = [
        "\n\nCurrent session arc guidance:",
        "- Treat this as private pacing guidance, not text to recite.",
    ]
    if session_intent:
        lines.append(f"- session_intent: {session_intent}")
        if session_intent == "repair":
            lines.extend(
                [
                    "- Repair turn: briefly acknowledge the miss, own it without "
                    "over-apologizing, and reset to the user's stated meaning or "
                    "preference.",
                    "- Do not defend, explain your intent, argue, or add advice, "
                    "an exercise, or structure unless the user explicitly asks "
                    "for that in this same turn.",
                    "- Ask at most one concise reset question only if the user's "
                    "preferred direction is unclear.",
                ]
            )
    if session_stage:
        lines.append(f"- session_stage: {session_stage}")
    if guidance_permission:
        lines.append(f"- guidance_permission: {guidance_permission}")
        if guidance_permission == "not_yet":
            lines.append(
                "- Respect the user's pacing: validate and reflect before "
                "offering advice, exercises, or structured problem-solving."
            )
        elif guidance_permission == "granted":
            lines.append(
                "- The user has invited guidance; use structure when it fits "
                "the selected response style."
            )
        elif guidance_permission == "unknown":
            lines.append(
                "- Do not assume the user wants coaching yet; keep the next "
                "move light and collaborative."
            )
    if response_guidance:
        lines.append(f"- response_guidance: {response_guidance}")
    return "\n".join(lines)


def _format_active_flow_response_block(state: AgentState) -> str:
    """Return response guidance for active-flow lifecycle turns.

    Args:
        state: Current runtime state.

    Returns:
        str: Optional system-prompt block for active-flow continuity.
    """

    active_flow = current_turn_lifecycle(state)
    if (
        active_flow.active_flow == "guided_exercise"
        and active_flow.action == "preserve"
    ):
        return (
            "\n\nActive-flow continuity:\n"
            "- A guided exercise is paused while you answer this side request.\n"
            "- Answer the current request first. Do not advance, restart, or end "
            "the exercise.\n"
            "- If natural, close with a brief option to return to the exercise "
            "when the user is ready."
        )
    if (
        active_flow.active_flow == "pending_memory_action"
        and active_flow.action == "clear"
    ):
        return (
            "\n\nPending memory action cleared:\n"
            "- The user moved away from a pending memory deletion. Briefly "
            "acknowledge that you did not change memory.\n"
            "- Then answer the current request normally. If this is a closing "
            "turn, keep the whole reply short."
        )
    return ""


def _read_approach(state: AgentState) -> str | None:
    """Read the runtime-selected therapeutic approach from top-level state.

    Args:
        state: Current runtime state.

    Returns:
        Current therapeutic approach, if one is set.
    """

    return state.get("therapeutic_approach")


def build_supportive_system_prompt(state: AgentState) -> str:
    """Build the system prompt for supportive responses.

    Loads the therapeutic approach overlay selected by the runtime
    (defaults to MI if no approach was set, for backward compatibility).

    Args:
        state: Current runtime state.

    Returns:
        System prompt for supportive responses.
    """

    approach = _read_approach(state) or "motivational_interviewing"
    files = _knowledge_for_response_style("supportive", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _SUPPORTIVE_INSTRUCTIONS, state)


def build_reflective_system_prompt(state: AgentState) -> str:
    """Build the system prompt for reflective responses.

    Args:
        state: Current runtime state.

    Returns:
        System prompt for reflective responses.
    """

    approach = _read_approach(state)
    files = _knowledge_for_response_style("reflective", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _REFLECTIVE_INSTRUCTIONS, state)


def build_clarifying_system_prompt(state: AgentState) -> str:
    """Build the system prompt for clarifying responses.

    Args:
        state: Current runtime state.

    Returns:
        System prompt for clarifying responses.
    """

    files = _knowledge_for_response_style("clarifying")
    if _crisis_needs_safety_clarification(state):
        files = (*files, _SAFETY_CHECK_FILE)

    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _CLARIFYING_INSTRUCTIONS, state)


def build_psychoeducation_system_prompt(state: AgentState) -> str:
    """Build the system prompt for psychoeducation responses.

    Args:
        state: Current runtime state.

    Returns:
        System prompt for psychoeducation responses.
    """

    approach = _read_approach(state)
    files = _knowledge_for_response_style("psychoeducation", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(
        knowledge, _PSYCHOEDUCATION_INSTRUCTIONS, state
    )


def build_closing_system_prompt(state: AgentState) -> str:
    """Build the system prompt for closing responses.

    Args:
        state: Current runtime state.

    Returns:
        System prompt for closing responses.
    """

    knowledge = _compose(*_knowledge_for_response_style("closing"))
    return _compose_system_prompt_with_state(knowledge, _CLOSING_INSTRUCTIONS, state)


def build_technique_system_prompt(state: AgentState) -> str:
    """Build the system prompt for technique responses.

    For technique responses, the therapeutic approach drives the response
    shape. The approach knowledge files are loaded as the primary
    behavioral guidance — the technique instructions just say "follow
    the approach's process guidance." This is the one response style
    where the approach is loud, not background.

    A valid therapeutic approach MUST be set in state. If no
    approach is active, this falls back to the supportive prompt to
    avoid generating an ungrounded response.

    Args:
        state: Current runtime state.

    Returns:
        System prompt for technique responses.
    """

    approach = _read_approach(state)
    if not approach or approach == "none":
        # No approach active — technique without an approach
        # doesn't make sense. Fall back to supportive.
        return build_supportive_system_prompt(state)

    files = _knowledge_for_response_style("technique", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _TECHNIQUE_INSTRUCTIONS, state)


def build_therapeutic_response_prompt(
    state: AgentState,
    *,
    response_style: str,
    step_directive: str | None = None,
) -> str:
    """Build the user/task prompt for any therapeutic response style.

    All response styles share the same user-prompt structure — the system
    prompt (which differs per style) is what shapes the response
    character. The user prompt provides the conversation context
    and current message.

    Args:
        state: Current runtime state with history and working memory.
        response_style: The runtime-selected response style, injected as context for
            observability in the prompt.
        step_directive: For multi-turn styles (guided_exercise), an
            explicit instruction about what the LLM should generate.
            This bridges the exercise runner's state transition to the LLM's
            prose generation: the runner knows which step to produce, and tells
            the LLM via this directive.

    Returns:
        User/task prompt for the therapeutic response LLM call.
    """

    memory_block = _format_working_memory(state)
    directive_block = f"\n\nStep directive:\n{step_directive}" if step_directive else ""

    clarification_block = _format_clarification_response_block(state)
    return (
        f"Write the next assistant message for a mental health support "
        f"conversation in the {response_style} response style.\n\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {state['message']}"
        f"{directive_block}{clarification_block}"
    )
