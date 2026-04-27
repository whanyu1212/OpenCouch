"""System and task prompt builders for therapeutic response modes."""

from __future__ import annotations

from agent.prompts.shared import (
    compose_sources as _compose,
    format_recent_history as _format_recent_history,
    load_prompt_source as _load_knowledge_file,
)
from agent.state import AgentState
from agent.therapeutic.prompting.context import (
    _format_procedural_rules_block,
    _format_recall_toggle_constraint,
    _format_working_memory,
    _has_episodic_context,
)
from agent.therapeutic.prompting.instructions import (
    _CLARIFYING_INSTRUCTIONS,
    _CLOSING_INSTRUCTIONS,
    _CONTINUITY_FILE,
    _GUIDED_EXERCISE_INSTRUCTIONS,
    _PSYCHOEDUCATION_INSTRUCTIONS,
    _REFLECTIVE_INSTRUCTIONS,
    _SUPPORTIVE_INSTRUCTIONS,
    _TECHNIQUE_INSTRUCTIONS,
)
from agent.therapeutic.prompting.sources import _knowledge_for_mode


def _compose_system_prompt_with_state(
    knowledge: str,
    instructions: str,
    state: AgentState,
) -> str:
    """Assemble a system prompt from static + dynamic parts.

    The static parts are the knowledge-file composition and the
    mode-specific instructions block. The dynamic parts are the
    procedural rules block, the recall toggle constraint (both
    read from state), and the cross-session continuity guidance
    (loaded only for returning users with episodic memory).

    Procedural rules are injected as silent constraints; the recall toggle
    only controls explicit references to semantic and episodic memory.

    Args:
        knowledge: Static source-file prompt content.
        instructions: Mode-specific behavioral instructions.
        state: Current graph state used for dynamic prompt blocks.

    Returns:
        Full system prompt for a therapeutic response mode.
    """

    continuity_block = ""
    if _has_episodic_context(state):
        continuity_block = "\n\n" + _load_knowledge_file(_CONTINUITY_FILE)

    rules_block = _format_procedural_rules_block(state)
    recall_block = _format_recall_toggle_constraint(state)
    return f"{knowledge}\n\n{instructions}{continuity_block}{rules_block}{recall_block}"


def _read_approach(state: AgentState) -> str | None:
    """Read the dispatcher-selected modality from top-level state.

    Args:
        state: Current graph state.

    Returns:
        Current therapeutic approach, if one is set.
    """

    return state.get("therapeutic_approach")


def build_supportive_system_prompt(state: AgentState) -> str:
    """Build the system prompt for supportive-mode responses.

    Loads the modality overlay selected by the dispatcher (defaults to
    MI if no modality was set, for backward compatibility).

    Args:
        state: Current graph state.

    Returns:
        System prompt for supportive-mode responses.
    """

    modality = _read_approach(state) or "motivational_interviewing"
    files = _knowledge_for_mode("supportive", modality)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _SUPPORTIVE_INSTRUCTIONS, state)


def build_reflective_system_prompt(state: AgentState) -> str:
    """Build the system prompt for reflective-mode responses.

    Args:
        state: Current graph state.

    Returns:
        System prompt for reflective-mode responses.
    """

    modality = _read_approach(state)
    files = _knowledge_for_mode("reflective", modality)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _REFLECTIVE_INSTRUCTIONS, state)


def build_clarifying_system_prompt(state: AgentState) -> str:
    """Build the system prompt for clarifying-mode responses.

    Args:
        state: Current graph state.

    Returns:
        System prompt for clarifying-mode responses.
    """

    knowledge = _compose(*_knowledge_for_mode("clarifying"))
    return _compose_system_prompt_with_state(knowledge, _CLARIFYING_INSTRUCTIONS, state)


def build_psychoeducation_system_prompt(state: AgentState) -> str:
    """Build the system prompt for psychoeducation-mode responses.

    Args:
        state: Current graph state.

    Returns:
        System prompt for psychoeducation-mode responses.
    """

    modality = _read_approach(state)
    files = _knowledge_for_mode("psychoeducation", modality)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(
        knowledge, _PSYCHOEDUCATION_INSTRUCTIONS, state
    )


def build_closing_system_prompt(state: AgentState) -> str:
    """Build the system prompt for closing-mode responses.

    Args:
        state: Current graph state.

    Returns:
        System prompt for closing-mode responses.
    """

    knowledge = _compose(*_knowledge_for_mode("closing"))
    return _compose_system_prompt_with_state(knowledge, _CLOSING_INSTRUCTIONS, state)


def build_guided_exercise_system_prompt(state: AgentState) -> str:
    """Build the system prompt for guided_exercise-mode responses.

    Loads the base exercise knowledge plus the approach overlay (CBT for
    thought records, ACT for defusion, etc.). Prefers the stable
    ``exercise_state.exercise_modality`` captured at exercise start over the
    per-turn top-level ``therapeutic_approach`` — the latter can drift if
    the user takes a clarifying or psychoeducation side-turn mid-exercise.

    Args:
        state: Current graph state.

    Returns:
        System prompt for guided-exercise responses.
    """

    exercise_state = state.get("exercise_state", {})
    # Only trust exercise_modality when an exercise is actually active.
    approach = (
        exercise_state.get("exercise_modality")
        if exercise_state.get("exercise_type")
        else None
    ) or _read_approach(state)
    files = _knowledge_for_mode("guided_exercise", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(
        knowledge, _GUIDED_EXERCISE_INSTRUCTIONS, state
    )


def build_technique_system_prompt(state: AgentState) -> str:
    """Build the system prompt for technique-mode responses.

    In technique mode, the therapeutic approach drives the response
    shape. The approach knowledge files are loaded as the primary
    behavioral guidance — the technique instructions just say "follow
    the approach's process guidance." This is the one response style
    where the approach is loud, not background.

    A valid therapeutic approach MUST be set in state. If no
    approach is active, this falls back to the supportive prompt to
    avoid generating an ungrounded response.

    Args:
        state: Current graph state.

    Returns:
        System prompt for technique-mode responses.
    """

    approach = _read_approach(state)
    if not approach or approach == "none":
        # No approach active — technique mode without an approach
        # doesn't make sense. Fall back to supportive.
        return build_supportive_system_prompt(state)

    files = _knowledge_for_mode("technique", approach)
    knowledge = _compose(*files)
    return _compose_system_prompt_with_state(knowledge, _TECHNIQUE_INSTRUCTIONS, state)


def build_therapeutic_response_prompt(
    state: AgentState,
    *,
    mode: str,
    step_directive: str | None = None,
) -> str:
    """Build the user/task prompt for any therapeutic mode.

    All modes share the same user-prompt structure — the system
    prompt (which differs per mode) is what shapes the response
    character. The user prompt provides the conversation context
    and current message.

    Args:
        state: Current graph state with history and working memory.
        mode: The dispatched mode name, injected as context for
            observability in the prompt.
        step_directive: For multi-turn modes (guided_exercise), an
            explicit instruction about what the LLM should generate.
            This bridges the node's deterministic state transition
            to the LLM's prose generation — the node knows *which*
            step to produce, and tells the LLM via this directive.

    Returns:
        User/task prompt for the therapeutic response LLM call.
    """

    memory_block = _format_working_memory(state)
    directive_block = f"\n\nStep directive:\n{step_directive}" if step_directive else ""

    return (
        f"Write the next assistant message for a mental health support "
        f"conversation in {mode} mode.\n\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {state['message']}"
        f"{directive_block}"
    )
