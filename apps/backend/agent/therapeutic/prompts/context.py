"""State-derived prompt context for therapeutic response styles."""

from __future__ import annotations

from agent.state import AgentState
from agent.memory.entries import format_working_memory_entries


def _format_working_memory(state: AgentState) -> str:
    """Format working memory snippets for prompt injection.

    If there is no working memory (incognito mode, or no facts extracted
    yet), returns an empty string so the prompt section is simply absent
    rather than showing an empty list. When proactive recall is off, the
    response prompt receives only a private availability signal, not the
    specific saved facts or session details.

    Args:
        state: Current graph state.

    Returns:
        Formatted working-memory block, or an empty string when absent.
    """

    snippets = format_working_memory_entries(state.get("working_memory", []))
    if not snippets:
        return ""
    if not _proactive_recall_enabled(state):
        return (
            "\nPrivate memory context is available for this turn, but proactive "
            "recall is off. Use it only as a silent signal for pacing and "
            "continuity. Do NOT mention or imply any specific saved names, "
            "facts, events, quotes, past sessions, or memories.\n"
        )
    joined = "\n".join(f"- {snippet}" for snippet in snippets)
    return f"\nRelevant context from past sessions:\n{joined}\n"


# Procedural rules are always injected as silent constraints. The recall
# toggle only governs explicit references to semantic and episodic memory.


def _format_procedural_rules_block(state: AgentState) -> str:
    """Format the user's procedural rules as a silent-constraint block.

    Returns the empty string when no rules exist. When rules are present,
    returns a prompt suffix that lists them with explicit instructions to
    follow them silently — never quote, cite, or narrate compliance.

    The block is unconditional with respect to the recall toggle. Rules
    are applied on every response regardless of whether the user has
    enabled or disabled proactive memory recall.

    Args:
        state: Current graph state.

    Returns:
        Formatted procedural-rules block, or an empty string when absent.
    """

    procedural_profile = state.get("procedural_profile", {}) or {}
    rules = procedural_profile.get("procedural_rules") or []
    if not rules:
        return ""

    rule_lines = "\n".join(f"- {rule}" for rule in rules)
    return (
        "\n\n═══ Style rules from past conversations with this user ═══\n"
        f"{rule_lines}\n"
        "\n"
        "Follow these rules silently. Do NOT quote them, cite them, or "
        "narrate your compliance with them (e.g., never say 'as per your "
        "earlier request...'). The user already knows they asked for "
        "these; acknowledging them makes the interaction feel "
        "customer-service-y. Just apply the rules as part of how you "
        "respond."
    )


def _format_recall_toggle_constraint(state: AgentState) -> str:
    """Format the recall-toggle constraint block for the system prompt.

    Returns a prompt suffix whose content depends on
    ``procedural_profile.proactive_recall_enabled``:

    - **When False (default)**: tells the model to use retrieved memories
      for silent shaping but NOT to explicitly reference past sessions or
      past statements. This is the "invisible but effective" mode users
      get when they turn off proactive recall.
    - **When True**: relaxes the constraint so the model may reference
      past memories sparingly when they add value to the current moment.

    The constraint refers specifically to "past sessions or past
    statements" — semantic facts and episodic summaries — and does NOT
    govern procedural rules, which are separately handled by
    :func:`_format_procedural_rules_block`.

    Args:
        state: Current graph state.

    Returns:
        Formatted memory-reference guidance block.
    """

    if _proactive_recall_enabled(state):
        # Recall ON: relaxed constraint.
        return (
            "\n\n═══ Memory reference guidance (proactive recall: ON) ═══\n"
            "You may reference relevant past memories when it adds value "
            "to the current moment, but do so sparingly and never for "
            "emotionally charged topics without strong contextual fit."
        )

    # Recall OFF (default): silent-shaping constraint.
    return (
        "\n\n═══ Memory reference guidance (proactive recall: OFF) ═══\n"
        "do NOT explicitly reference past sessions, saved memories, or "
        "specific saved facts unless the user has just asked about them. "
        "If a private memory-availability signal is present, treat it only "
        "as a silent pacing and continuity cue."
    )


def _proactive_recall_enabled(state: AgentState) -> bool:
    procedural_profile = state.get("procedural_profile", {}) or {}
    return bool(procedural_profile.get("proactive_recall_enabled", False))


def _has_episodic_context(state: AgentState) -> bool:
    """Return whether the user has prior session history available.

    Args:
        state: Current graph state.

    Returns:
        Whether working memory contains episodic context.
    """

    working_memory = state.get("working_memory", [])
    return any(entry.get("type") == "episodic" for entry in working_memory)
