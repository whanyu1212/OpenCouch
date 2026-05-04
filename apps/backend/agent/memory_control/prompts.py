"""Prompt builders for memory-control routing classification."""

from __future__ import annotations

from agent.conversation import format_recent_history
from agent.state import AgentState


def build_memory_control_prompt(state: AgentState) -> str:
    """Build the LLM prompt for ambiguous memory-control routing.

    Args:
        state (AgentState): Current graph state.

    Returns:
        str: Prompt asking for a structured memory-control decision.
    """

    recent_history = format_recent_history(state, limit=6, empty="(none)")
    return (
        "Decide whether the user's message is an explicit request to manage "
        "OpenCouch's saved memory before ordinary therapeutic routing.\n\n"
        "Route to memory control only for requests to list or inspect saved "
        "memories, check memory status, enable or disable proactive recall, "
        "delete a concrete saved memory, or save a preference about how the "
        "assistant should respond or use memory.\n\n"
        "Do not route ordinary autobiographical facts, requests for help with "
        "human memory, or reflective statements such as 'I remember...', "
        "'I keep forgetting...', or 'help me remember to...'. Do not route "
        "new facts like names, pets, places, or life details; normal memory "
        "extraction handles those later. Use action_type='none' when uncertain.\n\n"
        "For forget_by_query, provide a concrete saved-memory target from the "
        "message or recent conversation. Do not confirm deletion; the memory "
        "control node will ask the user before deleting anything.\n\n"
        "For save_preference, only save response or memory-use preferences. "
        "Return rule_text as a concise second-person rule, for example "
        "'You prefer concise replies.'\n\n"
        "Recent conversation:\n"
        f"{recent_history}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


def build_memory_control_system_prompt() -> str:
    """Build the system prompt for the memory-control classifier.

    Returns:
        str: System instruction for structured memory-control routing.
    """

    return (
        "You are a strict routing classifier. Return only the structured "
        "decision. You do not answer the user."
    )
