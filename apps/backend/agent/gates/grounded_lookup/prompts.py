"""Prompt builders for grounded lookup routing classification."""

from __future__ import annotations

from agent.conversation import format_recent_history
from agent.state import AgentState


def build_grounded_lookup_prompt(state: AgentState) -> str:
    """Build the LLM prompt for ambiguous grounded-lookup routing.

    Args:
        state (AgentState): Current graph state.

    Returns:
        str: Prompt asking for a structured lookup-routing decision.
    """

    recent_history = format_recent_history(state, limit=6, empty="(none)")
    return (
        "Decide whether the user's message should route to grounded web/current "
        "factual lookup before therapeutic response generation.\n\n"
        "Route to lookup only when the user is asking for external factual, "
        "current, official, research, evidence, price, eligibility, schedule, "
        "resource, URL, product, or service information that should be verified "
        "outside the conversation.\n\n"
        "Do not route to lookup for subjective therapeutic reassurance, emotional "
        "validation, relationship advice, or questions like 'am I overreacting?', "
        "'am I a bad person?', 'is it normal to feel this way?', or 'what should "
        "I do about this feeling?'.\n\n"
        "If lookup is needed, set should_lookup=true and provide a concise search "
        "query. If uncertain, set should_lookup=false.\n\n"
        "Recent conversation:\n"
        f"{recent_history}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


def build_grounded_lookup_system_prompt() -> str:
    """Build the system prompt for the grounded-lookup classifier.

    Returns:
        str: System instruction for structured lookup routing.
    """

    return (
        "You are a strict routing classifier. Return only the structured "
        "decision. You do not answer the user."
    )
