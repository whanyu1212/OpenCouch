"""Reusable core prompt fragments shared across agent nodes."""

from __future__ import annotations

from agent.prompts.loader import compose_knowledge_sections


def build_core_system_prompt() -> str:
    """Build the shared identity and boundary prompt fragment.

    Returns:
        The shared core system prompt assembled from repo knowledge files.
    """

    return compose_knowledge_sections(
        "soul.md",
        "identity.md",
        "policy/boundaries.md",
        "policy/privacy.md",
    )
