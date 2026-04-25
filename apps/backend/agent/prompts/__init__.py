"""Prompt builder re-exports for the agent graph."""

from __future__ import annotations

from agent.prompts.crisis import (  # noqa: E402
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)
from agent.prompts.shared import (
    CORE_SOURCES,
    compose_sources,
    format_recent_history,
    load_prompt_source,
    prompt_sources_root,
)

__all__ = [
    # Shared
    "CORE_SOURCES",
    "prompt_sources_root",
    "load_prompt_source",
    "compose_sources",
    "format_recent_history",
    # Crisis builders
    "build_crisis_classifier_prompt",
    "build_crisis_classifier_system_prompt",
    "build_crisis_response_prompt",
    "build_crisis_response_system_prompt",
]
