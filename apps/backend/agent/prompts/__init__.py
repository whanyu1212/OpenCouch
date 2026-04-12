"""Prompt builders for the agent graph.

Only the crisis-mode builders survive the legacy cleanup. Other prompt
machinery (response modes, modalities, knowledge catalog) will be redesigned
alongside the therapeutic graph rebuild.
"""

from agent.prompts.crisis import (
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)

__all__ = [
    "build_crisis_classifier_prompt",
    "build_crisis_classifier_system_prompt",
    "build_crisis_response_prompt",
    "build_crisis_response_system_prompt",
]
