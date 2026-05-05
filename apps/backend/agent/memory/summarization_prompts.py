"""Compatibility shim: re-exports moved to ``agent.memory.prompts.summarization``."""

from agent.memory.prompts.summarization import (
    build_summarization_system_prompt,
    build_summarization_user_prompt,
)

__all__ = [
    "build_summarization_system_prompt",
    "build_summarization_user_prompt",
]
