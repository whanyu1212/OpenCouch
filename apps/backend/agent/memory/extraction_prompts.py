"""Compatibility shim: re-exports moved to ``agent.memory.prompts.extraction``."""

from agent.memory.prompts.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)

__all__ = [
    "build_extraction_system_prompt",
    "build_extraction_user_prompt",
]
