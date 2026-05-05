"""Compatibility shim: re-exports moved to ``agent.memory.prompts.procedural``."""

from agent.memory.prompts.procedural import (
    build_procedural_writer_system_prompt,
    build_procedural_writer_user_prompt,
)

__all__ = [
    "build_procedural_writer_system_prompt",
    "build_procedural_writer_user_prompt",
]
