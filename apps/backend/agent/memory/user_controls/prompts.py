"""Compatibility shim: re-exports moved to ``agent.memory.prompts.control``."""

from agent.memory.prompts.control import (
    build_memory_control_prompt,
    build_memory_control_system_prompt,
)

__all__ = [
    "build_memory_control_prompt",
    "build_memory_control_system_prompt",
]
