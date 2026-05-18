"""Memory-layer prompt builders.

This package houses the LLM prompt builders for the memory subsystem. Builders
are grouped here so that prompt versioning and global tweaks have a single home.

The legacy import paths
``agent.memory.summarization_prompts``
remain valid as compatibility shims that re-export from this package.
"""

from agent.memory.prompts.summarization import (
    build_summarization_system_prompt,
    build_summarization_user_prompt,
)

__all__ = [
    "build_summarization_system_prompt",
    "build_summarization_user_prompt",
]
