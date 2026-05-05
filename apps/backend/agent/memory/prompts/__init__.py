"""Memory-layer prompt builders.

This package houses the LLM prompt builders for the memory subsystem:
extraction, procedural rule writing, session summarization, and
memory-management routing. Builders are grouped here so that prompt
versioning and global tweaks have a single home.

The legacy import paths
``agent.memory.extraction_prompts``,
``agent.memory.procedural_prompts``,
``agent.memory.summarization_prompts``, and
``agent.memory.user_controls.prompts``
remain valid as compatibility shims that re-export from this package.
"""

from agent.memory.prompts.control import (
    build_memory_control_prompt,
    build_memory_control_system_prompt,
)
from agent.memory.prompts.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from agent.memory.prompts.procedural import (
    build_procedural_writer_system_prompt,
    build_procedural_writer_user_prompt,
)
from agent.memory.prompts.summarization import (
    build_summarization_system_prompt,
    build_summarization_user_prompt,
)

__all__ = [
    "build_extraction_system_prompt",
    "build_extraction_user_prompt",
    "build_procedural_writer_system_prompt",
    "build_procedural_writer_user_prompt",
    "build_summarization_system_prompt",
    "build_summarization_user_prompt",
    "build_memory_control_system_prompt",
    "build_memory_control_prompt",
]
