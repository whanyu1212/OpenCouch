"""Therapeutic prompt builders split by responsibility.

The package is the public surface for therapeutic prompt construction.
``builders`` holds the system/task prompt builders, ``context`` formats
graph state into prompt-ready blocks, ``instructions`` carries the
per-style instruction text, and ``sources`` selects the markdown
knowledge files composed into each prompt.
"""

from __future__ import annotations

from agent.therapeutic.prompts.builders import (
    _compose_system_prompt_with_state,
    _read_approach,
    build_clarifying_system_prompt,
    build_closing_system_prompt,
    build_guided_exercise_system_prompt,
    build_psychoeducation_system_prompt,
    build_reflective_system_prompt,
    build_supportive_system_prompt,
    build_technique_system_prompt,
    build_therapeutic_response_prompt,
)
from agent.therapeutic.prompts.context import (
    _format_procedural_rules_block,
    _format_recall_toggle_constraint,
    _format_working_memory,
    _has_episodic_context,
)
from agent.therapeutic.prompts.instructions import (
    _CLARIFYING_INSTRUCTIONS,
    _CLOSING_INSTRUCTIONS,
    _CONTINUITY_FILE,
    _GUIDED_EXERCISE_INSTRUCTIONS,
    _PSYCHOEDUCATION_INSTRUCTIONS,
    _REFLECTIVE_INSTRUCTIONS,
    _SUPPORTIVE_INSTRUCTIONS,
    _TECHNIQUE_INSTRUCTIONS,
)
from agent.therapeutic.prompts.sources import (
    _RESPONSE_STYLE_BASE_KNOWLEDGE,
    _THERAPEUTIC_APPROACH_FILES,
    _knowledge_for_response_style,
)

__all__ = [
    "_CLARIFYING_INSTRUCTIONS",
    "_CLOSING_INSTRUCTIONS",
    "_CONTINUITY_FILE",
    "_GUIDED_EXERCISE_INSTRUCTIONS",
    "_PSYCHOEDUCATION_INSTRUCTIONS",
    "_REFLECTIVE_INSTRUCTIONS",
    "_RESPONSE_STYLE_BASE_KNOWLEDGE",
    "_SUPPORTIVE_INSTRUCTIONS",
    "_TECHNIQUE_INSTRUCTIONS",
    "_THERAPEUTIC_APPROACH_FILES",
    "_compose_system_prompt_with_state",
    "_format_procedural_rules_block",
    "_format_recall_toggle_constraint",
    "_format_working_memory",
    "_has_episodic_context",
    "_knowledge_for_response_style",
    "_read_approach",
    "build_clarifying_system_prompt",
    "build_closing_system_prompt",
    "build_guided_exercise_system_prompt",
    "build_psychoeducation_system_prompt",
    "build_reflective_system_prompt",
    "build_supportive_system_prompt",
    "build_technique_system_prompt",
    "build_therapeutic_response_prompt",
]
