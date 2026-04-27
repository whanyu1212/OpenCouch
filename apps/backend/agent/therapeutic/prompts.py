"""System prompt builders for therapeutic response modes.

This module remains the public compatibility surface. Prompt source selection,
state-context formatting, instruction text, and builders live under
``agent.therapeutic.prompting``.
"""

from __future__ import annotations

from agent.therapeutic.prompting.builders import (
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
from agent.therapeutic.prompting.context import (
    _format_working_memory,
    _format_procedural_rules_block,
    _format_recall_toggle_constraint,
    _has_episodic_context,
)
from agent.therapeutic.prompting.instructions import (
    _CLARIFYING_INSTRUCTIONS,
    _CLOSING_INSTRUCTIONS,
    _CONTINUITY_FILE,
    _GUIDED_EXERCISE_INSTRUCTIONS,
    _PSYCHOEDUCATION_INSTRUCTIONS,
    _REFLECTIVE_INSTRUCTIONS,
    _SUPPORTIVE_INSTRUCTIONS,
    _TECHNIQUE_INSTRUCTIONS,
)
from agent.therapeutic.prompting.sources import (
    _MODALITY_FILES,
    _MODE_BASE_KNOWLEDGE,
    _knowledge_for_mode,
)

__all__ = [
    "_CLARIFYING_INSTRUCTIONS",
    "_CLOSING_INSTRUCTIONS",
    "_CONTINUITY_FILE",
    "_GUIDED_EXERCISE_INSTRUCTIONS",
    "_MODALITY_FILES",
    "_MODE_BASE_KNOWLEDGE",
    "_PSYCHOEDUCATION_INSTRUCTIONS",
    "_REFLECTIVE_INSTRUCTIONS",
    "_SUPPORTIVE_INSTRUCTIONS",
    "_TECHNIQUE_INSTRUCTIONS",
    "_compose_system_prompt_with_state",
    "_format_procedural_rules_block",
    "_format_recall_toggle_constraint",
    "_format_working_memory",
    "_has_episodic_context",
    "_knowledge_for_mode",
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
