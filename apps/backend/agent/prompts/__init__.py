"""Prompt builders and reusable fragments for the agent."""

from agent.prompts.builders import (
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
    build_guided_exercise_response_prompt,
    build_guided_exercise_system_prompt,
    build_orientation_response_prompt,
    build_orientation_system_prompt,
    build_out_of_scope_response_prompt,
    build_out_of_scope_system_prompt,
    build_realignment_response_prompt,
    build_realignment_system_prompt,
    build_reflection_response_prompt,
    build_reflection_system_prompt,
    build_therapeutic_response_prompt,
    build_therapeutic_system_prompt,
    format_recent_history,
)
from agent.prompts.catalog import Modality, ResponseMode
from agent.prompts.core import build_core_system_prompt
from agent.prompts.modes import build_mode_prompt, build_modality_prompt, build_system_prompt

__all__ = [
    "Modality",
    "ResponseMode",
    "build_core_system_prompt",
    "build_crisis_classifier_prompt",
    "build_crisis_classifier_system_prompt",
    "build_crisis_response_prompt",
    "build_crisis_response_system_prompt",
    "build_guided_exercise_response_prompt",
    "build_guided_exercise_system_prompt",
    "build_orientation_response_prompt",
    "build_orientation_system_prompt",
    "build_mode_prompt",
    "build_modality_prompt",
    "build_out_of_scope_response_prompt",
    "build_out_of_scope_system_prompt",
    "build_realignment_response_prompt",
    "build_realignment_system_prompt",
    "build_reflection_response_prompt",
    "build_reflection_system_prompt",
    "build_system_prompt",
    "build_therapeutic_response_prompt",
    "build_therapeutic_system_prompt",
    "format_recent_history",
]
