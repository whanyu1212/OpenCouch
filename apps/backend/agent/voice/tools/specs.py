"""Canonical metadata for OpenCouch voice tools."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent.voice.tools.context import VoiceToolHandler
from agent.voice.tools.handlers import (
    _handle_answer_grounded_lookup,
    _handle_cancel_memory_deletion,
    _handle_confirm_memory_deletion,
    _handle_get_crisis_support_template,
    _handle_list_guided_exercise_skills,
    _handle_load_guided_exercise_skill,
    _handle_load_therapeutic_response_skill,
    _handle_lookup_crisis_resources,
    _handle_prepare_memory_deletion_by_index,
    _handle_prepare_memory_deletion_by_query,
    _handle_recall_saved_memory,
    _handle_record_guided_exercise_progress,
    _handle_save_response_preference,
    _handle_set_proactive_memory_recall,
    _handle_show_memory_status,
    _handle_start_guided_exercise,
    _handle_show_saved_memory,
    _handle_wait_for_user,
)


@dataclass(frozen=True)
class VoiceToolSpec:
    """Canonical schema, dispatch, and route metadata for one voice tool."""

    name: str
    description: str
    properties: Mapping[str, Any]
    required: tuple[str, ...]
    handler: VoiceToolHandler
    requires_context: bool = True
    persistent_only: bool = False
    memory_mutator: bool = False
    intent_gated_mutator: bool = False
    route: str | None = None
    response_style: str | None = None
    route_priority: int | None = None

    def __post_init__(self) -> None:
        missing_required = set(self.required) - set(self.properties)
        if missing_required:
            raise ValueError(
                f"{self.name!r} requires undefined properties: "
                f"{sorted(missing_required)!r}"
            )
        has_route = self.route is not None or self.response_style is not None
        if has_route and (self.route is None or self.response_style is None):
            raise ValueError(
                f"{self.name!r} must define both route and response_style together"
            )
        if has_route != (self.route_priority is not None):
            raise ValueError(
                f"{self.name!r} must define route_priority with route metadata"
            )

    def as_realtime_function_tool(self) -> dict[str, Any]:
        """Build an isolated OpenAI Realtime function-tool schema."""

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": deepcopy(dict(self.properties)),
                "required": list(self.required),
                "additionalProperties": False,
            },
        }


def _with_user_quote_property(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        **properties,
        "user_quote": {
            "type": "string",
            "description": (
                "Exact recent user words that explicitly requested this memory "
                "change. The server verifies this quote before executing."
            ),
        },
    }


VOICE_TOOL_SPECS: tuple[VoiceToolSpec, ...] = (
    VoiceToolSpec(
        name="wait_for_user",
        description=(
            "Call this when the latest audio should not receive a spoken "
            "reply, such as silence, background noise, hold music, TV audio, "
            "side conversation, or speech not addressed to OpenCouch. Side "
            "effects: none. After calling this tool, do not respond."
        ),
        properties={},
        required=(),
        handler=_handle_wait_for_user,
        requires_context=False,
    ),
    VoiceToolSpec(
        name="show_saved_memory",
        description=(
            "Show a concise overview of saved facts, session summaries, "
            "and preferences for the current OpenCouch user. Side effects: none."
        ),
        properties={},
        required=(),
        handler=_handle_show_saved_memory,
        persistent_only=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=11,
    ),
    VoiceToolSpec(
        name="recall_saved_memory",
        description=(
            "Query the user's saved memory for facts and session arcs "
            "relevant to a specific topic. Use when the user mentions a "
            "topic that might have prior saved context (e.g. an ongoing "
            "concern, a relationship, a past exercise); do not call "
            "every turn. Refused server-side in incognito mode and when "
            "the user has proactive recall disabled. Side effects: none."
        ),
        properties={
            "query": {
                "type": "string",
                "description": (
                    "Topic to search saved memory for. Use the user's "
                    "own words when possible (e.g. 'work stress', "
                    "'partner conversation', 'sleep')."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of entries to return. Defaults to 5; "
                    "use a small value to keep voice replies concise."
                ),
            },
        },
        required=("query",),
        handler=_handle_recall_saved_memory,
        persistent_only=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=7,
    ),
    VoiceToolSpec(
        name="save_response_preference",
        description=(
            "Save an explicit response-style or memory-use preference. "
            "Use only when the user explicitly asks you to remember such "
            "a preference. Side effects: writes durable procedural memory."
        ),
        properties=_with_user_quote_property(
            {
                "preference_text": {
                    "type": "string",
                    "description": "The explicit user preference to save.",
                }
            }
        ),
        required=("preference_text", "user_quote"),
        handler=_handle_save_response_preference,
        persistent_only=True,
        memory_mutator=True,
        intent_gated_mutator=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=8,
    ),
    VoiceToolSpec(
        name="set_proactive_memory_recall",
        description=(
            "Turn proactive memory recall on or off for the current "
            "OpenCouch user. Use only when explicitly requested."
        ),
        properties=_with_user_quote_property(
            {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether proactive recall should be enabled.",
                }
            }
        ),
        required=("enabled", "user_quote"),
        handler=_handle_set_proactive_memory_recall,
        persistent_only=True,
        memory_mutator=True,
        intent_gated_mutator=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=9,
    ),
    VoiceToolSpec(
        name="prepare_memory_deletion_by_index",
        description=(
            "Prepare deletion of a saved memory selected by visible "
            "kind and one-based index. Side effects: pending deletion only."
        ),
        properties=_with_user_quote_property(
            {
                "target_kind": {
                    "type": "string",
                    "enum": ["fact", "session", "rule"],
                },
                "target_index": {
                    "type": "integer",
                    "description": "One-based index from the visible memory list.",
                },
            }
        ),
        required=("target_kind", "target_index", "user_quote"),
        handler=_handle_prepare_memory_deletion_by_index,
        persistent_only=True,
        memory_mutator=True,
        intent_gated_mutator=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=5,
    ),
    VoiceToolSpec(
        name="prepare_memory_deletion_by_query",
        description=(
            "Prepare deletion of a saved memory selected by a concrete "
            "query. Side effects: pending deletion only."
        ),
        properties=_with_user_quote_property(
            {
                "query": {
                    "type": "string",
                    "description": "Concrete saved-memory deletion query.",
                }
            }
        ),
        required=("query", "user_quote"),
        handler=_handle_prepare_memory_deletion_by_query,
        persistent_only=True,
        memory_mutator=True,
        intent_gated_mutator=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=6,
    ),
    VoiceToolSpec(
        name="confirm_memory_deletion",
        description=(
            "Confirm and perform a pending saved-memory deletion. Use "
            "only when the user clearly confirms."
        ),
        properties={},
        required=(),
        handler=_handle_confirm_memory_deletion,
        persistent_only=True,
        memory_mutator=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=4,
    ),
    VoiceToolSpec(
        name="cancel_memory_deletion",
        description=(
            "Cancel a pending saved-memory deletion. Use only when the "
            "user cancels or declines."
        ),
        properties={},
        required=(),
        handler=_handle_cancel_memory_deletion,
        persistent_only=True,
        memory_mutator=True,
        route="memory_control",
        response_style="memory_control",
        route_priority=3,
    ),
    VoiceToolSpec(
        name="show_memory_status",
        description=(
            "Show whether memory is enabled and summarize saved-memory "
            "counts for the current OpenCouch user. Side effects: none."
        ),
        properties={},
        required=(),
        handler=_handle_show_memory_status,
        route="memory_control",
        response_style="memory_control",
        route_priority=10,
    ),
    VoiceToolSpec(
        name="load_therapeutic_response_skill",
        description=(
            "Load a side-effect-free therapeutic response-style skill block "
            "for ordinary non-crisis replies. Side effects: none."
        ),
        properties={
            "response_style": {
                "type": "string",
                "enum": [
                    "supportive",
                    "reflective",
                    "clarifying",
                    "psychoeducation",
                    "closing",
                    "technique",
                ],
                "description": "Therapeutic response style to load.",
            },
            "therapeutic_approach": {
                "type": ["string", "null"],
                "description": "Optional therapeutic approach overlay.",
            },
        },
        required=("response_style",),
        handler=_handle_load_therapeutic_response_skill,
    ),
    VoiceToolSpec(
        name="answer_grounded_lookup",
        description=(
            "Answer an explicit current, factual, official, source-backed, "
            "resource-seeking, or externally verifiable request using the "
            "OpenCouch grounded lookup service. Side effects: none."
        ),
        properties={
            "query": {
                "type": "string",
                "description": "The concise factual lookup query to verify.",
            }
        },
        required=("query",),
        handler=_handle_answer_grounded_lookup,
        route="grounded_lookup",
        response_style="grounded_lookup",
        route_priority=2,
    ),
    VoiceToolSpec(
        name="lookup_crisis_resources",
        description=(
            "Look up verified crisis resources for the current crisis turn "
            "using the user's stated location when available. Side effects: none."
        ),
        properties={},
        required=(),
        handler=_handle_lookup_crisis_resources,
        route="crisis",
        response_style="crisis_response",
        route_priority=0,
    ),
    VoiceToolSpec(
        name="get_crisis_support_template",
        description=(
            "Load a deterministic crisis-response safety scaffold to shape "
            "the current spoken reply when the user expresses self-harm, "
            "suicidal ideation, or imminent danger. It does not replace "
            "lookup_crisis_resources and must not be used to invent phone "
            "numbers. Verified resource details from a prior "
            "lookup_crisis_resources call are reused automatically. Side "
            "effects: none."
        ),
        properties={
            "risk_level": {
                "type": "string",
                "enum": ["moderate", "high", "imminent"],
                "description": (
                    "Severity of the current crisis turn: moderate, high, or imminent."
                ),
            },
            "inferred_location": {
                "type": ["string", "null"],
                "description": (
                    "User-stated location, only if the user already shared it."
                ),
            },
        },
        required=("risk_level",),
        handler=_handle_get_crisis_support_template,
        route="crisis",
        response_style="crisis_response",
        route_priority=1,
    ),
    VoiceToolSpec(
        name="list_guided_exercise_skills",
        description=(
            "List metadata-only guided exercises available for the current "
            "channel and therapeutic approach. Side effects: none."
        ),
        properties={
            "therapeutic_approach": {
                "type": ["string", "null"],
                "description": "Optional therapeutic approach filter.",
            },
            "channel": {
                "type": ["string", "null"],
                "description": "Optional delivery channel filter.",
            },
        },
        required=(),
        handler=_handle_list_guided_exercise_skills,
        route="guided_exercise",
        response_style="guided_exercise",
        route_priority=12,
    ),
    VoiceToolSpec(
        name="start_guided_exercise",
        description=(
            "Start one registered guided exercise and return the initial skill "
            "context. Use only after selecting an exercise for the user. Side "
            "effects: active exercise state update."
        ),
        properties={
            "exercise_type": {
                "type": "string",
                "description": "Registered guided exercise skill identifier.",
            },
            "therapeutic_approach": {
                "type": ["string", "null"],
                "description": (
                    "Optional therapeutic approach captured for this exercise."
                ),
            },
        },
        required=("exercise_type",),
        handler=_handle_start_guided_exercise,
        route="guided_exercise",
        response_style="guided_exercise",
        route_priority=13,
    ),
    VoiceToolSpec(
        name="load_guided_exercise_skill",
        description=(
            "Load the runtime-selected guided-exercise skill block for "
            "the current step and action. Side effects: none."
        ),
        properties={
            "exercise_type": {
                "type": "string",
                "description": "Registered guided exercise skill identifier.",
            },
            "runtime_action": {
                "type": "string",
                "description": "Runtime-approved action for this step.",
            },
            "current_step_index": {
                "type": ["integer", "null"],
                "description": "Current exercise step index when applicable.",
            },
        },
        required=("exercise_type", "runtime_action"),
        handler=_handle_load_guided_exercise_skill,
        route="guided_exercise",
        response_style="guided_exercise",
        route_priority=14,
    ),
    VoiceToolSpec(
        name="record_guided_exercise_progress",
        description=(
            "Record the user's latest response to the active guided-exercise "
            "step in Realtime voice. The runtime validates the expected skill "
            "and step, computes the next action, and may update active "
            "exercise state."
        ),
        properties={
            "expected_skill_id": {
                "type": "string",
                "description": "Active skill id expected by the runtime.",
            },
            "expected_step_id": {
                "type": "string",
                "description": "Active step id expected by the runtime.",
            },
            "outcome": {
                "type": "string",
                "enum": ["complete", "partial", "hold", "stuck", "exit", "unsafe"],
                "description": "Observed outcome from the user response.",
            },
            "user_response_summary": {
                "type": "string",
                "description": "Brief summary of what the user did or said.",
            },
        },
        required=(
            "expected_skill_id",
            "expected_step_id",
            "outcome",
            "user_response_summary",
        ),
        handler=_handle_record_guided_exercise_progress,
        route="guided_exercise",
        response_style="guided_exercise",
        route_priority=15,
    ),
)

VOICE_TOOL_SPECS_BY_NAME: dict[str, VoiceToolSpec] = {
    spec.name: spec for spec in VOICE_TOOL_SPECS
}
if len(VOICE_TOOL_SPECS_BY_NAME) != len(VOICE_TOOL_SPECS):
    raise ValueError("Voice tool specs must have unique names")

__all__ = [
    "VOICE_TOOL_SPECS",
    "VOICE_TOOL_SPECS_BY_NAME",
    "VoiceToolSpec",
]
