"""OpenAI Realtime function tool schemas for voice sessions."""

from __future__ import annotations

from typing import Any

_SUPPORTED_VOICE_TOOL_NAMES = {
    "wait_for_user",
    "show_memory_status",
    "show_saved_memory",
    "recall_saved_memory",
    "set_proactive_memory_recall",
    "save_response_preference",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
    "confirm_memory_deletion",
    "cancel_memory_deletion",
    "answer_grounded_lookup",
    "lookup_crisis_resources",
    "get_crisis_support_template",
    "list_guided_exercise_skills",
    "load_therapeutic_response_skill",
    "load_guided_exercise_skill",
    "record_guided_exercise_progress",
}

_PERSISTENT_ONLY_TOOL_NAMES = {
    "show_saved_memory",
    "recall_saved_memory",
    "set_proactive_memory_recall",
    "save_response_preference",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
    "confirm_memory_deletion",
    "cancel_memory_deletion",
}

_VOICE_MEMORY_MUTATOR_TOOL_NAMES = {
    "set_proactive_memory_recall",
    "save_response_preference",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
    "confirm_memory_deletion",
    "cancel_memory_deletion",
}

_INTENT_GATED_MUTATOR_TOOL_NAMES = {
    "set_proactive_memory_recall",
    "save_response_preference",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
}


def build_voice_realtime_tools(*, memory_mode: str) -> list[dict[str, Any]]:
    """Return the narrow function-tool surface exposed to Realtime."""

    persistent = memory_mode.strip().lower() == "persistent"
    tools: list[dict[str, Any]] = [
        _function_tool(
            name="wait_for_user",
            description=(
                "Call this when the latest audio should not receive a spoken "
                "reply, such as silence, background noise, hold music, TV audio, "
                "side conversation, or speech not addressed to OpenCouch. Side "
                "effects: none. After calling this tool, do not respond."
            ),
            properties={},
            required=[],
        ),
        _function_tool(
            name="show_memory_status",
            description=(
                "Show whether memory is enabled and summarize saved-memory "
                "counts for the current OpenCouch user. Side effects: none."
            ),
            properties={},
            required=[],
        ),
        _function_tool(
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
            required=["response_style"],
        ),
        _function_tool(
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
            required=["query"],
        ),
        _function_tool(
            name="lookup_crisis_resources",
            description=(
                "Look up verified crisis resources for the current crisis turn "
                "using the user's stated location when available. Side effects: none."
            ),
            properties={},
            required=[],
        ),
        _function_tool(
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
                        "Severity of the current crisis turn: moderate, high, "
                        "or imminent."
                    ),
                },
                "inferred_location": {
                    "type": ["string", "null"],
                    "description": (
                        "User-stated location, only if the user already shared it."
                    ),
                },
            },
            required=["risk_level"],
        ),
        _function_tool(
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
            required=[],
        ),
        _function_tool(
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
            required=["exercise_type", "runtime_action"],
        ),
        _function_tool(
            name="record_guided_exercise_progress",
            description=(
                "Record the user's latest response to the active guided-exercise "
                "step. Side effects: active exercise state may update."
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
            required=[
                "expected_skill_id",
                "expected_step_id",
                "outcome",
                "user_response_summary",
            ],
        ),
    ]

    if persistent:
        tools[1:1] = [
            _function_tool(
                name="show_saved_memory",
                description=(
                    "Show a concise overview of saved facts, session summaries, "
                    "and preferences for the current OpenCouch user. Side effects: none."
                ),
                properties={},
                required=[],
            ),
            _function_tool(
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
                required=["query"],
            ),
            _function_tool(
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
                required=["preference_text", "user_quote"],
            ),
            _function_tool(
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
                required=["enabled", "user_quote"],
            ),
            _function_tool(
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
                            "description": (
                                "One-based index from the visible memory list."
                            ),
                        },
                    }
                ),
                required=["target_kind", "target_index", "user_quote"],
            ),
            _function_tool(
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
                required=["query", "user_quote"],
            ),
            _function_tool(
                name="confirm_memory_deletion",
                description=(
                    "Confirm and perform a pending saved-memory deletion. Use "
                    "only when the user clearly confirms."
                ),
                properties={},
                required=[],
            ),
            _function_tool(
                name="cancel_memory_deletion",
                description=(
                    "Cancel a pending saved-memory deletion. Use only when the "
                    "user cancels or declines."
                ),
                properties={},
                required=[],
            ),
        ]

    return tools


def _function_tool(
    *,
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
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
