"""OpenAI Realtime function tool schemas for OpenCouch voice sessions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent.tools.crisis import execute_crisis_resource_lookup_tool
from agent.tools.grounded import execute_grounded_lookup_tool
from agent.tools.guided_exercise import (
    execute_guided_exercise_discovery_tool,
    execute_guided_exercise_progress_tool,
    execute_guided_exercise_skill_tool,
)
from agent.tools.memory import (
    execute_memory_tool_action,
    execute_read_only_memory_action,
)
from agent.tools.therapeutic import execute_therapeutic_response_skill_tool
from llm.base import BaseLLMClient

_SUPPORTED_VOICE_TOOL_NAMES = {
    "show_memory_status",
    "show_saved_memory",
    "set_proactive_memory_recall",
    "save_response_preference",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
    "confirm_memory_deletion",
    "cancel_memory_deletion",
    "answer_grounded_lookup",
    "lookup_crisis_resources",
    "list_guided_exercise_skills",
    "load_therapeutic_response_skill",
    "load_guided_exercise_skill",
    "record_guided_exercise_progress",
}

_PERSISTENT_ONLY_TOOL_NAMES = {
    "show_saved_memory",
    "set_proactive_memory_recall",
    "save_response_preference",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
    "confirm_memory_deletion",
    "cancel_memory_deletion",
}


def build_voice_realtime_tools(*, memory_mode: str) -> list[dict[str, Any]]:
    """Return the narrow function-tool surface exposed to Realtime."""

    persistent = memory_mode.strip().lower() == "persistent"
    tools: list[dict[str, Any]] = [
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
                name="save_response_preference",
                description=(
                    "Save an explicit response-style or memory-use preference. "
                    "Use only when the user explicitly asks you to remember such "
                    "a preference. Side effects: writes durable procedural memory."
                ),
                properties={
                    "preference_text": {
                        "type": "string",
                        "description": "The explicit user preference to save.",
                    }
                },
                required=["preference_text"],
            ),
            _function_tool(
                name="set_proactive_memory_recall",
                description=(
                    "Turn proactive memory recall on or off for the current "
                    "OpenCouch user. Use only when explicitly requested."
                ),
                properties={
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether proactive recall should be enabled.",
                    }
                },
                required=["enabled"],
            ),
            _function_tool(
                name="prepare_memory_deletion_by_index",
                description=(
                    "Prepare deletion of a saved memory selected by visible "
                    "kind and one-based index. Side effects: pending deletion only."
                ),
                properties={
                    "target_kind": {
                        "type": "string",
                        "enum": ["fact", "session", "rule"],
                    },
                    "target_index": {
                        "type": "integer",
                        "description": "One-based index from the visible memory list.",
                    },
                },
                required=["target_kind", "target_index"],
            ),
            _function_tool(
                name="prepare_memory_deletion_by_query",
                description=(
                    "Prepare deletion of a saved memory selected by a concrete "
                    "query. Side effects: pending deletion only."
                ),
                properties={
                    "query": {
                        "type": "string",
                        "description": "Concrete saved-memory deletion query.",
                    }
                },
                required=["query"],
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


async def execute_voice_tool_call(
    *,
    runtime: Any,
    tool_name: str,
    arguments: dict[str, object],
    thread_id: str,
    user_id: str | None,
    current_user_message: str,
    transcript: list[dict[str, object]],
    memory_mode: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, object]:
    """Execute one app-owned voice function tool call."""

    if tool_name not in _SUPPORTED_VOICE_TOOL_NAMES:
        raise ValueError(f"Unsupported voice tool: {tool_name!r}")

    effective_memory_mode = _effective_memory_mode(runtime, memory_mode)
    if effective_memory_mode == "incognito":
        if tool_name == "show_memory_status":
            return _incognito_memory_status_result()
        if tool_name in _PERSISTENT_ONLY_TOOL_NAMES:
            raise ValueError(f"{tool_name!r} is not available in incognito voice mode.")

    context = await runtime.build_voice_tool_context(
        thread_id=thread_id,
        user_id=user_id,
        current_user_message=current_user_message,
        transcript=transcript,
        llm_client=llm_client,
    )
    result: Any
    if tool_name == "show_memory_status":
        result = await execute_read_only_memory_action(context, {"type": "status"})
    elif tool_name == "show_saved_memory":
        result = await execute_read_only_memory_action(context, {"type": "list"})
    elif tool_name == "save_response_preference":
        result = await execute_memory_tool_action(
            context,
            {
                "type": "save_preference",
                "preference_text": str(arguments.get("preference_text") or ""),
            },
            side_effect="procedural_profile_update",
            retry_safe=False,
        )
    elif tool_name == "set_proactive_memory_recall":
        result = await execute_memory_tool_action(
            context,
            {
                "type": "set_recall",
                "enabled": bool(arguments.get("enabled")),
            },
            side_effect="procedural_profile_update",
            retry_safe=True,
        )
    elif tool_name == "prepare_memory_deletion_by_index":
        result = await execute_memory_tool_action(
            context,
            {
                "type": "forget_by_index",
                "target_kind": str(arguments.get("target_kind") or ""),
                "target_index": int(arguments.get("target_index") or 0),
            },
            side_effect="pending_deletion",
            retry_safe=True,
        )
    elif tool_name == "prepare_memory_deletion_by_query":
        result = await execute_memory_tool_action(
            context,
            {
                "type": "forget_by_query",
                "query": str(arguments.get("query") or ""),
            },
            side_effect="pending_deletion",
            retry_safe=True,
        )
    elif tool_name == "confirm_memory_deletion":
        result = await execute_memory_tool_action(
            context,
            {"type": "confirm_pending"},
            side_effect="delete_memory",
            retry_safe=False,
        )
    elif tool_name == "cancel_memory_deletion":
        result = await execute_memory_tool_action(
            context,
            {"type": "cancel_pending"},
            side_effect="cancel_pending",
            retry_safe=True,
        )
    elif tool_name == "answer_grounded_lookup":
        result = await execute_grounded_lookup_tool(
            context,
            query=str(arguments.get("query") or ""),
        )
    elif tool_name == "lookup_crisis_resources":
        result = await execute_crisis_resource_lookup_tool(context)
    elif tool_name == "list_guided_exercise_skills":
        result = await execute_guided_exercise_discovery_tool(
            context,
            therapeutic_approach=_optional_string(
                arguments.get("therapeutic_approach")
            ),
            channel=_optional_string(arguments.get("channel")),
        )
    elif tool_name == "load_therapeutic_response_skill":
        result = await execute_therapeutic_response_skill_tool(
            context,
            response_style=str(arguments.get("response_style") or "supportive"),
            therapeutic_approach=_optional_string(
                arguments.get("therapeutic_approach")
            ),
        )
    elif tool_name == "load_guided_exercise_skill":
        result = await execute_guided_exercise_skill_tool(
            context,
            exercise_type=str(arguments.get("exercise_type") or ""),
            runtime_action=str(arguments.get("runtime_action") or ""),
            current_step_index=_optional_int(arguments.get("current_step_index")),
        )
    elif tool_name == "record_guided_exercise_progress":
        result = await execute_guided_exercise_progress_tool(
            context,
            expected_skill_id=str(arguments.get("expected_skill_id") or ""),
            expected_step_id=str(arguments.get("expected_step_id") or ""),
            outcome=str(arguments.get("outcome") or "hold"),  # type: ignore[arg-type]
            user_response_summary=str(arguments.get("user_response_summary") or ""),
        )
    else:
        raise AssertionError(f"Unhandled voice tool: {tool_name!r}")

    if isinstance(result, BaseModel):
        return dict(result.model_dump(mode="json"))
    if isinstance(result, dict):
        return dict(result)
    return {"result": str(result)}


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _effective_memory_mode(runtime: Any, memory_mode: str | None) -> str:
    if memory_mode is not None:
        return memory_mode.strip().lower()
    runtime_mode = getattr(runtime, "memory_mode", None)
    if str(runtime_mode).lower().endswith("incognito"):
        return "incognito"
    return "persistent"


def _incognito_memory_status_result() -> dict[str, object]:
    return {
        "response_text": (
            "Memory is off for this incognito voice session. I won't save or "
            "use durable memory for this voice session."
        ),
        "memory_mode": "incognito",
        "memory_control": {"memory_mode": "incognito"},
        "side_effect": "none",
        "retry_safe": True,
    }
