"""OpenAI Realtime function tool schemas for OpenCouch voice sessions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agent.tools.crisis import (
    execute_crisis_resource_lookup_tool,
    execute_crisis_support_template_tool,
)
from agent.tools.grounded import execute_grounded_lookup_tool
from agent.tools.guided_exercise import (
    execute_guided_exercise_discovery_tool,
    execute_guided_exercise_progress_tool,
    execute_guided_exercise_skill_tool,
)
from agent.memory.modes import resolve_effective_memory_mode
from agent.memory.operations.procedural_profile import aget_procedural_profile
from agent.memory.retrieval.service import load_memory_for_turn
from agent.state import resolve_owner_id
from agent.tools.memory import (
    execute_memory_tool_action,
    execute_read_only_memory_action,
)
from agent.tools.therapeutic import execute_therapeutic_response_skill_tool
from llm.base import BaseLLMClient

if TYPE_CHECKING:
    from agent.audit.models import CrisisResourceLookupStatus

_RECALL_DEFAULT_LIMIT = 5
_RECALL_MAX_LIMIT = 10

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

_RECENT_USER_TURN_LIMIT = 3
_MIN_USER_QUOTE_LENGTH = 8


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

    if tool_name == "wait_for_user":
        return {
            "response_text": "",
            "should_respond": False,
            "side_effect": "none",
        }

    effective_memory_mode = _effective_memory_mode(runtime, memory_mode)
    if effective_memory_mode == "incognito":
        if tool_name == "show_memory_status":
            return _incognito_memory_status_result()
        if tool_name in _PERSISTENT_ONLY_TOOL_NAMES:
            raise ValueError(f"{tool_name!r} is not available in incognito voice mode.")

    if tool_name in _VOICE_MEMORY_MUTATOR_TOOL_NAMES and not _has_owner_or_session_id(
        user_id=user_id,
        thread_id=thread_id,
    ):
        return _voice_mutator_refusal(
            reason="owner_or_session_missing",
            response_text=(
                "I can't change saved memory because this voice session does "
                "not include a user or session identifier."
            ),
        )

    if tool_name in _INTENT_GATED_MUTATOR_TOOL_NAMES and not _user_quote_matches_turn(
        arguments=arguments,
        current_user_message=current_user_message,
        transcript=transcript,
    ):
        return _voice_mutator_refusal(
            reason="user_intent_not_verified",
            response_text=(
                "I can't change saved memory from voice unless the tool call "
                "includes exact recent user words that asked for that change."
            ),
        )

    context = await runtime.voice.build_voice_tool_context(
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
    elif tool_name == "recall_saved_memory":
        result = await _execute_recall_saved_memory(context, arguments)
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
        await runtime.voice.persist_voice_crisis_resource_lookup(
            thread_id=thread_id,
            user_id=user_id,
            inferred_location=result.inferred_location,
            found_resources=result.found_resources,
            resource_lookup_status=result.resource_lookup_status,
        )
    elif tool_name == "get_crisis_support_template":
        result = await _execute_crisis_support_template(context, arguments)
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
    """Return the resolved binary memory mode for a tool dispatch.

    Delegates to ``resolve_effective_memory_mode`` so incognito is the
    floor: if the runtime is incognito, the request cannot escalate to
    persistent. Routes resolve this upstream; the helper stays as
    defense-in-depth for direct callers (tests, future channels).
    """

    return resolve_effective_memory_mode(
        getattr(runtime, "memory_mode", None), memory_mode
    )


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


def _has_owner_or_session_id(*, user_id: str | None, thread_id: str) -> bool:
    return bool((user_id or "").strip() or thread_id.strip())


def _user_quote_matches_turn(
    *,
    arguments: dict[str, object],
    current_user_message: str,
    transcript: list[dict[str, object]],
) -> bool:
    raw_quote = arguments.get("user_quote")
    user_quote = raw_quote if isinstance(raw_quote, str) else ""
    normalized_quote = _normalize_user_quote_text(user_quote)
    if len(normalized_quote) < _MIN_USER_QUOTE_LENGTH:
        return False

    evidence_text = _recent_user_evidence_text(
        current_user_message=current_user_message,
        transcript=transcript,
    )
    normalized_evidence = _normalize_user_quote_text(evidence_text)
    return bool(normalized_evidence and normalized_quote in normalized_evidence)


def _recent_user_evidence_text(
    *,
    current_user_message: str,
    transcript: list[dict[str, object]],
) -> str:
    parts: list[str] = []
    current = current_user_message.strip()
    if current:
        parts.append(current)

    user_turns: list[str] = []
    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        if role != "user":
            continue
        content = turn.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if text:
            user_turns.append(text)
    parts.extend(user_turns[-_RECENT_USER_TURN_LIMIT:])
    return " ".join(parts)


def _normalize_user_quote_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", cleaned.strip().lower())


def _voice_mutator_refusal(*, reason: str, response_text: str) -> dict[str, object]:
    return {
        "response_text": response_text,
        "refused": True,
        "reason": reason,
        "side_effect": "none",
        "retry_safe": True,
    }


async def _execute_crisis_support_template(
    context: Any,
    arguments: dict[str, object],
) -> dict[str, object]:
    """Load the deterministic crisis scaffold for a spoken crisis turn.

    Mirrors the SDK ``get_crisis_support_template`` tool: when the model does
    not pass resource details, reuse the latest ``lookup_crisis_resources``
    result recorded on the context so verified numbers thread through without
    the voice model re-stating them.
    """

    inferred_location = _optional_string(arguments.get("inferred_location")) or ""
    found_resources: list[dict[str, str]] = []
    resource_lookup_status: CrisisResourceLookupStatus = "not_attempted"

    latest_lookup = context.latest_crisis_resource_tool_result()
    if latest_lookup is not None:
        found_resources = [dict(row) for row in latest_lookup.found_resources]
        if not inferred_location:
            inferred_location = latest_lookup.inferred_location
        resource_lookup_status = latest_lookup.resource_lookup_status

    result = await execute_crisis_support_template_tool(
        risk_level=str(arguments.get("risk_level") or "high"),
        inferred_location=inferred_location,
        found_resources=found_resources,
        resource_lookup_status=resource_lookup_status,
    )
    return dict(result.model_dump(mode="json"))


async def _execute_recall_saved_memory(
    context: Any,
    arguments: dict[str, object],
) -> dict[str, object]:
    """Run the recall_saved_memory tool with server-side gating.

    Gating order:
    1. Empty query -> refuse without touching the store.
    2. Procedural profile fetch (cheap document read) -> if the user has
       proactive recall disabled, refuse before any semantic retrieval.
       A model tool call is not the same as explicit user consent, so a
       recall-off setting always takes precedence.
    3. Otherwise call ``load_memory_for_turn`` and project results.

    Incognito refusal is handled upstream by ``_PERSISTENT_ONLY_TOOL_NAMES``.
    """

    raw_query = arguments.get("query")
    query = raw_query.strip() if isinstance(raw_query, str) else ""
    if not query:
        return {
            "response_text": (
                "No recall query was provided. Try again with a topic to "
                "search saved memory for."
            ),
            "results": [],
            "refused": True,
            "side_effect": "none",
            "retry_safe": True,
        }

    raw_limit = arguments.get("limit")
    try:
        limit = int(raw_limit) if raw_limit is not None else _RECALL_DEFAULT_LIMIT
    except (TypeError, ValueError):
        limit = _RECALL_DEFAULT_LIMIT
    limit = max(1, min(limit, _RECALL_MAX_LIMIT))

    workflow_context = context.workflow_context
    owner_id = resolve_owner_id(context.agent_state)

    profile = await aget_procedural_profile(
        workflow_context.memory_store, user_id=owner_id
    )
    if not profile.proactive_recall_enabled:
        return {
            "response_text": (
                "Saved memory exists, but proactive recall is off for this "
                "user. Honor that setting and continue without quoting saved "
                "facts; suggest turning recall on only if the user asks."
            ),
            "results": [],
            "refused": True,
            "reason": "proactive_recall_disabled",
            "side_effect": "none",
            "retry_safe": True,
        }

    result = await load_memory_for_turn(
        memory_store=workflow_context.memory_store,
        embedding_provider=workflow_context.embedding_provider,
        owner_id=owner_id,
        query=query,
        is_first_turn=False,
    )

    entries = [
        _recall_entry_payload(entry) for entry in list(result.working_memory)[:limit]
    ]
    entries = [entry for entry in entries if entry]

    return {
        "response_text": (
            "Recalled memory entries follow. Use them only when relevant to "
            "the current turn and avoid reciting them verbatim."
        ),
        "query": query,
        "results": entries,
        "result_count": len(entries),
        "side_effect": "none",
        "retry_safe": True,
    }


def _recall_entry_payload(entry: Any) -> dict[str, object] | None:
    """Project a WorkingMemoryEntry into a compact tool-result shape."""

    if entry is None:
        return None
    if isinstance(entry, dict):
        snippet = (
            entry.get("evidence_quote")
            or entry.get("summary")
            or entry.get("text")
            or ""
        )
        return {
            "snippet": str(snippet).strip(),
            "kind": str(entry.get("kind") or entry.get("source") or "memory"),
        }
    snippet = (
        getattr(entry, "evidence_quote", None)
        or getattr(entry, "summary", None)
        or getattr(entry, "text", None)
        or ""
    )
    return {
        "snippet": str(snippet).strip(),
        "kind": str(
            getattr(entry, "kind", None) or getattr(entry, "source", None) or "memory"
        ),
    }
