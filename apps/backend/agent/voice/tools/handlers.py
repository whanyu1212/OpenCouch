"""App-owned voice tool handler implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from agent.memory.operations.procedural_profile import aget_procedural_profile
from agent.memory.retrieval.service import load_memory_for_turn
from agent.state import resolve_owner_id
from agent.tools.crisis import (
    execute_crisis_resource_lookup_tool,
    execute_crisis_support_template_tool,
)
from agent.tools.grounded import execute_grounded_lookup_tool
from agent.tools.guided_exercise import (
    GuidedExerciseProgressOutcome,
    execute_guided_exercise_discovery_tool,
    execute_guided_exercise_progress_tool,
    execute_guided_exercise_skill_tool,
)
from agent.tools.memory import (
    execute_memory_tool_action,
    execute_read_only_memory_action,
)
from agent.tools.therapeutic import execute_therapeutic_response_skill_tool
from agent.voice.tools.context import VoiceToolDispatchContext

if TYPE_CHECKING:
    from agent.audit.models import CrisisResourceLookupStatus

_RECALL_DEFAULT_LIMIT = 5
_RECALL_MAX_LIMIT = 10


async def _handle_wait_for_user(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> dict[str, object]:
    del dispatch_context, arguments
    return {
        "response_text": "",
        "should_respond": False,
        "side_effect": "none",
    }


async def _handle_show_memory_status(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    del arguments
    return await execute_read_only_memory_action(
        dispatch_context.tool_context,
        {"type": "status"},
    )


async def _handle_show_saved_memory(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    del arguments
    return await execute_read_only_memory_action(
        dispatch_context.tool_context,
        {"type": "list"},
    )


async def _handle_recall_saved_memory(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> dict[str, object]:
    return await _execute_recall_saved_memory(dispatch_context.tool_context, arguments)


async def _handle_save_response_preference(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_memory_tool_action(
        dispatch_context.tool_context,
        {
            "type": "save_preference",
            "preference_text": str(arguments.get("preference_text") or ""),
        },
        side_effect="procedural_profile_update",
        retry_safe=False,
    )


async def _handle_set_proactive_memory_recall(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_memory_tool_action(
        dispatch_context.tool_context,
        {
            "type": "set_recall",
            "enabled": bool(arguments.get("enabled")),
        },
        side_effect="procedural_profile_update",
        retry_safe=True,
    )


async def _handle_prepare_memory_deletion_by_index(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_memory_tool_action(
        dispatch_context.tool_context,
        {
            "type": "forget_by_index",
            "target_kind": str(arguments.get("target_kind") or ""),
            "target_index": int(arguments.get("target_index") or 0),
        },
        side_effect="pending_deletion",
        retry_safe=True,
    )


async def _handle_prepare_memory_deletion_by_query(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_memory_tool_action(
        dispatch_context.tool_context,
        {
            "type": "forget_by_query",
            "query": str(arguments.get("query") or ""),
        },
        side_effect="pending_deletion",
        retry_safe=True,
    )


async def _handle_confirm_memory_deletion(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    del arguments
    return await execute_memory_tool_action(
        dispatch_context.tool_context,
        {"type": "confirm_pending"},
        side_effect="delete_memory",
        retry_safe=False,
    )


async def _handle_cancel_memory_deletion(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    del arguments
    return await execute_memory_tool_action(
        dispatch_context.tool_context,
        {"type": "cancel_pending"},
        side_effect="cancel_pending",
        retry_safe=True,
    )


async def _handle_answer_grounded_lookup(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_grounded_lookup_tool(
        dispatch_context.tool_context,
        query=str(arguments.get("query") or ""),
    )


async def _handle_lookup_crisis_resources(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    del arguments
    result = await execute_crisis_resource_lookup_tool(dispatch_context.tool_context)
    await dispatch_context.runtime.voice.persist_voice_crisis_resource_lookup(
        thread_id=dispatch_context.thread_id,
        user_id=dispatch_context.user_id,
        client_turn_id=dispatch_context.client_turn_id,
        inferred_location=result.inferred_location,
        found_resources=result.found_resources,
        resource_lookup_status=result.resource_lookup_status,
    )
    return result


async def _handle_get_crisis_support_template(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> dict[str, object]:
    return await _execute_crisis_support_template(
        dispatch_context.tool_context, arguments
    )


async def _handle_list_guided_exercise_skills(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_guided_exercise_discovery_tool(
        dispatch_context.tool_context,
        therapeutic_approach=_optional_string(arguments.get("therapeutic_approach")),
        channel=_optional_string(arguments.get("channel")),
    )


async def _handle_load_therapeutic_response_skill(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_therapeutic_response_skill_tool(
        dispatch_context.tool_context,
        response_style=str(arguments.get("response_style") or "supportive"),
        therapeutic_approach=_optional_string(arguments.get("therapeutic_approach")),
    )


async def _handle_load_guided_exercise_skill(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_guided_exercise_skill_tool(
        dispatch_context.tool_context,
        exercise_type=str(arguments.get("exercise_type") or ""),
        runtime_action=str(arguments.get("runtime_action") or ""),
        current_step_index=_optional_int(arguments.get("current_step_index")),
    )


async def _handle_record_guided_exercise_progress(
    dispatch_context: VoiceToolDispatchContext,
    arguments: dict[str, object],
) -> object:
    return await execute_guided_exercise_progress_tool(
        dispatch_context.tool_context,
        expected_skill_id=str(arguments.get("expected_skill_id") or ""),
        expected_step_id=str(arguments.get("expected_step_id") or ""),
        outcome=cast(
            GuidedExerciseProgressOutcome,
            str(arguments.get("outcome") or "hold"),
        ),
        user_response_summary=str(arguments.get("user_response_summary") or ""),
    )


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
