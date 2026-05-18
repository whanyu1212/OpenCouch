"""OpenAI Agents SDK memory tools for the text runtime migration."""

from __future__ import annotations

from typing import Any, Literal, Mapping, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.gates.memory_control.service import (
    MemoryControlRequest,
    execute_memory_control_request,
)
from agent.runtime.context import (
    MemoryActionType,
    MemoryReadActionType,
    MemoryToolSideEffect,
    OpenAITextRunContext,
)


class MemoryToolResult(BaseModel):
    """Structured result returned by memory-control tools."""

    response_text: str = Field(description="User-facing memory-control text.")
    memory_control: dict[str, Any] = Field(
        default_factory=dict,
        description="Memory-control state delta produced by the shared service.",
    )
    procedural_profile: dict[str, Any] | None = Field(
        default=None,
        description="Procedural profile delta when a memory preference changed.",
    )
    side_effect: MemoryToolSideEffect = Field(
        default="none",
        description="Side-effect category for this memory tool call.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the tool can duplicate or repeat side effects.",
    )


MemoryReadToolResult = MemoryToolResult


def memory_control_request_from_context(
    context: OpenAITextRunContext,
    action: Mapping[str, Any],
) -> MemoryControlRequest:
    """Build a neutral memory-control request from SDK local context."""

    owner_id = context.user_id or context.session_id or context.thread_id
    return MemoryControlRequest(
        owner_id=owner_id,
        current_user_message=context.current_user_message,
        action=dict(action),
        pending_action=(
            dict(context.pending_memory_action)
            if context.pending_memory_action is not None
            else None
        ),
        session_id=context.session_id or context.thread_id,
        turn_count=context.turn_count,
    )


async def execute_memory_tool_action(
    context: OpenAITextRunContext,
    action: dict[str, Any],
    *,
    side_effect: MemoryToolSideEffect,
    retry_safe: bool,
) -> MemoryToolResult:
    """Execute a memory action through the existing service layer."""

    action_type = action.get("type")
    if action_type not in {
        "list",
        "status",
        "set_recall",
        "save_preference",
        "forget_by_index",
        "forget_by_query",
        "confirm_pending",
        "cancel_pending",
    }:
        raise ValueError(f"Unsupported memory action: {action_type!r}")

    result = await execute_memory_control_request(
        memory_control_request_from_context(context, action),
        context.workflow_context,
    )
    tool_result = MemoryToolResult(
        response_text=result.response_text,
        memory_control=result.memory_control,
        procedural_profile=result.procedural_profile,
        side_effect=side_effect,
        retry_safe=retry_safe,
    )
    context.record_memory_tool_result(
        action_type=cast(MemoryActionType, action_type),
        response_text=tool_result.response_text,
        memory_control=tool_result.memory_control,
        procedural_profile=tool_result.procedural_profile,
        side_effect=tool_result.side_effect,
        retry_safe=tool_result.retry_safe,
    )
    return tool_result


async def execute_read_only_memory_action(
    context: OpenAITextRunContext,
    action: dict[str, Any],
) -> MemoryToolResult:
    """Execute a read-only memory action through the existing service layer."""

    action_type = action.get("type")
    if action_type not in {"list", "status"}:
        raise ValueError(f"Unsupported read-only memory action: {action_type!r}")
    return await execute_memory_tool_action(
        context,
        action,
        side_effect="none",
        retry_safe=True,
    )


async def _execute_memory_tool(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    action: dict[str, Any],
    *,
    side_effect: MemoryToolSideEffect,
    retry_safe: bool,
) -> MemoryToolResult:
    """Execute one memory tool action from SDK local context."""

    return await execute_memory_tool_action(
        wrapper.context,
        action,
        side_effect=side_effect,
        retry_safe=retry_safe,
    )


async def _execute_read_memory_tool(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    action_type: MemoryReadActionType,
) -> MemoryToolResult:
    """Execute one read-only memory tool action from SDK local context."""

    return await _execute_memory_tool(
        wrapper,
        {"type": action_type},
        side_effect="none",
        retry_safe=True,
    )


@function_tool(
    name_override="show_saved_memory",
    description_override=(
        "Show a concise overview of saved facts, session summaries, and style "
        "preferences for the current OpenCouch user. Use only when the user "
        "explicitly asks what is saved or remembered. Side effects: none. "
        "Retry safety: safe."
    ),
)
async def show_saved_memory(
    wrapper: RunContextWrapper[OpenAITextRunContext],
) -> MemoryToolResult:
    """Show the current user's saved memory without changing it."""

    return await _execute_read_memory_tool(wrapper, "list")


@function_tool(
    name_override="show_memory_status",
    description_override=(
        "Show counts for saved facts, session summaries, style preferences, "
        "and proactive recall for the current OpenCouch user. Use only when "
        "the user asks about memory status, whether memory is enabled, or how "
        "much memory exists. Side effects: none. Retry safety: safe."
    ),
)
async def show_memory_status(
    wrapper: RunContextWrapper[OpenAITextRunContext],
) -> MemoryToolResult:
    """Show memory status without changing durable memory."""

    return await _execute_read_memory_tool(wrapper, "status")


@function_tool(
    name_override="set_proactive_memory_recall",
    description_override=(
        "Turn proactive memory recall on or off for the current OpenCouch user. "
        "Use only when the user explicitly asks to enable or disable proactive "
        "recall. Side effects: updates procedural memory settings. Retry "
        "safety: safe when retried with the same enabled value."
    ),
)
async def set_proactive_memory_recall(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    enabled: bool,
) -> MemoryToolResult:
    """Set proactive memory recall for the current user."""

    return await _execute_memory_tool(
        wrapper,
        {"type": "set_recall", "enabled": enabled},
        side_effect="procedural_profile_update",
        retry_safe=True,
    )


@function_tool(
    name_override="save_response_preference",
    description_override=(
        "Save an explicit response-style or memory-use preference for the "
        "current OpenCouch user. Use only for explicit preferences about how "
        "the assistant should respond or use memory, not for personal facts or "
        "coping plans. Side effects: writes durable procedural memory. Retry "
        "safety: not guaranteed."
    ),
)
async def save_response_preference(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    preference_text: str,
) -> MemoryToolResult:
    """Save one explicit assistant response preference."""

    return await _execute_memory_tool(
        wrapper,
        {"type": "save_preference", "preference_text": preference_text},
        side_effect="procedural_profile_update",
        retry_safe=False,
    )


@function_tool(
    name_override="prepare_memory_deletion_by_index",
    description_override=(
        "Prepare deletion of a saved memory selected by visible kind and "
        "one-based index. This only creates a pending deletion and asks for "
        "confirmation; it must not delete by itself. Side effects: pending "
        "deletion state only. Retry safety: safe."
    ),
)
async def prepare_memory_deletion_by_index(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    target_kind: Literal["fact", "session", "rule"],
    target_index: int,
) -> MemoryToolResult:
    """Prepare deletion of a saved memory selected by kind/index."""

    return await _execute_memory_tool(
        wrapper,
        {
            "type": "forget_by_index",
            "target_kind": target_kind,
            "target_index": target_index,
        },
        side_effect="pending_deletion",
        retry_safe=True,
    )


@function_tool(
    name_override="prepare_memory_deletion_by_query",
    description_override=(
        "Prepare deletion of a saved memory selected by a concrete query. This "
        "only creates a pending deletion or asks the user to disambiguate; it "
        "must not delete by itself. Side effects: pending deletion state only. "
        "Retry safety: safe."
    ),
)
async def prepare_memory_deletion_by_query(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    query: str,
) -> MemoryToolResult:
    """Prepare deletion of a saved memory selected by query."""

    return await _execute_memory_tool(
        wrapper,
        {"type": "forget_by_query", "query": query},
        side_effect="pending_deletion",
        retry_safe=True,
    )


@function_tool(
    name_override="confirm_memory_deletion",
    description_override=(
        "Confirm and perform a pending saved-memory deletion. Use only when "
        "the runtime says a pending deletion exists and the user clearly "
        "confirms it. Side effects: deletes durable memory. Retry safety: not "
        "guaranteed."
    ),
)
async def confirm_memory_deletion(
    wrapper: RunContextWrapper[OpenAITextRunContext],
) -> MemoryToolResult:
    """Confirm a pending saved-memory deletion."""

    return await _execute_memory_tool(
        wrapper,
        {"type": "confirm_pending"},
        side_effect="delete_memory",
        retry_safe=False,
    )


@function_tool(
    name_override="cancel_memory_deletion",
    description_override=(
        "Cancel a pending saved-memory deletion. Use only when a pending "
        "deletion exists and the user cancels or declines. Side effects: clears "
        "pending deletion state. Retry safety: safe."
    ),
)
async def cancel_memory_deletion(
    wrapper: RunContextWrapper[OpenAITextRunContext],
) -> MemoryToolResult:
    """Cancel a pending saved-memory deletion."""

    return await _execute_memory_tool(
        wrapper,
        {"type": "cancel_pending"},
        side_effect="cancel_pending",
        retry_safe=True,
    )


def build_memory_tools() -> list[Any]:
    """Return memory-control tools for the OpenAI text agent."""

    return [
        show_saved_memory,
        show_memory_status,
        set_proactive_memory_recall,
        save_response_preference,
        prepare_memory_deletion_by_index,
        prepare_memory_deletion_by_query,
        confirm_memory_deletion,
        cancel_memory_deletion,
    ]


def build_read_only_memory_tools() -> list[Any]:
    """Return read-only memory tools for tests and compatibility."""

    return [show_saved_memory, show_memory_status]
