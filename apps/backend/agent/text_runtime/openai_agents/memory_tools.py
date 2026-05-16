"""OpenAI Agents SDK memory tools for the text runtime migration."""

from __future__ import annotations

from typing import Any, Literal, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.gates.memory_control.service import execute_memory_control_action
from agent.text_runtime.openai_agents.context import (
    MemoryReadActionType,
    OpenAITextRunContext,
)


class MemoryReadToolResult(BaseModel):
    """Structured result returned by read-only memory tools."""

    response_text: str = Field(description="User-facing memory readout.")
    memory_control: dict[str, Any] = Field(
        default_factory=dict,
        description="Memory-control state delta produced by the shared service.",
    )
    side_effect: Literal["none"] = Field(
        default="none",
        description="Read-only tools must not mutate durable memory.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the tool can duplicate side effects.",
    )


async def execute_read_only_memory_action(
    context: OpenAITextRunContext,
    action: dict[str, Any],
) -> MemoryReadToolResult:
    """Execute a read-only memory action through the existing service layer."""

    action_type = action.get("type")
    if action_type not in {"list", "status"}:
        raise ValueError(f"Unsupported read-only memory action: {action_type!r}")

    result = await execute_memory_control_action(
        context.agent_state_for_memory_action(action),
        context.workflow_context,
    )
    tool_result = MemoryReadToolResult(
        response_text=result.response_text,
        memory_control=result.memory_control,
    )
    context.record_memory_tool_result(
        action_type=cast(MemoryReadActionType, action_type),
        response_text=tool_result.response_text,
        memory_control=tool_result.memory_control,
    )
    return tool_result


async def _execute_memory_tool(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    action_type: MemoryReadActionType,
) -> MemoryReadToolResult:
    """Execute one read-only memory tool action from SDK local context."""

    return await execute_read_only_memory_action(
        wrapper.context,
        {"type": action_type},
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
) -> MemoryReadToolResult:
    """Show the current user's saved memory without changing it."""

    return await _execute_memory_tool(wrapper, "list")


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
) -> MemoryReadToolResult:
    """Show memory status without changing durable memory."""

    return await _execute_memory_tool(wrapper, "status")


def build_read_only_memory_tools() -> list[Any]:
    """Return the initial read-only memory tools for the OpenAI text agent."""

    return [show_saved_memory, show_memory_status]
