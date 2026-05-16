"""Dormant OpenAI Agents SDK foundations for the text runtime migration."""

from agent.text_runtime.openai_agents.agents import (
    CRISIS_AGENT_NAME,
    GUIDED_EXERCISE_AGENT_NAME,
    THERAPEUTIC_AGENT_NAME,
    OpenAITextAgentRoster,
    build_crisis_response_agent,
    build_guided_exercise_agent,
    build_openai_text_agent_roster,
    build_therapeutic_agent,
)
from agent.text_runtime.openai_agents.context import (
    MemoryReadActionType,
    MemoryToolCallRecord,
    OpenAITextRunContext,
)
from agent.text_runtime.openai_agents.memory_tools import (
    MemoryReadToolResult,
    build_read_only_memory_tools,
    execute_read_only_memory_action,
    show_memory_status,
    show_saved_memory,
)

__all__ = [
    "CRISIS_AGENT_NAME",
    "GUIDED_EXERCISE_AGENT_NAME",
    "THERAPEUTIC_AGENT_NAME",
    "MemoryReadToolResult",
    "MemoryReadActionType",
    "MemoryToolCallRecord",
    "OpenAITextAgentRoster",
    "OpenAITextRunContext",
    "build_crisis_response_agent",
    "build_guided_exercise_agent",
    "build_openai_text_agent_roster",
    "build_read_only_memory_tools",
    "build_therapeutic_agent",
    "execute_read_only_memory_action",
    "show_memory_status",
    "show_saved_memory",
]
