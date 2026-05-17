"""Factory helpers for text-agent runtime adapters."""

from __future__ import annotations

import os
from typing import Any

from agent.text_runtime.langgraph_adapter import (
    AgentWorkflowBuilder,
    LangGraphTextAgentAdapter,
)
from agent.text_runtime.openai_adapter import OpenAITextAgentAdapter
from agent.text_runtime.types import (
    TextAgentAdapter,
    TextAgentRuntimeName,
)

DEFAULT_TEXT_AGENT_RUNTIME: TextAgentRuntimeName = "openai"
TEXT_AGENT_RUNTIME_ENV = "OPENCOUCH_TEXT_AGENT_RUNTIME"


def resolve_text_agent_runtime(value: str | None = None) -> TextAgentRuntimeName:
    """Resolve the configured text-agent runtime name."""

    raw = value if value is not None else os.getenv(TEXT_AGENT_RUNTIME_ENV)
    runtime = (raw or DEFAULT_TEXT_AGENT_RUNTIME).strip().lower()
    if runtime == "langgraph":
        return "langgraph"
    if runtime == "openai":
        return "openai"
    raise ValueError(
        f"Unsupported {TEXT_AGENT_RUNTIME_ENV}={runtime!r}. "
        "Supported values: langgraph, openai."
    )


def create_text_agent_adapter(
    *,
    checkpointer: Any,
    graph_builder: AgentWorkflowBuilder,
    runtime_name: TextAgentRuntimeName = DEFAULT_TEXT_AGENT_RUNTIME,
) -> TextAgentAdapter:
    """Create the text-agent adapter for the configured runtime."""

    if runtime_name == "langgraph":
        return LangGraphTextAgentAdapter(graph_builder(checkpointer=checkpointer))
    if runtime_name == "openai":
        return OpenAITextAgentAdapter(
            checkpoint_adapter=LangGraphTextAgentAdapter(
                graph_builder(checkpointer=checkpointer)
            )
        )
    raise ValueError(f"Unsupported text-agent runtime {runtime_name!r}.")
