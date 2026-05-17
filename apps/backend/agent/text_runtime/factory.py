"""Factory helpers for the OpenAI text-agent runtime adapter."""

from __future__ import annotations

import os

from agent.text_runtime.openai_adapter import OpenAITextAgentAdapter
from agent.text_runtime.types import TextAgentAdapter, TextAgentRuntimeName

DEFAULT_TEXT_AGENT_RUNTIME: TextAgentRuntimeName = "openai"
TEXT_AGENT_RUNTIME_ENV = "OPENCOUCH_TEXT_AGENT_RUNTIME"


def resolve_text_agent_runtime(value: str | None = None) -> TextAgentRuntimeName:
    """Resolve the configured text-agent runtime name.

    The migration is complete for text: OpenAI Agents SDK is the only supported
    runtime. The environment hook remains as a compatibility check so stale
    deployments fail loudly when they still request ``langgraph``.
    """

    raw = value if value is not None else os.getenv(TEXT_AGENT_RUNTIME_ENV)
    runtime = (raw or DEFAULT_TEXT_AGENT_RUNTIME).strip().lower()
    if runtime == "openai":
        return "openai"
    raise ValueError(
        f"Unsupported {TEXT_AGENT_RUNTIME_ENV}={runtime!r}. Supported value: openai."
    )


def create_text_agent_adapter(
    *,
    runtime_name: TextAgentRuntimeName = DEFAULT_TEXT_AGENT_RUNTIME,
) -> TextAgentAdapter:
    """Create the text-agent adapter for the configured runtime."""

    if runtime_name == "openai":
        return OpenAITextAgentAdapter()
    raise ValueError(f"Unsupported text-agent runtime {runtime_name!r}.")
