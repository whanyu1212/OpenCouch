"""Framework-neutral text-agent runtime adapters."""

from agent.text_runtime.factory import (
    DEFAULT_TEXT_AGENT_RUNTIME,
    create_text_agent_adapter,
    resolve_text_agent_runtime,
)
from agent.text_runtime.langgraph_adapter import (
    AgentWorkflow,
    LangGraphTextAgentAdapter,
)
from agent.text_runtime.openai_adapter import OpenAITextAgentAdapter
from agent.text_runtime.types import (
    TextAgentAdapter,
    TextAgentRuntimeName,
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeShadowResult,
    TextRuntimeShadowStatus,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)

__all__ = [
    "DEFAULT_TEXT_AGENT_RUNTIME",
    "AgentWorkflow",
    "LangGraphTextAgentAdapter",
    "OpenAITextAgentAdapter",
    "TextAgentAdapter",
    "TextAgentRuntimeName",
    "TextRuntimeChunkEvent",
    "TextRuntimeStateEvent",
    "TextRuntimeShadowResult",
    "TextRuntimeShadowStatus",
    "TextRuntimeStatusEvent",
    "TextRuntimeStreamEvent",
    "create_text_agent_adapter",
    "resolve_text_agent_runtime",
]
