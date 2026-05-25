"""Public data models and stream event contracts for the agent runtime."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.memory.entries import WorkingMemoryEntry


class Channel(str, Enum):
    """Surface where the user message originated."""

    TEST = "test"
    WEB = "web"
    VOICE = "voice"


class MessageRole(str, Enum):
    """Conversation role used in model-visible history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ResponseCategory(str, Enum):
    """Public response category returned to callers."""

    THERAPEUTIC = "therapeutic"
    CRISIS = "crisis"


SessionAction = Literal["none", "suggest_end_session"]


class Message(BaseModel):
    """Validated, serializable conversation turn."""

    role: MessageRole
    content: str = Field(min_length=1)
    # Assistant turns may include the response style that shaped the reply.
    # User turns leave this unset.
    response_style: str | None = None


class CrisisAssessment(BaseModel):
    """Crisis classifier result carried separately from response text."""

    level: Literal[0, 1, 2, 3] = 0
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""
    needs_crisis_response: bool = False
    needs_clarification: bool = False


class AgentInput(BaseModel):
    """External input contract for one agent turn."""

    message: str = Field(min_length=1)
    channel: Channel = Channel.TEST
    user_id: str | None = None
    session_id: str | None = None
    history: list[Message] = Field(default_factory=list)
    working_memory: list[WorkingMemoryEntry] = Field(default_factory=list)
    installed_skills: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    """Normalized output contract shared by API, CLI, and tests."""

    response_text: str
    response_type: ResponseCategory
    crisis: CrisisAssessment
    response_style: str | None = None
    therapeutic_approach: str | None = None
    session_action: SessionAction = "none"
    should_persist_memory: bool = False
    # Per-turn diagnostics for CLI/API observability. Nodes write timings
    # and write-counts into ``state["diagnostics"]``; ``state_to_output``
    # copies them through without imposing a stable sub-schema.
    diagnostics: dict[str, Any] = Field(default_factory=dict)


# ── Stream event types ────────────────────────────────────────────────────────


class StatusEvent(BaseModel):
    """Pipeline progress update emitted before response generation begins."""

    type: Literal["status"] = "status"
    stage: str
    detail: str = ""


# Human-readable labels for pipeline stages. Used by both the CLI and
# WebSocket API so that all clients display consistent, friendly text.
STAGE_LABELS: dict[str, str] = {
    "load_memory": "loading memory",
    "memory_profile_load": "loading profile memory",
    "memory_graph_load": "querying graph memory",
    "memory_profile_save": "saving profile memory",
    "memory_graph_save": "writing graph memory",
    "crisis_gate": "safety check",
    "memory_control": "updating memory",
    "grounded_lookup": "looking up factual answer",
    "crisis_resource_lookup": "looking up crisis resources",
    "crisis_response": "generating crisis reply",
    "crisis_clarification": "checking immediate safety",
    "crisis_log": "writing crisis log",
    "therapeutic": "generating therapeutic reply",
    "finalize": "finalizing turn",
    "session_stage": "reading context",
    "response_generation": "generating",
}


def friendly_stage(stage: str) -> str:
    """Return the human-friendly label for a pipeline stage.

    Args:
        stage: Internal pipeline stage identifier.

    Returns:
        Friendly label for display surfaces, or ``stage`` if unknown.
    """

    return STAGE_LABELS.get(stage, stage)


class ChunkEvent(BaseModel):
    """Incremental text chunk from response generation.

    Text runtimes emit these while streaming response text.
    """

    type: Literal["chunk"] = "chunk"
    text: str


class ResponseReadyEvent(BaseModel):
    """Non-terminal event emitted when the reply is finalized.

    This fires after turn finalization has appended the assistant
    reply to transcript. The output payload is intentionally partial:
    response text, routing, and crisis metadata are ready; tail diagnostics
    like ``turn_total_ms`` still land on the terminal ``DoneEvent``.
    """

    type: Literal["response_ready"] = "response_ready"
    output: AgentOutput


class DoneEvent(BaseModel):
    """Terminal event carrying the complete agent output."""

    type: Literal["done"] = "done"
    output: AgentOutput


StreamEvent = StatusEvent | ChunkEvent | ResponseReadyEvent | DoneEvent
