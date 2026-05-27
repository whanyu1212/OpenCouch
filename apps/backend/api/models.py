"""Pydantic request/response schemas for the HTTP API.

These models are the API's public contract: they define what callers
send and receive. They intentionally do not re-export the internal
``AgentState``, ``AgentInput``, or ``AgentOutput`` types because those
carry implementation details that should not leak to HTTP callers.

The route handlers own the mapping between these public schemas and the
internal agent models, keeping the API surface stable as runtime state
evolves.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from agent.feedback.models import FeedbackLabel
from config import ResponseModelTier


# Request models


class ApiMemoryMode(StrEnum):
    """API-facing memory mode selector."""

    PERSISTENT = "persistent"
    INCOGNITO = "incognito"


class ChatRequest(BaseModel):
    """POST /api/chat request body."""

    message: str = Field(min_length=1, description="The user's message text.")
    thread_id: str = Field(
        min_length=1,
        description="Thread identifier for conversation continuity. "
        "Reuse the same thread_id to continue a conversation.",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional stable owner identifier for cross-thread "
        "memory. When set, memory is namespaced by user_id rather "
        "than thread_id.",
    )
    memory_mode: ApiMemoryMode | None = Field(
        default=None,
        description=(
            "Optional memory mode for this chat session. When omitted, "
            "the API default from OPENCOUCH_MEMORY_MODE is used."
        ),
    )
    response_model_tier: ResponseModelTier | None = Field(
        default="fast",
        description=(
            "Optional text-response tier. 'fast' favors lower latency; "
            "'quality' favors richer prose. Safety, routing, memory, "
            "and summarization stay pinned to the control model."
        ),
    )


class EndSessionRequest(BaseModel):
    """POST /api/threads/{thread_id}/end request body.

    All fields are optional so clients can post an empty body while the
    endpoint remains extensible for future per-request hints.
    """

    feedback: FeedbackLabel | None = Field(
        default=None,
        description=(
            "Optional end-of-session rating: 'positive', 'negative', "
            "or 'skip'. When set, written to the session_feedback "
            "store before summarization runs. When null or omitted, "
            "no feedback record is created and summarization proceeds "
            "as usual."
        ),
    )


class VoiceRealtimeSessionRequest(BaseModel):
    """POST /api/voice/realtime/session request body."""

    thread_id: str = Field(min_length=1)
    user_id: str | None = None
    memory_mode: Literal["incognito", "persistent"] = "incognito"
    assistant_voice: (
        Literal[
            "alloy",
            "ash",
            "ballad",
            "cedar",
            "coral",
            "echo",
            "marin",
            "sage",
            "shimmer",
            "verse",
        ]
        | None
    ) = None


class VoiceToolCallRequest(BaseModel):
    """POST /api/voice/realtime/tools request body."""

    thread_id: str = Field(min_length=1)
    user_id: str | None = None
    current_user_message: str = ""
    transcript: list[dict[str, object]] = Field(default_factory=list)
    memory_mode: Literal["incognito", "persistent"] = "incognito"
    tool_name: str = Field(min_length=1)
    arguments: dict[str, object] = Field(default_factory=dict)


class VoiceRecordedToolCall(BaseModel):
    """One Realtime tool call observed during a voice turn."""

    tool_name: str = Field(min_length=1)
    status: Literal["started", "completed", "failed"]
    output: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


class VoiceTurnRecordRequest(BaseModel):
    """POST /api/voice/realtime/turn request body."""

    thread_id: str = Field(min_length=1)
    user_id: str | None = None
    user_text: str = ""
    assistant_text: str = ""
    memory_mode: Literal["incognito", "persistent"] = "incognito"
    route: str | None = None
    response_style: str | None = None
    tool_calls: list[VoiceRecordedToolCall] = Field(default_factory=list)


class VoiceTurnPolicyRequest(BaseModel):
    """POST /api/voice/realtime/turn-policy request body."""

    thread_id: str = Field(min_length=1)
    user_id: str | None = None
    user_text: str = Field(min_length=1)
    memory_mode: Literal["incognito", "persistent"] = "incognito"


class VoiceEndSessionRequest(BaseModel):
    """POST /api/voice/realtime/end request body."""

    thread_id: str = Field(min_length=1)
    memory_mode: Literal["incognito", "persistent"] = "incognito"


# Response models


class CrisisInfo(BaseModel):
    """Crisis assessment included in every chat response."""

    level: int = Field(description="Crisis severity: 0 (none) to 3 (imminent).")
    confidence: str = Field(description="Classifier confidence: low, medium, high.")
    reason: str = Field(description="One-line explanation of the classification.")
    needs_crisis_response: bool
    needs_clarification: bool


class ChatResponse(BaseModel):
    """POST /api/chat response body."""

    response_text: str = Field(description="The agent's reply text.")
    response_type: str = Field(description="'therapeutic' or 'crisis'.")
    response_style: str | None = Field(
        default=None,
        description="Which response style shaped the reply "
        "(supportive, reflective, technique, clarifying, psychoeducation, "
        "guided_exercise, closing, or None for crisis path).",
    )
    therapeutic_approach: str | None = Field(
        default=None,
        description="Which therapeutic approach informed the reply "
        "(motivational_interviewing, cbt, act, dbt_skills, "
        "grief_support, interpersonal_therapy, pfa, or none).",
    )
    session_action: Literal["none", "suggest_end_session"] = Field(
        default="none",
        description="Optional session-level UI hint. "
        "'suggest_end_session' means the assistant has produced a closing "
        "reply and the client may offer explicit session finalization.",
    )
    crisis: CrisisInfo
    diagnostics: dict[str, object] = Field(
        default_factory=dict,
        description="Per-turn timing and decision metadata from the graph nodes.",
    )


class ThreadSummaryResponse(BaseModel):
    """One entry in GET /api/threads list."""

    thread_id: str
    turn_count: int
    message_count: int
    has_context: bool


class ThreadSessionStatusResponse(BaseModel):
    """GET /api/threads/{id}/session-status response."""

    has_active_session: bool


class MessageResponse(BaseModel):
    """One transcript entry in GET /api/threads/{id}/history."""

    role: str = Field(description="'user' or 'assistant'.")
    content: str
    response_style: str | None = Field(
        default=None,
        description="Response style for assistant turns, None for user turns.",
    )


class SessionArcResponse(BaseModel):
    """Response from POST /api/threads/{id}/end when a summary is produced."""

    summary: str
    themes: list[str]
    mood_opened: str
    mood_closed: str
    turn_count: int
    open_loops: list[str]
    resolved_threads: list[str]


class VoiceRealtimeSessionResponse(BaseModel):
    """POST /api/voice/realtime/session response body."""

    client_secret: str
    thread_id: str
    user_id: str | None = None
    memory_mode: Literal["incognito", "persistent"]
    session_config: dict[str, object]


class VoiceToolCallResponse(BaseModel):
    """POST /api/voice/realtime/tools response body."""

    output: dict[str, object]


class VoiceTurnRecordResponse(BaseModel):
    """POST /api/voice/realtime/turn response body."""

    recorded: bool
    thread_id: str
    message_count: int


class VoiceTurnPolicyResponse(BaseModel):
    """POST /api/voice/realtime/turn-policy response body."""

    route: str
    response_style: str
    required_tool_name: str | None = None
    required_tool_arguments: dict[str, object] = Field(default_factory=dict)
    instructions: str


class VoiceEndSessionResponse(BaseModel):
    """POST /api/voice/realtime/end response body."""

    finalized: bool
    summary: str | None = None
    detail: str


class MemoryStatusResponse(BaseModel):
    """GET /api/memory/status response."""

    memory_mode: str
    owner_id: str
    counts: dict[str, int] = Field(
        description="User-visible memory counts keyed by kind: "
        "active semantic facts, episodic session arcs, and "
        "active procedural rules.",
    )
    crisis_log_count: int
    session_feedback_count: int = Field(
        default=0,
        description=(
            "Total number of session-feedback records across all "
            "sessions. Surfaced for observability dashboards and the "
            "CLI's /memory status panel."
        ),
    )
    proactive_recall_enabled: bool


class MemoryRecallUpdateRequest(BaseModel):
    """PATCH /api/memory/recall request."""

    enabled: bool


class MemoryRecallUpdateResponse(BaseModel):
    """PATCH /api/memory/recall response."""

    owner_id: str
    proactive_recall_enabled: bool
    detail: str


class DeleteResponse(BaseModel):
    """Response for DELETE endpoints."""

    deleted: bool
    detail: str


# WebSocket message models


class StreamStatusMessage(BaseModel):
    """WebSocket message: node progress update."""

    type: str = "status"
    stage: str
    detail: str = ""


class StreamChunkMessage(BaseModel):
    """WebSocket message: incremental response text."""

    type: str = "chunk"
    text: str


class StreamDoneMessage(BaseModel):
    """WebSocket message: terminal event with full response."""

    type: str = "done"
    response: ChatResponse


class StreamErrorMessage(BaseModel):
    """WebSocket message: terminal error event."""

    type: str = "error"
    code: str
    message: str
