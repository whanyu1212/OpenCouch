"""Pydantic request/response schemas for the HTTP API.

These models are the API's public contract: they define what callers
send and receive. They intentionally do not re-export the internal
``AgentState``, ``AgentInput``, or ``AgentOutput`` types because those
carry implementation details that should not leak to HTTP callers.

The route handlers own the mapping between these public schemas and the
internal agent models, keeping the API surface stable as the graph state
evolves.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.memory.models import FeedbackLabel
from core.config import ResponseModelTier


# Request models


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
    response_style_source: str | None = Field(
        default=None,
        description="How the response style was selected: keyword, llm, default.",
    )
    therapeutic_approach: str | None = Field(
        default=None,
        description="Which therapeutic approach informed the reply "
        "(motivational_interviewing, cbt, act, dbt_skills, "
        "grief_support, interpersonal_therapy, pfa, or none).",
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
