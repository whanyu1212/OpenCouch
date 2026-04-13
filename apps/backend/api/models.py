"""Pydantic request/response schemas for the HTTP API.

These models are the API's public contract — they define what
callers send and receive. They intentionally do NOT re-export the
internal ``AgentState``, ``AgentInput``, or ``AgentOutput`` types
directly, because those carry implementation detail (field names,
optional fields, internal enums) that shouldn't leak to HTTP
callers.

Instead, the API models are thin wrappers that map to/from the
internal types in the route handlers. This keeps the API surface
stable even if the internal state schema evolves.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────────


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


class EndSessionRequest(BaseModel):
    """POST /api/threads/{thread_id}/end request body."""

    # No required fields — thread_id comes from the path parameter.
    # The body exists so we can add optional fields later (e.g.,
    # a custom farewell message) without changing the endpoint shape.
    pass


# ── Response models ─────────────────────────────────────────────────


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
    mode: str | None = Field(
        default=None,
        description="Which therapeutic mode shaped the reply "
        "(supportive, reflective, clarifying, psychoeducation, "
        "guided_exercise, closing, or None for crisis path).",
    )
    mode_source: str | None = Field(
        default=None,
        description="How the mode was selected: keyword, llm, default.",
    )
    modality: str | None = Field(
        default=None,
        description="Which therapeutic modality informed the reply "
        "(motivational_interviewing, cbt, act, dbt_skills, "
        "grief_support, interpersonal_therapy, pfa, or none).",
    )
    crisis: CrisisInfo
    diagnostics: dict = Field(
        default_factory=dict,
        description="Per-turn timing and decision metadata from the graph nodes.",
    )


class ThreadSummaryResponse(BaseModel):
    """One entry in GET /api/threads list."""

    thread_id: str
    turn_count: int
    message_count: int
    has_context: bool


class MessageResponse(BaseModel):
    """One transcript entry in GET /api/threads/{id}/history."""

    role: str = Field(description="'user' or 'assistant'.")
    content: str
    mode: str | None = Field(
        default=None,
        description="Routing mode for assistant turns, None for user turns.",
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
        description="Record counts keyed by namespace kind: "
        "semantic, episodic, procedural.",
    )
    crisis_log_count: int
    proactive_recall_enabled: bool


class DeleteResponse(BaseModel):
    """Response for DELETE endpoints."""

    deleted: bool
    detail: str


# ─��� WebSocket message models ────────────────────────────────────────


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
