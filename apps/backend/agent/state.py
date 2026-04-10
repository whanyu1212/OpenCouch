"""Internal state container for the OpenCouch agent graph."""

from typing import NotRequired, TypedDict

from agent.models import Channel, CrisisAssessment, ModeType, ResponseKind


class SessionMemoryState(TypedDict):
    """Durable session memory that should persist across turns."""

    # Rolling summary used to keep the session coherent as history grows.
    summary: str
    # Current user concerns that should stay salient across turns.
    active_concerns: list[str]
    # Unresolved threads the agent should avoid dropping.
    open_loops: list[str]
    # Best-effort guess at what the user wants from the current session.
    current_goal: str | None


class SessionProgressState(TypedDict):
    """Session-level progression and intent signals."""

    # Soft steering signal for the overall shape of the current session.
    intent: NotRequired[str | None]
    # Tracks whether the current intent came from explicit user language or inference.
    intent_source: NotRequired[str | None]
    # Structured session progression signal for shaping the next response.
    stage: NotRequired[str]
    # Tracks whether the current stage came from deterministic logic or LLM refinement.
    stage_source: NotRequired[str | None]
    # Short rationale for debugging and evals.
    stage_reason: NotRequired[str]
    # Count of user turns including the current inbound message.
    turn_count: int
    # Whether the current runtime is in ephemeral guest mode.
    is_guest: NotRequired[bool]


class RoutingState(TypedDict):
    """Routing and mode selection outputs for the current turn."""

    # Records which high-level path the graph decided to take.
    route: NotRequired[str]
    # Tracks the active non-crisis response mode inside the therapeutic path.
    mode: NotRequired[str]
    # Tracks whether the mode came from keyword, session_intent, llm, or default.
    mode_source: NotRequired[str | None]
    # Distinguishes operational routing states from therapeutic and crisis modes.
    mode_type: NotRequired[ModeType]
    # Active modality overlays selected for the current response node.
    active_modalities: NotRequired[list[str]]
    # Cached semantic interpretation shared by routing and prompt shaping.
    semantic_signals: NotRequired[dict[str, bool]]


class ResponseState(TypedDict):
    """Response shaping and output payload for the current turn."""

    # Turn-specific prompt shaping guidance derived after mode selection.
    guidance: NotRequired[str]
    # Makes the output type explicit before the final response is returned.
    kind: NotRequired[ResponseKind]
    # Holds the generated text from the chosen response node.
    text: NotRequired[str]
    # Signals whether a later wrapper should persist memory after reply completion.
    should_persist_memory: NotRequired[bool]
    # Best-effort location inferred from the user's recent language during crisis flow.
    inferred_location: NotRequired[str]
    # Search-derived crisis resources scoped to the inferred location.
    found_resources: NotRequired[list[dict[str, str]]]
    # Set by the closing therapeutic mode as a hint to the runtime that it
    # may want to shorten the inactivity timeout for this session. Does NOT
    # directly terminate the session — session termination is runtime-owned.
    closing_hint_shown: NotRequired[bool]
    # Multi-turn exercise step counter for guided_exercise mode. Reset to
    # None when the exercise is abandoned or completed.
    exercise_step: NotRequired[int | None]


class AgentState(TypedDict):
    """Shared graph state for a single turn plus persisted session context."""

    # The current user message the graph is processing.
    message: str
    # Channel stays normalized so graph logic does not depend on transport-specific code.
    channel: Channel
    # Optional until auth/session plumbing exists, but reserved for later ownership checks.
    user_id: str | None
    # Optional session identifier so the kernel can be wrapped by persistence later.
    session_id: str | None
    # Skill names are loaded externally, then resolved into prompt behavior by the graph.
    installed_skills: list[str]

    # Prior turns that may be injected into the model context for continuity.
    history: list[dict[str, str]]
    # Full durable transcript for rebuilding the next turn's session context.
    transcript: NotRequired[list[dict[str, str]]]
    # Scratch space for retrieved facts or interim context once memory exists.
    working_memory: list[str]

    # Grouped long-horizon memory and progression fields.
    memory: SessionMemoryState
    progress: SessionProgressState

    # Safety, routing, and output groupings for the current turn.
    crisis: CrisisAssessment
    routing: RoutingState
    response: ResponseState
