"""Internal state container for the MVP agent graph.

This is intentionally more detailed than the external input/output models because it
tracks intermediate decisions between graph nodes.
"""

from typing import NotRequired, TypedDict

from agent.models import Channel, CrisisAssessment, ResponseKind


class AgentState(TypedDict):
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
    # Rolling summary used to keep the session coherent as history grows.
    session_summary: str
    # Current user concerns that should stay salient across turns.
    active_concerns: list[str]
    # Unresolved threads the agent should avoid dropping.
    open_loops: list[str]
    # Best-effort guess at what the user wants from the current session.
    current_goal: str | None
    # Soft steering signal for the overall shape of the current session.
    session_intent: NotRequired[str | None]
    # Tracks whether the current intent came from explicit user language or inference.
    session_intent_source: NotRequired[str | None]
    # Structured session progression signal for shaping the next response.
    session_stage: NotRequired[str]
    # Tracks whether the current stage came from deterministic logic or LLM refinement.
    session_stage_source: NotRequired[str | None]
    # Short rationale for debugging and evals.
    session_stage_reason: NotRequired[str]
    # Count of user turns including the current inbound message.
    turn_count: int

    # Safety decision must live in state so every downstream node can respect it.
    crisis: CrisisAssessment

    # Records which high-level path the graph decided to take.
    route: NotRequired[str]
    # Tracks the active non-crisis response mode inside the therapeutic path.
    mode: NotRequired[str]
    # Active modality overlays selected for the current response node.
    active_modalities: NotRequired[list[str]]
    # Makes the output type explicit before the final response is returned.
    response_type: NotRequired[ResponseKind]
    # Holds the generated text from the chosen response node.
    response_text: NotRequired[str]

    # Signals whether a later wrapper should persist memory after the reply is complete.
    should_persist_memory: NotRequired[bool]
