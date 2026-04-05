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
    # Scratch space for retrieved facts or interim context once memory exists.
    working_memory: list[str]

    # Safety decision must live in state so every downstream node can respect it.
    crisis: CrisisAssessment

    # Records which high-level path the graph decided to take.
    route: NotRequired[str]
    # Makes the output type explicit before the final response is returned.
    response_kind: NotRequired[ResponseKind]
    # Holds the generated text from the chosen response node.
    response_text: NotRequired[str]

    # Signals whether a later wrapper should persist memory after the reply is complete.
    should_persist_memory: NotRequired[bool]
