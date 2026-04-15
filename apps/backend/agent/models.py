from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.working_memory import WorkingMemoryEntry


# Normalizes the surface the message came from so routing can stay provider-agnostic.
class Channel(str, Enum):
    TEST = "test"
    WEB = "web"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    VOICE = "voice"


# Distinguishes conversation turns when building history for the model.
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# Keeps the public agent result explicit about whether the reply is normal or safety-driven.
class ResponseKind(str, Enum):
    THERAPEUTIC = "therapeutic"
    CRISIS = "crisis"


class ModeType(str, Enum):
    OPERATIONAL = "operational"
    THERAPEUTIC = "therapeutic"
    CRISIS = "crisis"


# Represents one prior conversation turn in a validated, serializable form.
class Message(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)
    # v0.8 observability pass: optional per-turn mode annotation for
    # assistant messages. Populated by ``run_finalize_turn_node`` when
    # the assistant reply is appended, sourced from the active routing
    # mode (support, psychoeducation, guided_exercise, closing, crisis,
    # etc.). User messages leave this as ``None``. The CLI's ``/history``
    # renderer shows it as a compact column so operators can retroactively
    # see which mode shaped each reply. Optional + default-None so legacy
    # Message constructors (tests, eval harnesses) don't need to care.
    mode: str | None = None


# Carries the safety decision separately from the reply so crisis handling is inspectable.
class CrisisAssessment(BaseModel):
    level: Literal[0, 1, 2, 3] = 0
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""
    needs_crisis_response: bool = False
    needs_clarification: bool = False


# Defines the minimal external contract needed to run the MVP agent kernel.
class AgentInput(BaseModel):
    message: str = Field(min_length=1)
    channel: Channel = Channel.TEST
    user_id: str | None = None
    session_id: str | None = None
    history: list[Message] = Field(default_factory=list)
    working_memory: list[WorkingMemoryEntry] = Field(default_factory=list)
    installed_skills: list[str] = Field(default_factory=list)


# Defines the normalized result so API, CLI, and tests can all consume the same shape.
class AgentOutput(BaseModel):
    response_text: str
    response_type: ResponseKind
    crisis: CrisisAssessment
    mode: str | None = None
    mode_type: ModeType | None = None
    mode_source: str | None = None
    modality: str | None = None
    should_persist_memory: bool = False
    # Per-turn diagnostics added for CLI observability. Holds stage
    # timings (``load_memory_ms``, ``crisis_gate_ms``,
    # ``therapeutic_ms``, ``extract_facts_ms``,
    # ``extract_procedural_ms``, ``turn_total_ms``), plus memory-
    # write deltas (``semantic_writes``, ``procedural_writes``).
    # Nodes write their own timings into ``state["diagnostics"]``
    # (see :class:`SessionDiagnostics` in ``agent/state.py``), and
    # ``state_to_output`` copies the dict through to this field.
    # Intentionally typed as ``dict[str, Any]`` rather than a
    # pydantic sub-model so nodes can freely add keys without a
    # schema ripple — this is an observability channel, not a
    # stable contract.
    diagnostics: dict[str, Any] = Field(default_factory=dict)


# ── Stream event types ────────────────────────────────────────────────────────


class StatusEvent(BaseModel):
    """Pipeline progress update emitted before response generation begins."""

    type: Literal["status"] = "status"
    stage: str
    detail: str = ""


class ChunkEvent(BaseModel):
    """Incremental text chunk from response generation.

    v0.9: emitted in real time as the LLM generates tokens. Mode nodes
    call ``get_stream_writer()`` to push chunks through LangGraph's
    custom stream channel, which ``run_turn_stream`` maps to ChunkEvents.
    The frontend renders these incrementally for ~2s time-to-first-token.
    """

    type: Literal["chunk"] = "chunk"
    text: str


class DoneEvent(BaseModel):
    """Terminal event carrying the complete agent output."""

    type: Literal["done"] = "done"
    output: AgentOutput


StreamEvent = StatusEvent | ChunkEvent | DoneEvent
