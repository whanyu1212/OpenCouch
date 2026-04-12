"""Internal state container for the OpenCouch agent graph."""

from typing import Any, NotRequired, TypedDict

from agent.memory.models import CrisisClassifierPath, CrisisOverrideKind
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

    # ── Procedural memory (v0.7 Stage C) ────────────────────────────────
    # Rendered procedural rule strings loaded by ``load_memory_node`` at
    # the start of each turn. Populated from the user's
    # :class:`ProceduralProfile` via the helpers in
    # ``agent/memory/procedural.py``.
    #
    # STRUCTURALLY SEPARATE from ``working_memory``. Procedural rules
    # are CONSTRAINTS (silent style-shaping directives the response
    # builders apply as a system-prompt suffix), not CONTENT (semantic
    # facts and episodic summaries that the agent may or may not
    # reference in its reply). The two need different prompt treatment
    # in v0.7 Stage D — keeping them in different state fields prevents
    # the stringly-typed-prefix-parsing drift that would happen if
    # rules and content shared a flat list.
    #
    # ``proactive_recall_enabled`` is the user's ``/memory recall on|off``
    # toggle. When False (the default), Stage D injects a "do not
    # explicitly reference past sessions" constraint into the system
    # prompt. When True, the constraint is relaxed. The field is
    # NotRequired because incognito-mode sessions never populate it.
    procedural_rules: NotRequired[list[str]]
    proactive_recall_enabled: NotRequired[bool]


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

    # ── Multi-turn exercise state (v0.6 Stage C) ─────────────────────────
    # When the guided_exercise mode node starts a multi-turn exercise, it
    # writes the exercise identifier and current step index here. The
    # dispatcher reads these fields on subsequent turns via a fast-path
    # check: if ``exercise_type`` and ``exercise_step`` are both set, the
    # dispatcher short-circuits to the guided_exercise node instead of
    # re-classifying the message. This is how multi-turn exercise state
    # persists across turns without requiring the LLM classifier to read
    # prior history and infer "we're mid-exercise."
    #
    # Both fields are cleared (set to None) when the exercise completes
    # naturally OR when the user exits mid-exercise. The guided_exercise
    # node owns the lifecycle; the dispatcher only reads.
    #
    # v0.6 Stage C supports exactly one exercise type:
    #   - "grounding_5_4_3_2_1" — the 5-4-3-2-1 sensory grounding exercise
    # Future stages can add more exercise_type values without schema
    # changes; the type is stored as a string so the schema doesn't need
    # to enumerate every supported exercise at the TypedDict level.
    exercise_type: NotRequired[str | None]
    exercise_step: NotRequired[int | None]


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

    # ── Crisis debug metadata (v0.2) ─────────────────────────────────────
    # These three fields are written by ``run_crisis_gate_node`` on the
    # crisis branch and read by ``run_crisis_log_node`` so the audit
    # record accurately reflects which code path produced the
    # classification. They're prefixed ``crisis_`` to make ownership
    # unambiguous and prevent collisions with any future therapeutic
    # classifier metadata.
    #
    # All three are NotRequired — non-crisis turns leave them unset,
    # and the log node's fallback path reads defaults if missing.
    crisis_override_kind: NotRequired[CrisisOverrideKind]
    crisis_classifier_path: NotRequired[CrisisClassifierPath]
    crisis_llm_failure_occurred: NotRequired[bool]


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

    # ── Per-turn diagnostics (v0.8 observability pass) ───────────────────
    # Free-form dict nodes write into for CLI observability. Typical
    # entries include per-stage timings (``load_memory_ms``,
    # ``crisis_gate_ms``, ``extract_facts_ms``, etc.) and memory-write
    # counters (``semantic_writes``, ``procedural_writes``). Not part
    # of the stable schema — nodes add whatever keys help debug
    # dogfood turns, and ``state_to_output`` copies the dict into
    # ``AgentOutput.diagnostics`` so the CLI can render it.
    #
    # NotRequired because only the CLI reads it today; other callers
    # (unit tests, eval runners) can leave it unset and everything
    # downstream falls back to an empty dict via ``state.get``.
    diagnostics: NotRequired[dict[str, Any]]
