"""Internal state container for the OpenCouch agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict

from agent.memory.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.models import Channel, CrisisAssessment, ModeType, ResponseCategory
from agent.working_memory import WorkingMemoryEntry


def resolve_owner_id(state: dict[str, Any]) -> str:
    """Return the memory-namespace owner for the current session.

    Prefers ``user_id`` (explicit identity), falls back to
    ``session_id`` (thread-scoped identity). Raises ``ValueError``
    if neither is set — callers must provide at least one to prevent
    silent cross-contamination of memory namespaces.
    """

    owner = state.get("user_id") or state.get("session_id")
    if not owner:
        raise ValueError(
            "Cannot resolve memory owner: both user_id and session_id are "
            "absent from state. Provide at least one to prevent memory "
            "namespace cross-contamination."
        )
    return owner


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges two dicts with right-side precedence.

    Used for the ``diagnostics`` and ``progress`` state fields so
    multiple nodes can independently write their own keys without
    manually spreading the existing dict. LangGraph calls this
    reducer automatically when merging node deltas into the
    accumulated state.

    Defensive against ``None`` — if either side is ``None`` (e.g.,
    a node returns ``{"diagnostics": None}``), we treat it as an
    empty dict rather than raising ``TypeError``.
    """
    return {**(left or {}), **(right or {})}


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
    # Therapeutic modality captured at exercise start. The prompt builder
    # reads this instead of ``routing.therapeutic_approach`` so that
    # mid-exercise side-turns (clarifying, psychoeducation) cannot drift
    # the modality used for exercise continuation prompts.
    exercise_modality: NotRequired[str | None]


class RoutingState(TypedDict):
    """Routing and response style selection outputs for the current turn."""

    # Records which high-level path the graph decided to take.
    route: NotRequired[str]
    # Tracks the active non-crisis response style inside the therapeutic path.
    response_style: NotRequired[str]
    # Tracks whether the response style came from keyword, session_intent, llm, or default.
    response_style_source: NotRequired[str | None]
    # Distinguishes operational routing states from therapeutic and crisis styles.
    response_style_type: NotRequired[ModeType]
    # The therapeutic approach picked for this turn (CBT, ACT, grief, etc.).
    # Set by the dispatcher alongside the response style. "none" when no
    # approach applies (clarifying, closing). The response node reads this
    # to load the matching knowledge file into its system prompt.
    therapeutic_approach: NotRequired[str | None]
    # Active approach overlays selected for the current response node.
    active_approaches: NotRequired[list[str]]
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
    crisis_override_kind: NotRequired[CrisisOverrideOutcome]
    crisis_classifier_path: NotRequired[CrisisClassifierPath]
    crisis_llm_failure_occurred: NotRequired[bool]


class ResponseState(TypedDict):
    """Response shaping and output payload for the current turn."""

    # Turn-specific prompt shaping guidance derived after mode selection.
    guidance: NotRequired[str]
    # Makes the output type explicit before the final response is returned.
    kind: NotRequired[ResponseCategory]
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

    # Prior turns accumulated via ``operator.add`` reducer. Each turn
    # emits only the new entries (current user message at init, assistant
    # response at finalize); the checkpointer + reducer accumulates them
    # across turns automatically. One-shot callers (``run_agent``) without
    # a checkpointer see only the entries emitted within that single turn.
    history: Annotated[list[dict[str, str]], operator.add]
    # Full durable transcript — same reducer semantics as ``history``.
    transcript: NotRequired[Annotated[list[dict[str, str]], operator.add]]
    # Raw retrieved memory entries for the current turn. Prompt/CLI
    # surfaces format these on demand rather than storing pre-rendered
    # prose in state.
    working_memory: list[WorkingMemoryEntry]

    # Grouped long-horizon memory and progression fields.
    memory: SessionMemoryState
    # Uses ``_merge_dicts`` reducer so that ``build_initial_state`` can
    # set per-turn fields (``turn_count``, ``stage``) while the
    # checkpoint preserves cross-turn fields (``exercise_type``,
    # ``exercise_step``). Without the reducer, the fresh progress dict
    # from the input would overwrite the checkpoint's progress and
    # destroy any in-progress exercise state.
    progress: Annotated[SessionProgressState, _merge_dicts]

    # Safety, routing, and output groupings for the current turn.
    crisis: CrisisAssessment
    # Uses ``_merge_dicts`` reducer so that ``build_initial_state`` can
    # reset per-turn fields (``route``, ``mode``, ``mode_source``,
    # ``mode_type``) while the checkpoint preserves cross-turn fields
    # like ``modality``. Without the reducer, the fresh routing dict
    # from the input would overwrite the checkpoint's routing and
    # destroy the modality set by the dispatcher on the prior turn —
    # breaking multi-turn exercise modality continuity.
    routing: Annotated[RoutingState, _merge_dicts]
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
    diagnostics: NotRequired[Annotated[dict[str, Any], _merge_dicts]]
