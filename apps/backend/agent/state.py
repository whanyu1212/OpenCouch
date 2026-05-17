"""Typed state contracts for the OpenCouch text runtime.

``AgentGraphInputState`` and related names remain for compatibility with older
callers, but the active runtime is OpenAI Agents SDK based. ``AgentState`` is
the internal product snapshot persisted by ``PersistentAgentRuntime``.

The ``Annotated`` reducer metadata is retained as documentation for how turn
deltas are merged: ``transcript`` appends list entries with ``operator.add``,
while grouped dict channels use ``_merge_dicts`` so services can update only
the nested fields they own.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from agent.audit.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.memory.models import GuidancePermission, SessionIntent, SessionStage
from agent.models import Channel, CrisisAssessment, SessionAction
from agent.memory.entries import WorkingMemoryEntry


def resolve_owner_id(state: Mapping[str, Any]) -> str:
    """Return the memory owner for the current session.

    Args:
        state: The current agent state or state-like dict.

    Returns:
        The resolved owner identifier.

    Raises:
        ValueError: If neither ``user_id`` nor ``session_id`` is present.
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
    """Merge two dict-like reducer values with right-side precedence.

    Args:
        left: The existing accumulated state value.
        right: The incoming node delta.

    Returns:
        A merged dict. ``None`` is treated as an empty dict on either side.
    """
    return {**(left or {}), **(right or {})}


class SessionMemoryState(TypedDict):
    """Prompt-visible session continuity carried across turns.

    ``load_memory_node`` writes the current summary into this channel and
    preserves sibling fields. Therapeutic prompt builders read it indirectly
    through the subgraph input, while session-end summarization and memory
    promotion use the corresponding persisted episodic records rather than this
    channel as the source of truth.
    """

    summary: str
    active_concerns: NotRequired[list[str]]
    open_loops: NotRequired[list[str]]
    current_goal: NotRequired[str | None]


class ProceduralProfileState(TypedDict):
    """Prompt-shaping procedural profile loaded for the current owner.

    ``load_memory_node`` reads the durable procedural profile from the memory
    store and writes the prompt-ready rules plus the proactive-recall toggle
    here. Therapeutic prompt builders read this channel; CLI/API mutation
    commands update the backing store rather than this in-flight state directly.
    """

    procedural_rules: NotRequired[list[str]]
    proactive_recall_enabled: NotRequired[bool]


class SessionProgressState(TypedDict):
    """Session-level counters and arc signals.

    ``build_initial_state`` seeds ``turn_count`` for each turn, using
    persisted runtime state when available. Persistent runtime summaries, feedback
    records, and memory extractors read ``turn_count`` for provenance. The
    therapeutic dispatcher may also write ``session_intent`` and
    ``session_stage`` to keep response generation aware of the conversation arc
    without adding extra runtime branches.
    """

    turn_count: int
    is_guest: NotRequired[bool]
    session_intent: NotRequired[SessionIntent]
    session_stage: NotRequired[SessionStage]
    guidance_permission: NotRequired[GuidancePermission]


class ExerciseState(TypedDict):
    """Active guided-exercise continuity state.

    ``guided_exercise`` owns writes to this channel when starting, advancing,
    completing, or exiting an exercise. The therapeutic dispatcher and prompt
    builders read it to keep side turns inside an active exercise and to reuse
    the therapeutic approach captured when the exercise began.
    """

    exercise_type: NotRequired[str | None]
    exercise_step: NotRequired[int | None]
    exercise_step_id: NotRequired[str | None]
    exercise_version: NotRequired[int | None]
    exercise_therapeutic_approach: NotRequired[str | None]


def cleared_exercise_state() -> ExerciseState:
    """Return a graph delta that clears active guided-exercise continuity.

    Returns:
        ExerciseState: Exercise-state fields set to their inactive values.
    """

    return {
        "exercise_type": None,
        "exercise_step": None,
        "exercise_step_id": None,
        "exercise_version": None,
        "exercise_therapeutic_approach": None,
    }


class MemoryControlState(TypedDict):
    """User-directed memory-control continuity.

    ``memory_control_node`` writes pending destructive actions here so the next
    turn can confirm or cancel them. ``turn_dispatch_node`` writes the current
    turn's explicit memory command into ``action``.
    """

    pending_action: NotRequired[dict[str, Any] | None]
    action: NotRequired[dict[str, Any]]


class GroundedLookupState(TypedDict):
    """Explicit factual lookup scratch state.

    ``turn_dispatch_node`` writes the current turn's search query and initial
    status. ``grounded_answer_node`` updates the status after attempting the
    grounded response.
    """

    query: NotRequired[str]
    status: NotRequired[str]


class TurnLifecycleState(TypedDict):
    """Current-turn active-flow lifecycle decision.

    ``turn_dispatch_node`` writes this after deciding whether an active
    exercise or pending memory action should continue, pause, resume, or clear.
    Downstream nodes read it as behavior state. Diagnostics may mirror these
    values for observability, but diagnostics are not the source of truth.
    """

    active_flow: Literal["none", "guided_exercise", "pending_memory_action"]
    action: Literal["none", "continue", "preserve", "resume", "clear"]


class MemoryReferenceState(TypedDict):
    """Current-turn permission to reference retrieved user memories.

    ``turn_dispatch_node`` writes this after classifying the current safe
    turn. It is distinct from the durable proactive-recall toggle: a user may
    keep proactive recall off while explicitly asking "what did we work out
    last time?" for this one turn.
    """

    mode: Literal["none", "explicit"]


class CrisisAuditState(TypedDict):
    """Turn-scoped crisis-classifier provenance.

    ``crisis_gate_node`` writes this alongside the ``crisis`` assessment.
    ``crisis_log_node`` reads it to persist classifier path, override outcome,
    and LLM-failure metadata in the audit log.
    """

    crisis_override_kind: NotRequired[CrisisOverrideOutcome]
    crisis_classifier_path: NotRequired[CrisisClassifierPath]
    crisis_llm_failure_occurred: NotRequired[bool]


class AgentIdentityState(TypedDict):
    """External request identity seeded from ``AgentInput``.

    ``build_initial_state`` writes these fields at turn start. Nodes use
    ``message`` as the current user text, ``user_id`` / ``session_id`` through
    ``resolve_owner_id`` for memory namespace ownership, and
    ``installed_skills`` for capability-aware routing.
    """

    message: str
    channel: Channel
    user_id: str | None
    session_id: str | None
    installed_skills: list[str]


class AgentConversationState(TypedDict):
    """Conversation and working-memory channels used during a turn.

    ``build_initial_state`` emits the current user turn into ``transcript``.
    The text adapter appends the assistant turn. The transcript uses
    ``operator.add`` metadata to document append semantics. Turn memory owns
    ``working_memory`` for prompt-time semantic and episodic recall.
    """

    transcript: NotRequired[Annotated[list[dict[str, str]], operator.add]]
    working_memory: list[WorkingMemoryEntry]


class AgentPersistentState(TypedDict):
    """Continuity channels persisted in runtime state snapshots.

    These grouped dicts are long-lived text-runtime state. Services should
    return partial nested deltas for only the fields they own; the
    dict reducers preserve existing sibling keys from prior turns.
    """

    session_memory: Annotated[SessionMemoryState, _merge_dicts]
    procedural_profile: Annotated[ProceduralProfileState, _merge_dicts]
    session_progress: Annotated[SessionProgressState, _merge_dicts]
    exercise_state: Annotated[ExerciseState, _merge_dicts]
    memory_control: Annotated[MemoryControlState, _merge_dicts]
    grounded_lookup: Annotated[GroundedLookupState, _merge_dicts]


class AgentCrisisState(TypedDict):
    """Current-turn crisis assessment shared by runtime branches.

    The crisis gate owns this field. Crisis response, crisis logging, and
    downstream output conversion read it; therapeutic turns keep the safe
    assessment so callers still receive explicit safety metadata.
    """

    crisis: CrisisAssessment


class AgentGraphOutputState(AgentCrisisState, TypedDict):
    """Public text-runtime output channels.

    Response branches own the response fields, ``guided_exercise`` may set
    ``should_persist_memory`` on completion, and observability-producing
    services contribute to ``diagnostics`` through merge semantics.
    """

    therapeutic_approach: NotRequired[str | None]
    response_style: NotRequired[str]
    session_action: NotRequired[SessionAction]
    response_text: NotRequired[str]
    should_persist_memory: NotRequired[bool]
    diagnostics: NotRequired[Annotated[dict[str, Any], _merge_dicts]]


class AgentPrivateState(TypedDict):
    """Internal-only routing, audit, and scratch channels.

    These fields are available during runtime execution but are not part of the
    public ``AgentOutput``. ``route`` lets extractors skip crisis turns,
    ``turn_lifecycle`` carries current-turn active-flow behavior from dispatch
    to downstream nodes, ``memory_reference`` controls one-turn permission to
    cite retrieved memories, ``crisis_audit`` feeds the crisis log, and
    crisis resource lookup writes ``inferred_location`` /
    ``found_resources`` / ``resource_lookup_status`` for crisis-resource lookup
    turns.
    """

    route: NotRequired[str]
    turn_lifecycle: NotRequired[TurnLifecycleState]
    memory_reference: NotRequired[MemoryReferenceState]
    response_guidance: NotRequired[str]
    crisis_audit: NotRequired[CrisisAuditState]
    inferred_location: NotRequired[str]
    found_resources: NotRequired[list[dict[str, str]]]
    resource_lookup_status: NotRequired[str]


class AgentGraphInputState(
    AgentIdentityState,
    AgentConversationState,
    AgentPersistentState,
    AgentGraphOutputState,
    AgentPrivateState,
):
    """Top-level turn input schema produced by ``build_initial_state``.

    The input schema includes the current user turn, clean defaults for
    turn-scoped channels, and continuity groups. The OpenAI adapter merges this
    turn input with prior runtime state so persistent channels are preserved
    instead of reset.
    """


class AgentState(AgentGraphInputState):
    """Full internal state schema used by text-runtime services.

    Runtime branches, prompt builders, tests, and evals use this type when
    reading or returning state-like dictionaries. It is not a public API
    response shape; use ``AgentGraphOutputState`` and
    ``agent.graph.state_to_output`` for caller-facing data.
    """
