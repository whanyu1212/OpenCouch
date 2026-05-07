"""Typed state contracts for the OpenCouch LangGraph workflow.

The top-level graph in ``agent.graph`` uses ``AgentGraphInputState`` as its
input schema, ``AgentState`` as its internal schema, and
``AgentGraphOutputState`` as its public output schema. The therapeutic subgraph
imports the smaller state fragments below to define its own input and output
boundaries.

LangGraph treats each top-level key as a channel. Reducer-backed channels can
receive partial deltas from multiple nodes or turns: ``transcript`` appends
list entries with ``operator.add``, while grouped dict channels use
``_merge_dicts`` so nodes can update only the nested fields they own.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Annotated, Any, NotRequired, TypedDict

from agent.audit.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.models import Channel, CrisisAssessment
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
    """Session-level counters.

    ``build_initial_state`` seeds ``turn_count`` for each turn, using
    checkpointed state when available. Persistent runtime summaries, feedback
    records, and memory extractors read ``turn_count`` for provenance.
    """

    turn_count: int
    is_guest: NotRequired[bool]


class ExerciseState(TypedDict):
    """Active guided-exercise continuity state.

    ``guided_exercise`` owns writes to this channel when starting, advancing,
    completing, or exiting an exercise. The therapeutic dispatcher and prompt
    builders read it to keep side turns inside an active exercise and to reuse
    the therapeutic approach captured when the exercise began.
    """

    exercise_type: NotRequired[str | None]
    exercise_step: NotRequired[int | None]
    exercise_therapeutic_approach: NotRequired[str | None]
    exercise_selection_options: NotRequired[list[str] | None]


class MemoryControlState(TypedDict):
    """User-directed memory-control continuity.

    ``memory_control_node`` writes pending destructive actions here so the next
    turn can confirm or cancel them without relying on the LLM to infer which
    record was meant. ``memory_control_gate_node`` writes the current turn's
    explicit memory command into ``action``.
    """

    pending_action: NotRequired[dict[str, Any] | None]
    action: NotRequired[dict[str, Any]]


class GroundedLookupState(TypedDict):
    """Explicit factual lookup scratch state.

    ``grounded_lookup_gate_node`` writes the current turn's search query and
    initial status. ``grounded_answer_node`` updates the status after attempting
    the grounded response.
    """

    query: NotRequired[str]
    status: NotRequired[str]


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
    ``finalize_turn_node`` appends the assistant turn. The transcript uses
    ``operator.add`` so checkpointed conversation is extended instead of
    overwritten. ``load_memory_node`` owns ``working_memory`` for prompt-time
    semantic and episodic recall.
    """

    transcript: NotRequired[Annotated[list[dict[str, str]], operator.add]]
    working_memory: list[WorkingMemoryEntry]


class AgentPersistentState(TypedDict):
    """Reducer-backed continuity channels persisted by checkpoints.

    These grouped dicts are the long-lived channels inside LangGraph state.
    Nodes should return partial nested deltas for only the fields they own; the
    dict reducers preserve existing sibling keys from prior turns.
    """

    session_memory: Annotated[SessionMemoryState, _merge_dicts]
    procedural_profile: Annotated[ProceduralProfileState, _merge_dicts]
    session_progress: Annotated[SessionProgressState, _merge_dicts]
    exercise_state: Annotated[ExerciseState, _merge_dicts]
    memory_control: Annotated[MemoryControlState, _merge_dicts]
    grounded_lookup: Annotated[GroundedLookupState, _merge_dicts]


class AgentCrisisState(TypedDict):
    """Current-turn crisis assessment shared by both graph branches.

    ``crisis_gate_node`` owns this field. Crisis response, crisis logging, and
    downstream output conversion read it; therapeutic turns keep the safe
    assessment so callers still receive explicit safety metadata.
    """

    crisis: CrisisAssessment


class AgentGraphOutputState(AgentCrisisState, TypedDict):
    """Public parent-graph output channels.

    ``agent.graph`` registers this as the top-level output schema. Response
    nodes own the response fields, ``guided_exercise`` may set
    ``should_persist_memory`` on completion, and each observability-producing
    node contributes to ``diagnostics`` through the merge reducer.
    """

    therapeutic_approach: NotRequired[str | None]
    response_style: NotRequired[str]
    response_text: NotRequired[str]
    should_persist_memory: NotRequired[bool]
    diagnostics: NotRequired[Annotated[dict[str, Any], _merge_dicts]]


class AgentPrivateState(TypedDict):
    """Internal-only routing, audit, and scratch channels.

    These fields are available to nodes during graph execution but are not part
    of the public ``AgentOutput``. ``route`` lets extractors skip crisis turns,
    ``crisis_audit`` feeds the crisis log, and ``crisis_resource_lookup_node``
    writes ``inferred_location`` / ``found_resources`` /
    ``resource_lookup_status`` for crisis-resource lookup turns.
    """

    route: NotRequired[str]
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
    """Top-level graph input schema produced by ``build_initial_state``.

    The input schema includes the current user turn, clean defaults for
    turn-scoped channels, and reducer-backed continuity groups. When a
    checkpointer is active, LangGraph merges these inputs with the prior
    checkpoint so persistent channels are preserved instead of reset.
    """


class AgentState(AgentGraphInputState):
    """Full internal state schema used by graph nodes.

    Top-level nodes, therapeutic response nodes, prompt builders, tests, and
    evals use this type when reading or returning state-like dictionaries. It
    is not a public API response shape; use ``AgentGraphOutputState`` and
    ``agent.graph.state_to_output`` for caller-facing data.
    """
