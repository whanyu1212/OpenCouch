"""Pydantic models for the memory layer + therapeutic subgraph.

These are the **internal** data shapes used by the agent's memory nodes,
store implementations, and therapeutic dispatch. They are distinct from
``agent/models.py``, which owns the **public** contract types (AgentInput,
AgentOutput, CrisisAssessment, etc.).

Two design rules apply to every type in this file:

1. **Structured output first.** Every type that is produced by an LLM call
   is a target for ``generate_structured`` — the LLM is given the type as
   its response_schema, it returns strict JSON, and pydantic validates.
   That means every field must be expressible as JSON (no datetime objects,
   no Pydantic-only types, no custom serializers) unless the type is NOT
   an LLM target.

2. **Controlled vocabularies.** Anywhere a string could drift across
   extractions (kind values, mode names, confidence levels, proposal
   types), we use ``Literal[...]`` to nail the vocabulary at type-check
   time. Pydantic's built-in validator catches runtime mismatches for
   free — no custom ``@field_validator`` needed.

Conventions:
- Confidence levels are ``low | medium | high`` to match the existing
  ``CrisisAssessment`` convention in ``agent/models.py``. The memory
  schema.yaml says ``low | med | high``; that's a doc drift to fix later.
- Timestamps are stored as ISO-8601 strings (not datetime objects) so the
  types round-trip through JSON cleanly for the LangGraph BaseStore.
- IDs are UUIDv7 strings (not UUID objects) for the same reason.
- Fields that are optional in the phase-1 implementation but reserved in
  the schema use ``None`` defaults rather than being omitted entirely,
  so the shape is stable across phases.

Structure of this file:
    §1. Shared primitives       (ConfidenceLevel, EntityRef, etc.)
    §2. Semantic memory models  (MemoryWrite, SemanticFact)
    §3. Episodic memory models  (SessionArc)
    §4. Procedural memory models (ProceduralRule, ProceduralProfile)
    §5. Relationship models      (RelationshipKind, RelationshipEdge)
    §6. Crisis log models        (CrisisLogRecord, CrisisLogAggregate)
    §7. Consolidation models     (phase 4; deferred skeletons)
    §8. Therapeutic models       (DispatchDecision, TherapeuticMode)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─── §1. Shared primitives ──────────────────────────────────────────────────


# Confidence level used by every extraction and consolidation type.
# Matches CrisisAssessment's convention in agent/models.py.
ConfidenceLevel = Literal["low", "medium", "high"]


# The eight entity types that can appear as a subject or object in a
# semantic fact. Matches schema.yaml §3 entity_types.
EntityType = Literal[
    "User",
    "Person",
    "Concern",
    "Event",
    "CopingStrategy",
    "Goal",
    "Session",
    "Turn",
]


class EntityRef(BaseModel):
    """A reference to a graph entity by type and canonical identifier.

    Used as the subject and object of a MemoryWrite. The identifier is
    the canonical form (e.g., "Sarah" for a Person, "work stress" for a
    Concern) that the entity resolver uses to dedupe across extractions.

    This is NOT a full entity record — it's a pointer. The full entity
    lives in the graph store and can be fetched via ``store.get_entity``.
    """

    type: EntityType
    identifier: str = Field(min_length=1, max_length=200)


# The seven edge types used by hot-path extraction. Inter-entity edges
# (OVERLAPS_WITH, HELPED_WITH, DID_NOT_HELP, TRIGGERED, RELATES_TO) are
# populated by phase-3+ background reasoning, not extraction, so they
# don't appear in the extractor's schema.
HotPathEdgeType = Literal[
    "KNOWS",
    "WORRIES_ABOUT",
    "EXPERIENCED",
    "USES",
    "WANTS",
    "PARTICIPATED_IN",
    "MENTIONED_IN",
]


# ─── §2. Semantic memory models ─────────────────────────────────────────────


# Schema.yaml §2 namespaces.semantic.categories
SemanticCategory = Literal[
    "loss",
    "preference",
    "coping_strategy",
    "relationship",
    "trigger",
    "goal",
    "context",
]


class MemoryWrite(BaseModel):
    """One fact extracted from a turn, structured as a graph triple.

    This is the **LLM output shape** for the semantic extraction node.
    The extractor is given the current turn and recent history, and it
    returns zero or more MemoryWrite items. Each item becomes:

    1. A semantic fact record in the semantic namespace, AND
    2. An edge in the graph store from subject → object with the
       predicate as the edge type.

    The two-in-one representation is deliberate: we want vector search
    (against the fact's text form) AND graph traversal (against the
    triple structure) to work from the same underlying data.

    The extractor is constrained to produce ONLY facts it can support
    with a direct quote from the turn. Speculative extractions ("the
    user seems to be implying X") are prohibited by the system prompt.
    """

    category: SemanticCategory
    subject: EntityRef
    predicate: HotPathEdgeType
    object: EntityRef
    evidence_quote: str = Field(min_length=1, max_length=280)
    confidence: ConfidenceLevel

    # Provenance — the extractor fills these in so the writer can
    # attach the fact to the correct session/turn without re-deriving.
    source_session_id: str
    source_turn_index: int = Field(ge=0)


class SemanticFact(BaseModel):
    """A stored semantic fact record in the memory store.

    This is the **stored shape** — what actually lives in the semantic
    namespace. It's the MemoryWrite plus the metadata that the writer
    adds (id, timestamps, dormant/superseded state, user_visible flag).

    SemanticFact is what the load-memory node retrieves and what the
    CLI ``/memory forget fact <n>`` command operates on.
    """

    id: str  # uuid7 string
    category: SemanticCategory
    subject: EntityRef
    predicate: HotPathEdgeType
    object: EntityRef
    evidence_quote: str
    confidence: ConfidenceLevel
    source_session_id: str
    source_turn_index: int

    created_at: str  # ISO-8601
    last_referenced_at: str  # ISO-8601; bumped on retrieval or re-extract
    dormant_at: str | None = None  # set by consolidation; not deleted
    superseded_by: str | None = None  # uuid7 of the fact that replaced this one

    user_visible: bool = True


class ExtractionResult(BaseModel):
    """The structured-output shape returned by the semantic extractor LLM.

    The extractor LLM is given a turn and asked to produce zero or more
    :class:`MemoryWrite` items worth persisting as long-term semantic
    facts. Two design notes:

    1. **Zero facts is the common case.** The system prompt enforces a
       conservative stance — most turns (small talk, transient feelings,
       ambiguous statements) produce an empty ``facts`` list. This is
       by design; an aggressive extractor that saves every detail would
       produce "creepy memory" failures that erode user trust.
    2. **The ``reason`` field is always populated**, even on empty
       extractions. It provides a human-readable breadcrumb that
       explains why nothing was extracted (e.g., "small talk, no
       extractable facts") or summarizes what was extracted (e.g.,
       "extracted 2 relationship facts about user's sister"). The
       reason is for logs and debugging, not for the user — it should
       never be surfaced in response text.

    The wrapper shape (vs. a plain ``list[MemoryWrite]``) makes the
    empty case carry observability signal. Without it, zero-fact turns
    would be indistinguishable from "the LLM returned nothing because
    it got confused," and we'd lose the ability to tune the prompt
    based on why extractions succeed or fail.
    """

    facts: list[MemoryWrite] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=240)


# ─── §3. Episodic memory models ─────────────────────────────────────────────


class MoodArc(BaseModel):
    """Session-level mood summary — how the user opened and closed.

    Stored as short descriptor strings ("anxious", "tentatively_okay",
    "calmer") rather than numeric scores. The summarizer LLM produces
    these from the full transcript.
    """

    opened: str = Field(min_length=1, max_length=40)
    closed: str = Field(min_length=1, max_length=40)


class SessionArc(BaseModel):
    """A completed session's structured summary, stored in episodic memory.

    Written once per session by ``summarize_session_node`` at session end.
    One LLM call per session (not per turn), reading the full transcript
    from the checkpointer. This is the **LLM output shape** for the
    summarizer; the stored shape adds an id and user_visible flag.

    The summary is 2-4 sentences. Longer summaries tend to lose focus
    and shorter ones lose meaningful detail.
    """

    session_id: str
    started_at: str  # ISO-8601
    ended_at: str  # ISO-8601
    duration_seconds: int = Field(ge=0)
    turn_count: int = Field(ge=0)

    # 1-3 high-level tags, e.g. ["grief", "sleep"]. The summarizer is
    # constrained to pick from a short controlled list of themes to
    # keep downstream filtering and grouping reliable.
    primary_themes: list[str] = Field(min_length=0, max_length=3)

    summary: str = Field(min_length=1, max_length=600)
    mood_arc: MoodArc

    open_loops: list[str] = Field(default_factory=list)
    resolved_threads: list[str] = Field(default_factory=list)

    crisis_level_max: Literal[0, 1, 2, 3] = 0


class StoredSessionArc(SessionArc):
    """Stored shape of a SessionArc with memory-layer metadata added."""

    id: str  # uuid7
    user_visible: bool = True


# ─── §4. Procedural memory models ───────────────────────────────────────────


# Schema.yaml §2 namespaces.procedural.rules[].source
ProceduralRuleSource = Literal[
    "explicit_user",  # user directly asked us to remember a preference
    "consolidation",  # phase 4 nightly consolidation inferred the pattern
    "manual",  # operator-entered rule (debugging, overrides)
]


class ProceduralRule(BaseModel):
    """One learned rule about how to talk to this specific user.

    Rules MUST be written in **second-person, evidence-grounded** form.
    The system prompt for the writer enforces this:

        Good: "You've said meditation makes you more anxious."
        Bad:  "User dislikes meditation."

    The internal rule string IS the displayed string — there is no
    curation pass. See schema.yaml §9 q3 for the full visibility
    rationale.
    """

    rule: str = Field(min_length=1, max_length=280)
    evidence: list[str] = Field(default_factory=list)  # quotes or fact_ids
    confidence: ConfidenceLevel
    added_at: str  # ISO-8601
    source: ProceduralRuleSource


class ProceduralProfile(BaseModel):
    """The single procedural-memory document for a user.

    Stored as a profile (one doc per user, continuously updated) rather
    than a collection, because there are typically only ~5-15 rules and
    they need to be coherent. See schema.yaml §2 namespaces.procedural
    for the rationale.

    This is the shape stored under namespace=(user_id, "procedural") with
    key="user_response_style".
    """

    proactive_recall_enabled: bool = False  # see schema.yaml §9 q4
    rules: list[ProceduralRule] = Field(default_factory=list)
    last_consolidated_at: str | None = None  # ISO-8601


# ─── §5. Relationship models (RELATES_TO edges) ─────────────────────────────


# The 27-value controlled vocabulary from schema.yaml §4
# (relationship_kind_allowlist). Anything outside this list normalizes to
# "other" and the original phrasing is preserved in kind_raw.
RelationshipKind = Literal[
    # ── Family of origin ──────────────
    "mother",
    "father",
    "parent",  # for "my parent" or non-binary
    "step_mother",
    "step_father",
    "sister",
    "brother",
    "sibling",
    "child",
    "grandparent",
    "grandchild",
    "other_family",
    # ── Romantic / partnership ────────
    "partner",
    "spouse",
    "ex_partner",
    "ex_spouse",
    # ── Friendship ────────────────────
    "friend",
    "close_friend",
    "estranged_friend",
    # ── Work / professional ───────────
    "colleague",
    "boss",
    "subordinate",
    "client",
    # ── Care relationships ────────────
    "therapist",
    "doctor",
    "caregiver",
    "dependent",
    # ── Catch-all ─────────────────────
    "other",
]

# Cluster constants for group queries (e.g. "find all family members").
# These are also exported as frozensets so callers can do
# `if edge.kind in FAMILY_KINDS: ...` without enumerating every value.
FAMILY_KINDS: frozenset[RelationshipKind] = frozenset(
    {
        "mother",
        "father",
        "parent",
        "step_mother",
        "step_father",
        "sister",
        "brother",
        "sibling",
        "child",
        "grandparent",
        "grandchild",
        "other_family",
    }
)

ROMANTIC_KINDS: frozenset[RelationshipKind] = frozenset(
    {"partner", "spouse", "ex_partner", "ex_spouse"}
)

FRIENDSHIP_KINDS: frozenset[RelationshipKind] = frozenset(
    {"friend", "close_friend", "estranged_friend"}
)

PROFESSIONAL_KINDS: frozenset[RelationshipKind] = frozenset(
    {"colleague", "boss", "subordinate", "client"}
)

CARE_KINDS: frozenset[RelationshipKind] = frozenset(
    {"therapist", "doctor", "caregiver", "dependent"}
)


class RelatesToEdge(BaseModel):
    """An inter-entity relationship edge with controlled-vocabulary kind.

    Used for Person→Person and any other catch-all relationships that
    don't fit the specific User→Entity edge types. See schema.yaml §9 q1
    for the full rationale on generic edge + kind property vs. explicit
    edge types.
    """

    kind: RelationshipKind
    # Original phrasing when kind == "other", so phase-4 consolidation
    # can review frequent kind_raw values and promote them to first-class
    # kinds. Null when kind != "other".
    kind_raw: str | None = None
    first_observed_at: str  # ISO-8601
    last_observed_at: str  # ISO-8601
    confidence: ConfidenceLevel


# ─── §6. Crisis log models ──────────────────────────────────────────────────


CrisisOverrideKind = Literal["imminent_risk", "idiomatic_safe", "none"]
CrisisClassifierPath = Literal["deterministic", "llm_fallback", "override"]


class CrisisLogRecord(BaseModel):
    """One crisis event in the always-on safety log.

    Written by ``crisis_log_node`` alongside any crisis response,
    **regardless of memory mode** (see schema.yaml §2 namespaces.crisis_log
    for the always-on asymmetry rationale).

    This record contains classifier metadata + outcome flags. It does
    NOT contain the user's message text or any conversation history —
    only what the classifier decided, how it decided it, and whether
    the response node completed.

    Retention: 90 days per-user records (phase 1 default), indefinite
    aggregate stats. See schema.yaml §9 q6 for the legal-review caveat.
    """

    id: str  # uuid7
    # SHA-256 of session_id, no reverse mapping. Safe to retain even in
    # incognito mode because it can't be traced back to a user.
    session_id_opaque: str
    # Populated only in local/synced modes. Null in incognito.
    user_id_or_null: str | None = None
    detected_at: str  # ISO-8601

    level: Literal[0, 1, 2, 3]
    override_kind: CrisisOverrideKind
    classifier_path: CrisisClassifierPath
    reason: str = Field(max_length=500)
    response_node_completed: bool
    llm_failure_occurred: bool

    # Phase 2+ fields; null in phase 1. Allows specific events to be
    # flagged for extended retention during active investigations.
    retention_extended_until: str | None = None
    retention_extended_reason: str | None = None


class CrisisLogLevelCounts(BaseModel):
    """Per-level event counts for a single day's crisis log aggregate."""

    level_0: int = Field(default=0, ge=0)
    level_1: int = Field(default=0, ge=0)
    level_2: int = Field(default=0, ge=0)
    level_3: int = Field(default=0, ge=0)


class CrisisLogPathCounts(BaseModel):
    """Per-classifier-path event counts for a single day's aggregate."""

    deterministic: int = Field(default=0, ge=0)
    llm_fallback: int = Field(default=0, ge=0)
    override: int = Field(default=0, ge=0)


class CrisisLogAggregate(BaseModel):
    """Daily rollup of crisis events with NO per-user identifiers.

    Retained indefinitely (vs. 90-day retention on CrisisLogRecord) so
    that long-term safety eval and trend detection have stable signal
    even after per-user records are purged.

    Written by the daily rollup job (deferred to phase 2). Phase 1 just
    doesn't delete per-user records before phase 2 has backfill data.
    """

    date: str  # ISO-8601 date (YYYY-MM-DD), primary key
    events_total: int = Field(default=0, ge=0)
    events_by_level: CrisisLogLevelCounts
    events_by_classifier_path: CrisisLogPathCounts
    llm_failures_total: int = Field(default=0, ge=0)
    response_node_completion_rate: float = Field(ge=0.0, le=1.0)


# ─── §7. Consolidation models (phase 4 deferred) ────────────────────────────


# Schema.yaml §7 consolidation.phase_4.proposal_types
ConsolidationProposalType = Literal[
    "merge_facts",
    "mark_contradiction",
    "promote_to_procedural",
    "infer_graph_edge",
    "mark_dormant",
]


class ConsolidationProposal(BaseModel):
    """One proposal emitted by the phase-4 consolidation LLM pass.

    The consolidation LLM is given a slice of the user's recent memory
    and asked to identify consolidation opportunities. Each opportunity
    becomes one ConsolidationProposal. High-confidence proposals are
    auto-applied; medium-confidence ones are logged for review;
    low-confidence ones are discarded.

    Every proposal MUST reference the specific fact IDs that support it
    (``evidence_fact_ids``) — this is the guard against LLM hallucination
    of patterns that don't exist. A proposal with no evidence is
    rejected before it reaches the confidence filter.

    Phase: 4 (deferred; schema designed in phase 1 so the
    ``consolidation_runs`` table doesn't need future migration).
    """

    proposal_type: ConsolidationProposalType
    confidence: ConfidenceLevel
    rationale: str = Field(min_length=1, max_length=500)
    evidence_fact_ids: list[str] = Field(min_length=1)  # non-empty required


class ConsolidationRunRecord(BaseModel):
    """One row in the ``consolidation_runs`` observability table.

    Schema is designed in phase 1 (table empty until phase 4) so that
    phase 4 doesn't require a migration. Each run of the consolidation
    pass writes exactly one row summarizing what happened.
    """

    run_id: str  # uuid7
    user_id: str
    started_at: str  # ISO-8601
    duration_seconds: float = Field(ge=0.0)

    proposals_total: int = Field(default=0, ge=0)
    proposals_applied: int = Field(default=0, ge=0)
    proposals_discarded: int = Field(default=0, ge=0)
    proposals_logged_for_review: int = Field(default=0, ge=0)

    # Detailed audit trail of every merge proposal — source facts,
    # resulting merged fact, similarity score, rationale. Used for
    # /memory restore <id> in phase 4.
    merge_proposals: list[MergeProposalDetail] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)


class MergeProposalDetail(BaseModel):
    """Detailed record of one merge proposal for audit/rollback.

    Stored in ``ConsolidationRunRecord.merge_proposals``. Used by the
    phase-4 ``/memory restore <id>`` command to undo a bad merge.
    """

    source_fact_ids: list[str] = Field(min_length=2)  # at least 2 to merge
    merged_fact_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    rationale: str


# Rebuild the forward reference now that MergeProposalDetail is defined.
# Without this, ConsolidationRunRecord can't resolve the list item type.
ConsolidationRunRecord.model_rebuild()


# ─── §8. Therapeutic models ─────────────────────────────────────────────────


# The six therapeutic modes from agent/therapeutic/nodes_sketch.py.
# Duplicated here as a Literal so models.py stays self-contained and
# doesn't import from the therapeutic package. The source of truth is
# this file; nodes_sketch.py re-declares the same Literal for its own
# routing annotations (which is fine — Literal equality is structural,
# not nominal).
TherapeuticMode = Literal[
    "supportive",
    "reflective",
    "psychoeducation",
    "guided_exercise",
    "closing",
    "clarifying",
]


class DispatchDecision(BaseModel):
    """The structured output of the therapeutic_dispatch_node LLM call.

    The dispatcher is given the current message, recent history, and
    retrieved memory, and it returns a DispatchDecision. The decision
    includes the picked mode and a brief reasoning string for
    observability (LangSmith spans, debugging).

    The reasoning is SHORT — single sentence, max ~40 words. It exists
    for debugging, not for the user. It should never be shown to the
    user.
    """

    mode: TherapeuticMode
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: ConfidenceLevel


# ─── Type exports ───────────────────────────────────────────────────────────
#
# Explicit __all__ list so `from agent.memory.models import *` imports
# only the public shapes, not pydantic internals or type-alias helpers.

__all__ = [
    # §1 primitives
    "ConfidenceLevel",
    "EntityType",
    "EntityRef",
    "HotPathEdgeType",
    # §2 semantic
    "SemanticCategory",
    "MemoryWrite",
    "SemanticFact",
    "ExtractionResult",
    # §3 episodic
    "MoodArc",
    "SessionArc",
    "StoredSessionArc",
    # §4 procedural
    "ProceduralRuleSource",
    "ProceduralRule",
    "ProceduralProfile",
    # §5 relationships
    "RelationshipKind",
    "FAMILY_KINDS",
    "ROMANTIC_KINDS",
    "FRIENDSHIP_KINDS",
    "PROFESSIONAL_KINDS",
    "CARE_KINDS",
    "RelatesToEdge",
    # §6 crisis log
    "CrisisOverrideKind",
    "CrisisClassifierPath",
    "CrisisLogRecord",
    "CrisisLogLevelCounts",
    "CrisisLogPathCounts",
    "CrisisLogAggregate",
    # §7 consolidation
    "ConsolidationProposalType",
    "ConsolidationProposal",
    "ConsolidationRunRecord",
    "MergeProposalDetail",
    # §8 therapeutic
    "TherapeuticMode",
    "DispatchDecision",
]
