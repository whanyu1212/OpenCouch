"""Semantic fact extraction node with hot-path policy + session buffering.

Runs after the response generation node on every turn and extracts
zero or more memory-worthy facts from the current user message. Phase 1
splits "the extractor produced a fact" from "the store should persist it
immediately": each LLM output first becomes a semantic candidate, the
deterministic write policy decides whether it is safe to commit now, and
only ``commit_now`` candidates continue to dedup + store write.
Candidates marked for later review are buffered in the runtime-managed
session buffer and revisited at session end.

Phase-1 design rules:

1. **Conservative extraction.** The LLM is told via system prompt that
   most turns should produce zero facts. Small talk, transient moods,
   speculation, and ambiguous statements are all filtered out.

2. **Silent skip on incognito or no LLM.** The node is always registered
   in the parent graph, but it no-ops when either (a) the memory mode
   is INCOGNITO, or (b) no LLM client is available. Both are legitimate
   v0.3 states that shouldn't trigger any extraction-path machinery.

3. **Policy before persistence.** Extractor outputs become
   :class:`SemanticCandidate` instances first. The deterministic policy
   layer can downgrade a candidate to ``commit_at_session_end``,
   ``require_repetition``, or ``drop``. Session-end / repetition
   candidates are buffered for later review rather than written on the
   hot path.

4. **Hot-path dedup.** Every immediate-commit fact is checked against
   existing store records via :func:`find_near_duplicate`. Duplicates
   bump the matched record's ``last_referenced_at`` timestamp instead
   of writing a new row. Dedup uses token-set Jaccard similarity on
   evidence quotes; the vector-similarity variant lands in v0.8.

5. **Failures degrade silently.** LLM errors, schema validation errors,
   and store write errors are all logged at WARNING level but never
   propagate. The extraction node is a side-effect node — a failure
   here must not fail the parent turn.

6. **Always return an empty state delta.** The write is a side effect;
   state isn't modified. Returning ``{}`` is the canonical "I touched
   nothing" signal for LangGraph delta-return nodes.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from langgraph.runtime import Runtime

from agent.memory.candidates import build_semantic_candidate
from agent.memory.dedup import find_near_duplicate
from agent.memory.extraction_prompts import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.reconciliation import (
    filter_active_semantic_records,
    plan_semantic_write,
)
from agent.memory.models import ExtractionResult, MemoryWrite, SemanticFact
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.memory.write_policy import decide_semantic_candidate
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id

logger = logging.getLogger(__name__)


_EARLY_PATTERN_LANGUAGE = (
    "it keeps happening",
    "every new task makes me feel like",
    "every task makes me feel like",
    "i'm about to fail",
    "im about to fail",
)

_NEGATIVE_SELF_BELIEF_LANGUAGE = (
    "i always assume",
    "everyone will see i'm",
    "everyone will see im",
    "everyone will think i'm",
    "everyone will think im",
    "one mistake means",
    "i'm incompetent",
    "im incompetent",
    "i'm a failure",
    "im a failure",
)

_DURABILITY_MARKERS = (
    "for years",
    "for a long time",
    "i always",
    "i usually",
    "every time",
    "whenever",
    "ever since",
)


def _should_skip_early_emerging_pattern(message: str, turn_index: int) -> bool:
    """Return whether an early-turn emerging pattern should skip extraction.

    This is a narrow product guard for fresh, in-session interpretations that
    are reflective-worthy but not yet durable enough for long-term semantic
    memory. Prompt guidance should catch most of these cases; this helper adds
    a deterministic backstop for the highest-friction failure mode.

    Negative global self-beliefs get extra protection in early turns: even
    when phrased with durability markers like "I always", they are often
    better treated as fresh therapeutic material to explore first rather than
    stable semantic memory to persist immediately.
    """

    lowered = message.lower()
    if turn_index > 1:
        return False

    if any(marker in lowered for marker in _NEGATIVE_SELF_BELIEF_LANGUAGE):
        return True

    if not any(marker in lowered for marker in _EARLY_PATTERN_LANGUAGE):
        return False
    if any(marker in lowered for marker in _DURABILITY_MARKERS):
        return False
    return True


def _memory_write_to_semantic_fact(
    write: MemoryWrite,
    *,
    write_timing: str = "immediate",
    write_reason: str = "",
    policy_version: str = "phase1_v1",
) -> SemanticFact:
    """Convert an LLM-produced :class:`MemoryWrite` to a stored :class:`SemanticFact`.

    The MemoryWrite has the 7 fields the extractor produces; SemanticFact
    adds the store metadata (id, timestamps, dormant/superseded markers,
    visibility flag). This helper generates a fresh ID and timestamps for
    a new record.

    Uses uuid4 for the id. UUIDv7 (time-sortable) is the schema-preferred
    ID type but Python's stdlib ``uuid`` module doesn't add ``uuid7``
    until Python 3.14. When the runtime upgrades, switch this line to
    ``str(uuid.uuid7())`` — nothing else needs to change because the
    schema already declares the id as ``str``.
    """

    now = _iso_now()
    return SemanticFact(
        id=str(uuid4()),
        category=write.category,
        subject=write.subject,
        predicate=write.predicate,
        object=write.object,
        evidence_quote=write.evidence_quote,
        confidence=write.confidence,
        source_session_id=write.source_session_id,
        source_turn_index=write.source_turn_index,
        created_at=now,
        last_referenced_at=now,
        dormant_at=None,
        superseded_by=None,
        user_visible=True,
        write_timing=write_timing,  # type: ignore[arg-type]
        write_reason=write_reason,
        policy_version=policy_version,
    )


async def _fetch_existing_user_records(
    store: MemoryStore,
    *,
    owner_id: str,
) -> list[Any]:
    """Fetch all semantic-namespace records for a user.

    Returns a list of :class:`StoreRecord` objects (typed as ``list[Any]``
    at the boundary so the caller doesn't have to import StoreRecord
    from the store module). The typing penalty is small and the
    looseness keeps the node's import surface minimal.

    With the in-memory store and v0.3's conservative extraction, the
    record count per user is small (tens to low hundreds at most). The
    ``limit=1000`` cap is defensive; we don't expect to hit it. When
    v0.8 adds SQLite backing, this function can become a narrower query
    (e.g. "facts with similar triples") instead of fetching everything.
    """

    namespace = (owner_id, "semantic")
    records = await store.asearch(namespace, query=None, limit=1000)
    return filter_active_semantic_records(records)


async def _write_new_fact(
    store: MemoryStore,
    *,
    owner_id: str,
    fact: SemanticFact,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> None:
    """Persist a freshly-extracted SemanticFact to the store.

    Separated from the main node body so the error-handling scope is
    tight around the single store call. A failure here is logged but
    never raised to the caller.

    v0.8.1: accepts optional ``embedding`` and ``embedding_model``
    kwargs matching the :class:`MemoryStore.aput` extension. When
    provided, the embedding is stored alongside the record so the
    read path can use it for hybrid retrieval via
    :meth:`MemoryStore.asearch_similar`. When not provided (guest
    mode, null embedding provider, or embedding computation failed),
    the record is still written via the token-recall path only —
    graceful degradation preserved.
    """

    namespace = (owner_id, "semantic")
    await store.aput(
        namespace,
        key=fact.id,
        value=fact.model_dump(mode="json"),
        embedding=embedding,
        embedding_model=embedding_model,
    )


async def _bump_last_referenced_at(
    store: MemoryStore,
    *,
    matched_record: Any,
) -> None:
    """Update the matched record's ``last_referenced_at`` to now.

    The store is a key-value layer, so "update" means re-putting the
    record with the same namespace + key and a modified value. Only
    the timestamp changes; all other fields are preserved.
    """

    updated_value = dict(matched_record.value)
    updated_value["last_referenced_at"] = _iso_now()
    await store.aput(
        matched_record.namespace,
        key=matched_record.key,
        value=updated_value,
    )


async def _mark_fact_superseded(
    store: MemoryStore,
    *,
    matched_record: Any,
    replacement_fact_id: str,
) -> None:
    """Mark one stored semantic fact as superseded by a newer fact."""

    updated_value = dict(matched_record.value)
    now = _iso_now()
    updated_value["last_referenced_at"] = now
    updated_value["dormant_at"] = now
    updated_value["superseded_by"] = replacement_fact_id
    await store.aput(
        matched_record.namespace,
        key=matched_record.key,
        value=updated_value,
        embedding=getattr(matched_record, "embedding", None),
        embedding_model=getattr(matched_record, "embedding_model", None),
    )


async def run_extract_semantic_facts_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Extract and persist zero or more semantic facts from the current turn.

    Runs after the response generation node on both the crisis and
    therapeutic branches. Returns a state delta whose only meaningful
    content is the per-turn diagnostics entry (timing + write
    counts). The actual memory writes are side effects on the store.

    Silently skips when the runtime lacks an LLM client or is in
    incognito mode. All other failure modes (LLM errors, schema
    validation errors, store write errors) are logged at WARNING level
    with ``exc_info=True`` but never propagate.
    """

    # v0.8 observability: time the full extraction call and report
    # write counts via the diagnostics dict. Every return path below
    # composes its delta via ``_diagnostics_delta`` so skipped turns
    # are still distinguishable from turns where the node never ran.
    start = time.monotonic()

    def _diagnostics_delta(
        *,
        semantic_writes: int = 0,
        semantic_bumps: int = 0,
        semantic_candidates: int = 0,
        semantic_commit_now_candidates: int = 0,
        semantic_session_end_holds: int = 0,
        semantic_repeat_required: int = 0,
        semantic_policy_drops: int = 0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Return a state delta carrying just the extractor's diagnostics."""

        return {
            "diagnostics": {
                "extract_facts_ms": round((time.monotonic() - start) * 1000, 2),
                "semantic_writes": semantic_writes,
                "semantic_bumps": semantic_bumps,
                "semantic_candidates": semantic_candidates,
                "semantic_commit_now_candidates": semantic_commit_now_candidates,
                "semantic_session_end_holds": semantic_session_end_holds,
                "semantic_repeat_required": semantic_repeat_required,
                "semantic_policy_drops": semantic_policy_drops,
                "extract_facts_reason": reason,
            }
        }

    # ── Early exits ─────────────────────────────────────────────────────

    # v0.9: crisis gate first — fastest check, highest priority. Skip
    # extraction on crisis turns to avoid delaying crisis response
    # delivery. The DoneEvent is only yielded after END, so extractor
    # LLM calls (~3-5s) would block the user from seeing the crisis
    # response. Crisis responses are templated + PFA overlay, not rich
    # content worth extracting facts from.
    route = state.get("routing", {}).get("route")
    if route == "crisis":
        logger.debug(
            "extract_semantic_facts_node: crisis path; skipping to avoid "
            "delaying crisis response delivery"
        )
        return _diagnostics_delta(reason="skipped: crisis_path")

    llm_client = runtime.context.llm_client
    memory_mode = runtime.context.memory_mode

    if llm_client is None:
        logger.debug("extract_semantic_facts_node: no llm_client; skipping")
        return _diagnostics_delta(reason="skipped: no llm_client")
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_semantic_facts_node: incognito mode; skipping (no writes to "
            "persistent memory in incognito)"
        )
        return _diagnostics_delta(reason="skipped: incognito")

    store = runtime.context.memory_store
    embedding_provider = runtime.context.embedding_provider
    session_buffer = runtime.context.session_memory_buffer
    owner_id = resolve_owner_id(state)

    # v0.8.2: pre-extractor small-talk gate.
    from agent.memory.small_talk_gate import is_small_talk

    if is_small_talk(state["message"]):
        logger.debug(
            "extract_semantic_facts_node: small-talk gate triggered; skipping "
            "LLM call for message %r",
            state["message"][:40],
        )
        return _diagnostics_delta(reason="skipped: small_talk_gate")

    # Turn index is 0-based; state["progress"]["turn_count"] counts user
    # turns including the current one (1-based), so we subtract 1.
    progress = state.get("progress", {})
    turn_count = int(progress.get("turn_count", 1))
    turn_index = max(0, turn_count - 1)

    if _should_skip_early_emerging_pattern(state["message"], turn_index):
        logger.debug(
            "extract_semantic_facts_node: early emerging-pattern guard triggered; "
            "skipping extraction for message %r",
            state["message"][:80],
        )
        return _diagnostics_delta(reason="skipped: early_emerging_pattern")

    # ── LLM structured-output extraction ────────────────────────────────
    try:
        extraction: ExtractionResult = await llm_client.generate_structured(
            prompt=build_extraction_user_prompt(state, turn_index=turn_index),
            response_schema=ExtractionResult,
            system_instruction=build_extraction_system_prompt(),
        )
    except Exception:
        logger.warning(
            "extract_semantic_facts_node: LLM structured-output call failed; "
            "skipping extraction for this turn.",
            exc_info=True,
        )
        return _diagnostics_delta(reason="skipped: llm error")

    # Log the extraction reason regardless of whether facts were produced —
    # it's a free observability signal for prompt tuning. INFO level (not
    # DEBUG) so dogfood sessions can see extraction decisions in real time
    # without having to rewire logging. The conservative extractor rejects
    # most turns, and knowing *why* it rejected a turn is the fastest way
    # to catch prompt drift.
    logger.info(
        "extract_semantic_facts_node: %d facts, reason=%r",
        len(extraction.facts),
        extraction.reason,
    )

    if not extraction.facts:
        return _diagnostics_delta(reason=extraction.reason)

    # ── Build candidates and apply deterministic write policy ─────────
    immediate_candidates: list[tuple[Any, Any]] = []
    session_end_holds = 0
    repeat_required = 0
    policy_drops = 0

    for write in extraction.facts:
        candidate = build_semantic_candidate(write, message=state["message"])
        decision = decide_semantic_candidate(candidate)

        if decision.action == "commit_now":
            immediate_candidates.append((candidate, decision))
        elif decision.action == "commit_at_session_end":
            session_end_holds += 1
            if session_buffer is not None:
                session_buffer.semantic_candidates.append(candidate)
        elif decision.action == "require_repetition":
            repeat_required += 1
            if session_buffer is not None:
                session_buffer.semantic_candidates.append(candidate)
        else:
            policy_drops += 1

    if not immediate_candidates:
        logger.info(
            "extract_semantic_facts_node: policy held all %d facts "
            "(session_end=%d, repetition=%d, dropped=%d)",
            len(extraction.facts),
            session_end_holds,
            repeat_required,
            policy_drops,
        )
        return _diagnostics_delta(
            semantic_candidates=len(extraction.facts),
            semantic_session_end_holds=session_end_holds,
            semantic_repeat_required=repeat_required,
            semantic_policy_drops=policy_drops,
            reason=extraction.reason,
        )

    # ── Fetch existing records once for dedup ──────────────────────────
    try:
        existing_records = await _fetch_existing_user_records(store, owner_id=owner_id)
    except Exception:
        logger.warning(
            "extract_semantic_facts_node: failed to fetch existing records for "
            "dedup; skipping all candidates for this turn.",
            exc_info=True,
        )
        return _diagnostics_delta(
            semantic_candidates=len(extraction.facts),
            semantic_commit_now_candidates=len(immediate_candidates),
            semantic_session_end_holds=session_end_holds,
            semantic_repeat_required=repeat_required,
            semantic_policy_drops=policy_drops,
            reason="skipped: dedup fetch failed",
        )

    # ── Batch-compute embeddings for all candidates (v0.8.1) ───────────
    #
    # Embedding computation happens ONCE per turn in a single batch
    # call, even though dedup may later reject some candidates. The
    # alternative (compute after dedup, per surviving candidate) would
    # add per-candidate network round-trips in the common case of
    # 1-2 surviving facts — a single batch is simpler and cheaper
    # even with the wasted work on dedup rejects.
    #
    # We embed the evidence quote as the canonical representation of
    # the fact. This matches what the store's haystack looks like at
    # retrieval time: the retrieval query is a user message, and the
    # most retrieval-relevant field of a stored fact is the evidence
    # quote (the user's own words). Embedding the triple instead
    # (subject/predicate/object) would make the embedding less
    # comparable to natural-language queries. Dogfood can revisit
    # this if the eval harness shows the quote-only choice is wrong.
    #
    # The ``task_type="RETRIEVAL_DOCUMENT"`` hint tells Gemini this
    # is a document-side embedding (will be matched against query-
    # side embeddings later in load_memory_node). Matters for
    # asymmetric retrieval models; ignored by providers that don't
    # support task types.
    candidate_embeddings: list[list[float] | None] = [None] * len(immediate_candidates)
    embedding_model_name: str | None = None
    if embedding_provider is not None:
        try:
            quotes = [
                candidate.payload.evidence_quote
                for candidate, _ in immediate_candidates
            ]
            candidate_embeddings = await embedding_provider.aembed(
                quotes,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embedding_model_name = embedding_provider.model_name
            # NullEmbeddingProvider returns all-None; treat that as
            # "no embeddings available" so downstream writes don't
            # attach a bogus model name to NULL embeddings.
            if all(e is None for e in candidate_embeddings):
                embedding_model_name = None
        except Exception:
            # Embedding failures should never poison the write path —
            # fall back to all-None which means "write without
            # embeddings, token-recall still works." Logged so dogfood
            # can observe embedding-provider health without having to
            # tail a different log stream.
            logger.warning(
                "extract_semantic_facts_node: embedding batch failed; "
                "writing facts without embeddings for this turn.",
                exc_info=True,
            )
            candidate_embeddings = [None] * len(immediate_candidates)
            embedding_model_name = None

    # ── Per-candidate dedup + write ─────────────────────────────────────
    written = 0
    bumped = 0
    for candidate_index, (candidate, decision) in enumerate(immediate_candidates):
        write = candidate.payload
        try:
            matched = find_near_duplicate(write, existing_records)
        except Exception:
            logger.warning(
                "extract_semantic_facts_node: dedup check raised for candidate "
                "%r; skipping this candidate.",
                write.evidence_quote[:40],
                exc_info=True,
            )
            continue

        if matched is not None:
            # Dedup hit: bump last_referenced_at instead of writing a new row.
            try:
                await _bump_last_referenced_at(store, matched_record=matched)
                bumped += 1
            except Exception:
                logger.warning(
                    "extract_semantic_facts_node: failed to bump last_referenced_at "
                    "on matched record %r; continuing with other candidates.",
                    matched.key,
                    exc_info=True,
                )
            continue

        # No duplicate: materialize as SemanticFact and reconcile.
        try:
            fact = _memory_write_to_semantic_fact(
                write,
                write_timing="immediate",
                write_reason=decision.reason,
                policy_version=decision.policy_version,
            )
            reconciliation = plan_semantic_write(fact, existing_records)
            if reconciliation.bump_record is not None:
                await _bump_last_referenced_at(
                    store,
                    matched_record=reconciliation.bump_record,
                )
                bumped += 1
                continue
            # v0.8.1: pair the fact with the embedding computed in
            # the batch above. ``candidate_embeddings[i]`` is None
            # when no embedding provider is configured, when the
            # provider returned None for this specific candidate,
            # or when the batch call failed — all three cases are
            # handled the same way at the store layer (NULL embedding,
            # token-recall only).
            this_embedding = candidate_embeddings[candidate_index]
            this_model = embedding_model_name if this_embedding is not None else None
            await _write_new_fact(
                store,
                owner_id=owner_id,
                fact=fact,
                embedding=this_embedding,
                embedding_model=this_model,
            )
            written += 1
            # Include the newly-written fact in the existing_records view so
            # subsequent candidates in the same extraction batch can dedup
            # against it (rare but possible when the LLM returns near-duplicates
            # in a single call).
            from agent.memory.store import StoreRecord

            existing_records.append(
                StoreRecord(
                    namespace=(owner_id, "semantic"),
                    key=fact.id,
                    value=fact.model_dump(mode="json"),
                    embedding=this_embedding,
                    embedding_model=this_model,
                )
            )
            for superseded_record in reconciliation.supersede_records:
                try:
                    await _mark_fact_superseded(
                        store,
                        matched_record=superseded_record,
                        replacement_fact_id=fact.id,
                    )
                    superseded_record.value["last_referenced_at"] = fact.created_at
                    superseded_record.value["dormant_at"] = fact.created_at
                    superseded_record.value["superseded_by"] = fact.id
                except Exception:
                    logger.warning(
                        "extract_semantic_facts_node: failed to mark stale fact %r "
                        "as superseded after writing replacement.",
                        superseded_record.key,
                        exc_info=True,
                    )
        except Exception:
            logger.warning(
                "extract_semantic_facts_node: failed to write candidate %r; "
                "continuing with other candidates.",
                write.evidence_quote[:40],
                exc_info=True,
            )

    logger.info(
        "extract_semantic_facts_node: turn complete — %d written, %d bumped, "
        "%d immediate, %d held_for_session, %d repeat_required, %d dropped",
        written,
        bumped,
        len(immediate_candidates),
        session_end_holds,
        repeat_required,
        policy_drops,
    )
    return _diagnostics_delta(
        semantic_writes=written,
        semantic_bumps=bumped,
        semantic_candidates=len(extraction.facts),
        semantic_commit_now_candidates=len(immediate_candidates),
        semantic_session_end_holds=session_end_holds,
        semantic_repeat_required=repeat_required,
        semantic_policy_drops=policy_drops,
        reason=extraction.reason,
    )
