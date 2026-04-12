"""Semantic fact extraction node — real implementation (v0.3).

Runs after the response generation node on every turn and extracts
zero or more memory-worthy facts from the current user message. Each
extracted fact becomes a :class:`SemanticFact` record in the unified
memory store, keyed by a UUID.

v0.3 design rules (locked via the v0.3 scoping discussion):

1. **Conservative extraction.** The LLM is told via system prompt that
   most turns should produce zero facts. Small talk, transient moods,
   speculation, and ambiguous statements are all filtered out.

2. **Silent skip on incognito or no LLM.** The node is always registered
   in the parent graph, but it no-ops when either (a) the memory mode
   is INCOGNITO, or (b) no LLM client is available. Both are legitimate
   v0.3 states that shouldn't trigger any extraction-path machinery.

3. **Hot-path dedup.** Every candidate fact is checked against existing
   store records via :func:`find_near_duplicate`. Duplicates bump the
   matched record's ``last_referenced_at`` timestamp instead of writing
   a new row. Dedup uses token-set Jaccard similarity on evidence
   quotes; the vector-similarity variant lands in v0.8.

4. **Failures degrade silently.** LLM errors, schema validation errors,
   and store write errors are all logged at WARNING level but never
   propagate. The extraction node is a side-effect node — a failure
   here must not fail the parent turn.

5. **Always return an empty state delta.** The write is a side effect;
   state isn't modified. Returning ``{}`` is the canonical "I touched
   nothing" signal for LangGraph delta-return nodes.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph.runtime import Runtime

from agent.memory.dedup import find_near_duplicate
from agent.memory.extraction_prompts import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from agent.memory.models import ExtractionResult, MemoryWrite, SemanticFact
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    """Return the current UTC time in ISO-8601 format with 'Z' suffix."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _memory_write_to_semantic_fact(write: MemoryWrite) -> SemanticFact:
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
    return await store.asearch(namespace, query=None, limit=1000)


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
        reason: str = "",
    ) -> dict[str, Any]:
        """Return a state delta carrying just the extractor's diagnostics."""

        return {
            "diagnostics": {
                **state.get("diagnostics", {}),
                "extract_facts_ms": round((time.monotonic() - start) * 1000, 2),
                "semantic_writes": semantic_writes,
                "semantic_bumps": semantic_bumps,
                "extract_facts_reason": reason,
            }
        }

    # ── Early exits ─────────────────────────────────────────────────────
    llm_client = runtime.context.get("llm_client")
    memory_mode = runtime.context.get("memory_mode", MemoryMode.INCOGNITO)

    if llm_client is None:
        logger.debug("extract_semantic_facts_node: no llm_client; skipping")
        return _diagnostics_delta(reason="skipped: no llm_client")
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_semantic_facts_node: incognito mode; skipping (no writes to "
            "persistent memory in incognito)"
        )
        return _diagnostics_delta(reason="skipped: incognito")

    store = runtime.context["memory_store"]
    # v0.8.1: embedding provider is optional in the context
    # (NotRequired in WorkflowContext). When absent or when the
    # runtime wires a NullEmbeddingProvider, the extractor writes
    # records without embeddings and the store's hybrid retrieval
    # degrades to token-recall for those rows. No error path —
    # embeddings are a quality boost, not a correctness requirement.
    embedding_provider = runtime.context.get("embedding_provider")
    owner_id = state.get("user_id") or state.get("session_id") or "local-default"

    # v0.8.2: pre-extractor small-talk gate. Skip the LLM call
    # entirely when the message is unambiguously small talk (short +
    # all tokens in the small-talk vocabulary). Saves ~3-5s per turn
    # on greetings and acknowledgments. Conservative by design —
    # false negatives waste one LLM call, false positives silently
    # lose memory. See ``agent/memory/small_talk_gate.py`` for the
    # heuristic rationale.
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

    # ── LLM structured-output extraction ────────────────────────────────
    try:
        extraction: ExtractionResult = await llm_client.generate_structured(
            prompt=build_extraction_user_prompt(state, turn_index=turn_index),
            response_schema=ExtractionResult,
            system_instruction=build_extraction_system_prompt(),
            temperature=0,
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

    # ── Fetch existing records once for dedup ──────────────────────────
    try:
        existing_records = await _fetch_existing_user_records(store, owner_id=owner_id)
    except Exception:
        logger.warning(
            "extract_semantic_facts_node: failed to fetch existing records for "
            "dedup; skipping all candidates for this turn.",
            exc_info=True,
        )
        return _diagnostics_delta(reason="skipped: dedup fetch failed")

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
    candidate_embeddings: list[list[float] | None] = [None] * len(extraction.facts)
    embedding_model_name: str | None = None
    if embedding_provider is not None:
        try:
            quotes = [c.evidence_quote for c in extraction.facts]
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
            candidate_embeddings = [None] * len(extraction.facts)
            embedding_model_name = None

    # ── Per-candidate dedup + write ─────────────────────────────────────
    written = 0
    bumped = 0
    for candidate_index, candidate in enumerate(extraction.facts):
        try:
            matched = find_near_duplicate(candidate, existing_records)
        except Exception:
            logger.warning(
                "extract_semantic_facts_node: dedup check raised for candidate "
                "%r; skipping this candidate.",
                candidate.evidence_quote[:40],
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

        # No duplicate: materialize as SemanticFact and write.
        try:
            fact = _memory_write_to_semantic_fact(candidate)
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
        except Exception:
            logger.warning(
                "extract_semantic_facts_node: failed to write candidate %r; "
                "continuing with other candidates.",
                candidate.evidence_quote[:40],
                exc_info=True,
            )

    logger.info(
        "extract_semantic_facts_node: turn complete — %d written, %d bumped, "
        "%d total candidates",
        written,
        bumped,
        len(extraction.facts),
    )
    return _diagnostics_delta(
        semantic_writes=written,
        semantic_bumps=bumped,
        reason=extraction.reason,
    )
