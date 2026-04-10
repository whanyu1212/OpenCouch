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
from agent.memory.store import OpenCouchMemoryStore
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
    store: OpenCouchMemoryStore,
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
    store: OpenCouchMemoryStore,
    *,
    owner_id: str,
    fact: SemanticFact,
) -> None:
    """Persist a freshly-extracted SemanticFact to the store.

    Separated from the main node body so the error-handling scope is
    tight around the single store call. A failure here is logged but
    never raised to the caller.
    """

    namespace = (owner_id, "semantic")
    await store.aput(
        namespace,
        key=fact.id,
        value=fact.model_dump(mode="json"),
    )


async def _bump_last_referenced_at(
    store: OpenCouchMemoryStore,
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
    therapeutic branches. Returns an empty state delta; the write is
    a side effect on the memory store.

    Silently skips when the runtime lacks an LLM client or is in
    incognito mode. All other failure modes (LLM errors, schema
    validation errors, store write errors) are logged at WARNING level
    with ``exc_info=True`` but never propagate.
    """

    # ── Early exits ─────────────────────────────────────────────────────
    llm_client = runtime.context.get("llm_client")
    memory_mode = runtime.context.get("memory_mode", MemoryMode.INCOGNITO)

    if llm_client is None:
        logger.debug("extract_semantic_facts_node: no llm_client; skipping")
        return {}
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_semantic_facts_node: incognito mode; skipping (no writes to "
            "persistent memory in incognito)"
        )
        return {}

    store = runtime.context["memory_store"]
    owner_id = state.get("user_id") or state.get("session_id") or "local-default"

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
        return {}

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
        return {}

    # ── Fetch existing records once for dedup ──────────────────────────
    try:
        existing_records = await _fetch_existing_user_records(store, owner_id=owner_id)
    except Exception:
        logger.warning(
            "extract_semantic_facts_node: failed to fetch existing records for "
            "dedup; skipping all candidates for this turn.",
            exc_info=True,
        )
        return {}

    # ── Per-candidate dedup + write ─────────────────────────────────────
    written = 0
    bumped = 0
    for candidate in extraction.facts:
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
            await _write_new_fact(store, owner_id=owner_id, fact=fact)
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
    return {}
