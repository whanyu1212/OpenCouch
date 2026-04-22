"""Load-memory node for the OpenCouch graph.

Runs on every turn to retrieve semantic and episodic working memory for
the current user message, plus procedural state for prompt shaping.

Key behaviors:
- semantic and episodic entries are merged into ``working_memory``
- procedural rules and the recall toggle stay on ``state["memory"]``
- first-turn sessions prepend the latest episodic summary as catch-up
- incognito mode skips retrieval and returns empty memory state
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal

from langgraph.runtime import Runtime

from agent.memory.modes import MemoryMode
from agent.memory.procedural import aget_procedural_profile
from agent.memory.reconciliation import is_active_semantic_record_value
from agent.memory.store import MemoryStore
from agent.memory.text_tokens import tokenize_meaningful
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id
from agent.working_memory import (
    WorkingMemoryEntry,
    make_episodic_working_memory_entry,
    make_semantic_working_memory_entry,
)

if TYPE_CHECKING:
    from agent.memory.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

SEMANTIC_SEARCH_LIMIT = 20
SEMANTIC_WORKING_MEMORY_LIMIT = 5
EPISODIC_SEARCH_LIMIT = 2
EPISODIC_MAX_AGE_DAYS = 30

RetrievalPath = Literal[
    "hybrid_rrf",
    "token_recall",
    "token_recall_after_embed_error",
    "skipped_empty_store",
]


async def _retrieve_semantic_working_memory(
    store: MemoryStore,
    *,
    owner_id: str,
    query: str,
    query_embedding: list[float] | None,
    embedding_model: str | None,
) -> list[WorkingMemoryEntry]:
    """Fetch semantic working-memory entries for a user.

    Args:
        store (MemoryStore): Memory store to query.
        owner_id (str): Owner id whose semantic namespace should be searched.
        query (str): Query text for lexical retrieval.
        query_embedding (list[float] | None): Optional dense query embedding.
        embedding_model (str | None): Optional query embedding model identifier.

    Returns:
        list[WorkingMemoryEntry]: Top active semantic entries for working memory.
    """

    namespace = (owner_id, "semantic")
    records = await store.asearch_similar(
        namespace,
        query_text=query,
        query_embedding=query_embedding,
        embedding_model=embedding_model,
        limit=SEMANTIC_SEARCH_LIMIT,
        record_filter=lambda record: is_active_semantic_record_value(record.value),
    )
    entries: list[WorkingMemoryEntry] = []
    for record in records:
        entry = _semantic_entry_from_record(record.value)
        if entry is not None:
            entries.append(entry)
        if len(entries) >= SEMANTIC_WORKING_MEMORY_LIMIT:
            break
    return entries


def _semantic_entry_from_record(
    record_value: dict[str, Any],
) -> WorkingMemoryEntry | None:
    """Convert a stored semantic fact into a working-memory entry.

    Args:
        record_value (dict[str, Any]): Serialized semantic record payload.

    Returns:
        WorkingMemoryEntry | None: Structured semantic working-memory entry, or ``None``.
    """

    quote = record_value.get("evidence_quote")
    if not quote:
        return None

    subject_ref = record_value.get("subject") or {}
    object_ref = record_value.get("object") or {}
    return make_semantic_working_memory_entry(
        evidence_quote=quote,
        category=record_value.get("category", ""),
        subject=subject_ref.get("identifier", "")
        if isinstance(subject_ref, dict)
        else str(subject_ref),
        predicate=record_value.get("predicate", ""),
        object=object_ref.get("identifier", "")
        if isinstance(object_ref, dict)
        else str(object_ref),
    )


def _episodic_entry_from_record(
    record_value: dict[str, Any],
    *,
    is_catch_up: bool,
) -> WorkingMemoryEntry | None:
    """Convert a stored session arc into a working-memory entry.

    Args:
        record_value (dict[str, Any]): Serialized episodic record payload.
        is_catch_up (bool): Whether the entry should be marked as catch-up context.

    Returns:
        WorkingMemoryEntry | None: Structured episodic working-memory entry, or ``None``.
    """

    summary = record_value.get("summary")
    if not summary:
        return None
    return make_episodic_working_memory_entry(
        summary=summary,
        primary_themes=record_value.get("primary_themes") or [],
        is_catch_up=is_catch_up,
        approach_used=record_value.get("approach_used"),
        approach_context=record_value.get("approach_context"),
    )


def _episodic_entry_identity(entry: WorkingMemoryEntry) -> tuple[str, tuple[str, ...]]:
    """Return the dedup identity for an episodic working-memory entry.

    Args:
        entry (WorkingMemoryEntry): Episodic working-memory entry to fingerprint.

    Returns:
        tuple[str, tuple[str, ...]]: Identity used to dedupe episodic entries.
    """

    if entry.get("type") != "episodic":
        return "", ()
    return (
        entry.get("summary", ""),
        tuple(entry.get("primary_themes") or []),
    )


async def _retrieve_episodic_working_memory(
    store: MemoryStore,
    *,
    owner_id: str,
    query: str,
    query_embedding: list[float] | None,
    embedding_model: str | None,
    is_first_turn: bool,
) -> list[WorkingMemoryEntry]:
    """Fetch episodic working-memory entries for a user.

    Args:
        store (MemoryStore): Memory store to query.
        owner_id (str): Owner id whose episodic namespace should be searched.
        query (str): Query text for lexical retrieval.
        query_embedding (list[float] | None): Optional dense query embedding.
        embedding_model (str | None): Optional query embedding model identifier.
        is_first_turn (bool): Whether this is the first turn of the current session.

    Returns:
        list[WorkingMemoryEntry]: Episodic entries for catch-up and query recall.
    """

    namespace = (owner_id, "episodic")
    entries: list[WorkingMemoryEntry] = []
    seen_identities: set[tuple[str, tuple[str, ...]]] = set()

    if is_first_turn:
        latest = await store.alatest(namespace)
        if latest is not None:
            catch_up = _episodic_entry_from_record(
                latest.value,
                is_catch_up=True,
            )
            if catch_up is not None:
                entries.append(catch_up)
                seen_identities.add(_episodic_entry_identity(catch_up))

    query_records = await store.asearch_similar(
        namespace,
        query_text=query,
        query_embedding=query_embedding,
        embedding_model=embedding_model,
        limit=EPISODIC_SEARCH_LIMIT,
        max_age_days=EPISODIC_MAX_AGE_DAYS,
    )
    for record in query_records:
        entry = _episodic_entry_from_record(
            record.value,
            is_catch_up=False,
        )
        if entry is None:
            continue
        identity = _episodic_entry_identity(entry)
        if identity in seen_identities:
            continue
        entries.append(entry)
        seen_identities.add(identity)

    return entries


async def _retrieve_procedural_state(
    store: MemoryStore,
    *,
    owner_id: str,
) -> tuple[list[str], bool]:
    """Load procedural rules and recall state for a user.

    Args:
        store (MemoryStore): Memory store to query.
        owner_id (str): Owner id whose procedural profile should be loaded.

    Returns:
        tuple[list[str], bool]: Rule texts plus the proactive-recall toggle.
    """

    profile = await aget_procedural_profile(store, user_id=owner_id)
    rule_texts = [rule.rule for rule in profile.rules]
    return rule_texts, profile.proactive_recall_enabled


async def _compute_query_embedding(
    embedding_provider: "EmbeddingProvider | None",
    query: str,
) -> tuple[list[float] | None, str | None, RetrievalPath]:
    """Compute the query embedding and retrieval-path metadata.

    Args:
        embedding_provider (EmbeddingProvider | None): Optional embedding provider.
        query (str): Query text to embed.

    Returns:
        tuple[list[float] | None, str | None, RetrievalPath]: Query embedding, model, and retrieval path.
    """

    if embedding_provider is None:
        return None, None, "token_recall"
    try:
        embeddings_out = await embedding_provider.aembed(
            [query],
            task_type="RETRIEVAL_QUERY",
        )
        if embeddings_out and embeddings_out[0] is not None:
            return embeddings_out[0], embedding_provider.model_name, "hybrid_rrf"
        return None, None, "token_recall"
    except Exception:
        logger.warning(
            "load_memory_node: embedding call failed; falling back to "
            "token-recall only for this turn.",
            exc_info=True,
        )
        return None, None, "token_recall_after_embed_error"


def _build_load_memory_summary(
    *,
    semantic_hits: int,
    semantic_store_size: int,
    episodic_hits: int,
    episodic_store_size: int,
    procedural_count: int,
    proactive_recall_enabled: bool,
    retrieval_path: RetrievalPath,
    meaningful_query_token_count: int,
) -> str:
    """Build the human-readable load-memory summary string.

    Args:
        semantic_hits (int): Number of semantic entries retrieved.
        semantic_store_size (int): Total semantic record count.
        episodic_hits (int): Number of episodic entries retrieved.
        episodic_store_size (int): Total episodic record count.
        procedural_count (int): Number of procedural rules loaded.
        proactive_recall_enabled (bool): Current recall-toggle state.
        retrieval_path (RetrievalPath): Retrieval path used for the turn.
        meaningful_query_token_count (int): Count of meaningful query tokens.

    Returns:
        str: Summary string for memory observability.
    """

    return (
        f"Retrieved {semantic_hits} of {semantic_store_size} semantic record(s) + "
        f"{episodic_hits} of {episodic_store_size} episodic record(s), "
        f"{procedural_count} procedural rule(s), "
        f"recall={'on' if proactive_recall_enabled else 'off'} "
        f"path={retrieval_path} "
        f"(query had {meaningful_query_token_count} meaningful token(s))."
    )


def _build_load_memory_diagnostics(
    *,
    retrieval_duration_ms: float,
    semantic_hits: int,
    semantic_store_size: int,
    episodic_hits: int,
    episodic_store_size: int,
    procedural_count: int,
    proactive_recall_enabled: bool,
    retrieval_path: RetrievalPath,
) -> dict[str, Any]:
    """Build the diagnostics payload for load-memory observability.

    Args:
        retrieval_duration_ms (float): Retrieval duration in milliseconds.
        semantic_hits (int): Number of semantic entries retrieved.
        semantic_store_size (int): Total semantic record count.
        episodic_hits (int): Number of episodic entries retrieved.
        episodic_store_size (int): Total episodic record count.
        procedural_count (int): Number of procedural rules loaded.
        proactive_recall_enabled (bool): Current recall-toggle state.
        retrieval_path (RetrievalPath): Retrieval path used for the turn.

    Returns:
        dict[str, Any]: Diagnostics payload for CLI and observability.
    """

    return {
        "load_memory_ms": round(retrieval_duration_ms, 2),
        "semantic_hits": semantic_hits,
        "semantic_store_size": semantic_store_size,
        "episodic_hits": episodic_hits,
        "episodic_store_size": episodic_store_size,
        "procedural_count": procedural_count,
        "proactive_recall": proactive_recall_enabled,
        "retrieval_path": retrieval_path,
    }


async def run_load_memory_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Retrieve semantic, episodic, and procedural memory for the turn.

    Args:
        state (AgentState): Current workflow state.
        runtime (Runtime[WorkflowContext]): LangGraph runtime with memory dependencies.

    Returns:
        dict[str, Any]: State delta with working memory, memory metadata, and diagnostics.
    """

    memory_store = runtime.context.memory_store
    memory_mode = runtime.context.memory_mode
    embedding_provider: EmbeddingProvider | None = runtime.context.embedding_provider
    is_guest_mode = memory_mode == MemoryMode.INCOGNITO

    if is_guest_mode:
        return {
            "working_memory": [],
            "memory": {
                **state.get("memory", {}),
                "summary": "Guest session without long-term memory.",
                "procedural_rules": [],
                "proactive_recall_enabled": False,
            },
        }

    owner_id = resolve_owner_id(state)
    episodic_ns = (owner_id, "episodic")
    query = state["message"]
    meaningful_query_tokens = tokenize_meaningful(query)

    # Determine whether this is the first turn of the current session.
    # ``build_initial_state`` appends the current user message to the
    # transcript, so a single-entry transcript means "no prior turns
    # have run in this session yet" — the trigger for catch-up injection.
    transcript = state.get("transcript", [])
    is_first_turn = len(transcript) == 1

    retrieval_start = time.monotonic()

    (
        semantic_store_size,
        episodic_store_size,
        (procedural_rules, proactive_recall_enabled),
    ) = await asyncio.gather(
        memory_store.arecord_count((owner_id, "semantic")),
        memory_store.arecord_count(episodic_ns),
        _retrieve_procedural_state(memory_store, owner_id=owner_id),
    )

    has_searchable_memory = semantic_store_size > 0 or episodic_store_size > 0

    if has_searchable_memory:
        (
            query_embedding,
            query_embedding_model,
            retrieval_path,
        ) = await _compute_query_embedding(embedding_provider, query)

        episodic_entries, semantic_entries = await asyncio.gather(
            _retrieve_episodic_working_memory(
                memory_store,
                owner_id=owner_id,
                query=query,
                query_embedding=query_embedding,
                embedding_model=query_embedding_model,
                is_first_turn=is_first_turn,
            ),
            _retrieve_semantic_working_memory(
                memory_store,
                owner_id=owner_id,
                query=query,
                query_embedding=query_embedding,
                embedding_model=query_embedding_model,
            ),
        )
    else:
        query_embedding = None
        query_embedding_model = None
        retrieval_path = "skipped_empty_store"
        episodic_entries: list[WorkingMemoryEntry] = []
        semantic_entries: list[WorkingMemoryEntry] = []

    retrieval_duration_ms = (time.monotonic() - retrieval_start) * 1000

    working_memory = [*episodic_entries, *semantic_entries]

    summary = _build_load_memory_summary(
        semantic_hits=len(semantic_entries),
        semantic_store_size=semantic_store_size,
        episodic_hits=len(episodic_entries),
        episodic_store_size=episodic_store_size,
        procedural_count=len(procedural_rules),
        proactive_recall_enabled=proactive_recall_enabled,
        retrieval_path=retrieval_path,
        meaningful_query_token_count=len(meaningful_query_tokens),
    )

    logger.info(
        "load_memory_node: semantic=%d/%d episodic=%d/%d procedural=%d "
        "recall=%s path=%s first_turn=%s query_tokens=%d duration_ms=%.2f owner=%r",
        len(semantic_entries),
        semantic_store_size,
        len(episodic_entries),
        episodic_store_size,
        len(procedural_rules),
        "on" if proactive_recall_enabled else "off",
        retrieval_path,
        is_first_turn,
        len(meaningful_query_tokens),
        retrieval_duration_ms,
        owner_id,
    )

    return {
        "working_memory": list(working_memory),
        "memory": {
            **state.get("memory", {}),
            "summary": summary,
            "procedural_rules": procedural_rules,
            "proactive_recall_enabled": proactive_recall_enabled,
        },
        "diagnostics": _build_load_memory_diagnostics(
            retrieval_duration_ms=retrieval_duration_ms,
            semantic_hits=len(semantic_entries),
            semantic_store_size=semantic_store_size,
            episodic_hits=len(episodic_entries),
            episodic_store_size=episodic_store_size,
            procedural_count=len(procedural_rules),
            proactive_recall_enabled=proactive_recall_enabled,
            retrieval_path=retrieval_path,
        ),
    }
