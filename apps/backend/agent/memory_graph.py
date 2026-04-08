"""Graph-memory interface and Graphiti-backed episodic memory adapter."""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from typing import Protocol

from core.config import load_runtime_env

from agent.semantic_signals import derive_semantic_signals
from agent.state import AgentState

logger = logging.getLogger(__name__)

ORIENTATION_QUERY_TERMS = (
    "what can you do",
    "how does this work",
    "i'm new here",
    "im new here",
    "what are you",
    "who are you",
    "help menu",
    "/help",
)

RECALL_TRIGGER_TERMS = (
    "again",
    "back here",
    "back again",
    "still",
    "similar",
    "same pattern",
    "pattern",
    "theme",
    "connection",
    "keeps happening",
    "keep happening",
    "why do i keep",
    "why does this keep",
    "is this like before",
    "is this similar",
    "last time",
    "before",
    "used to",
    "no longer",
    "worse again",
    "better this time",
    "got worse",
    "got better",
    "understand the pattern",
)

TEMPORAL_UPDATE_TERMS = (
    "again",
    "still",
    "lately",
    "recently",
    "these days",
    "used to",
    "no longer",
    "anymore",
    "got worse",
    "getting worse",
    "got better",
    "better this time",
    "since",
    "for months",
    "for years",
)

TRIGGER_CONTEXT_TERMS = (
    "whenever",
    "every time",
    "when i",
    "when my",
    "when we",
    "after i",
    "after we",
    "after talking",
    "after seeing",
    "trigger",
    "set off",
)

FOLLOW_UP_COMMITMENT_TERMS = (
    "i will",
    "i'll",
    "i can try",
    "i could try",
    "going to",
    "gonna",
    "next time",
    "plan to",
)

LOW_INFORMATION_REPLIES = {
    "yes",
    "yeah",
    "yep",
    "no",
    "nope",
    "ok",
    "okay",
    "maybe",
    "sure",
    "both",
}

SKIPPED_GRAPH_MEMORY_MODES = {
    "orientation",
    "out_of_scope",
    "realignment",
    "safety_check",
}


class GraphMemoryStore(Protocol):
    """Protocol for graph-backed episodic memory retrieval and writes."""

    async def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 4,
    ) -> list[str]:
        """Return graph-backed episodic memory snippets for the current turn."""

    async def record_episode(
        self,
        *,
        owner_id: str,
        state: AgentState,
    ) -> bool:
        """Persist one completed turn into graph-backed episodic memory."""


class NullGraphMemoryStore:
    """No-op graph-memory adapter used until Graphiti is wired in."""

    async def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 4,
    ) -> list[str]:
        del owner_id, query, limit
        return []

    async def record_episode(
        self,
        *,
        owner_id: str,
        state: AgentState,
    ) -> bool:
        del owner_id, state
        return False


def _normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase free text for heuristic matching.

    Args:
        text: Raw free-form text.

    Returns:
        Normalized text for deterministic term checks.
    """

    return " ".join(text.lower().split())


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """Return whether text contains any term from the candidate set.

    Args:
        text: Normalized text to inspect.
        terms: Candidate literal substrings.

    Returns:
        `True` when any term appears in the text.
    """

    return any(term in text for term in terms)


def _slug_fragment(value: str) -> str:
    """Convert a string into a compact ASCII-safe identifier fragment.

    Args:
        value: Raw string that should become part of an identifier.

    Returns:
        A lowercase identifier fragment using letters, digits, `_`, and `-`.
    """

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or "session"


def should_retrieve_graph_memory(
    *,
    message: str,
    prior_state: AgentState | None,
) -> bool:
    """Decide whether the current turn warrants Graphiti retrieval.

    Args:
        message: Current inbound user message.
        prior_state: Most recent persisted state for this thread, when available.

    Returns:
        `True` when long-range episodic recall is likely to help.
    """

    text = _normalize_text(message)
    if not text:
        return False
    if _contains_any(text, ORIENTATION_QUERY_TERMS):
        return False
    if text in LOW_INFORMATION_REPLIES and not _contains_any(
        text, RECALL_TRIGGER_TERMS
    ):
        return False
    if len(text.split()) <= 3 and not _contains_any(text, RECALL_TRIGGER_TERMS):
        return False
    if _contains_any(text, RECALL_TRIGGER_TERMS):
        return True
    if prior_state is None:
        return False

    prior_intent = prior_state.get("session_intent")
    prior_mode = prior_state.get("mode")
    prior_concerns = prior_state.get("active_concerns", [])
    if prior_intent == "reflection_and_pattern_finding" and len(text.split()) >= 5:
        return True
    if prior_mode == "pattern_reflection" and len(text.split()) >= 5:
        return True
    if prior_concerns and any(
        marker in text for marker in ("why", "before", "similar", "connection", "theme")
    ):
        return True
    return False


def build_graph_memory_query(
    *,
    message: str,
    prior_state: AgentState | None,
) -> str:
    """Build a compact Graphiti query from the current turn and prior context.

    Args:
        message: Current inbound user message.
        prior_state: Most recent persisted state for this thread, when available.

    Returns:
        A compact text query for owner-scoped graph retrieval.
    """

    parts = [f"Current user message: {message.strip()}"]
    if prior_state is None:
        return "\n".join(parts)

    concerns = prior_state.get("active_concerns", [])
    if concerns:
        parts.append(f"Active concerns: {', '.join(concerns[:3])}")

    session_intent = prior_state.get("session_intent")
    if session_intent == "reflection_and_pattern_finding":
        parts.append("Need prior recurring patterns or similar episodes.")
    elif session_intent == "psychoeducation":
        parts.append("Need prior symptom history or changes over time.")

    session_stage = prior_state.get("session_stage")
    if session_stage and session_stage != "closing":
        parts.append(f"Prior session stage: {session_stage}")

    return "\n".join(parts)


def should_record_graph_episode(state: AgentState) -> bool:
    """Decide whether the completed turn is durable enough for Graphiti.

    Args:
        state: Completed post-turn agent state.

    Returns:
        `True` when the turn contains durable, recall-worthy material.
    """

    mode_type = state.get("mode_type")
    if mode_type is None or mode_type.value != "therapeutic":
        return False

    mode = state.get("mode")
    if mode in SKIPPED_GRAPH_MEMORY_MODES:
        return False

    message = _normalize_text(state.get("message", ""))
    if not message or len(message.split()) < 4:
        return False

    signals = state.get("semantic_signals") or derive_semantic_signals(state)
    if signals.get("needs_supportive_boundary", False):
        return False

    has_temporal_update = _contains_any(message, TEMPORAL_UPDATE_TERMS)
    has_trigger_context = _contains_any(message, TRIGGER_CONTEXT_TERMS)
    has_follow_up_commitment = _contains_any(message, FOLLOW_UP_COMMITMENT_TERMS)
    has_durable_theme = any(
        (
            signals.get("has_relational_theme", False),
            signals.get("has_grief_theme", False),
            signals.get("wants_pattern_reflection", False),
        )
    )

    if mode == "pattern_reflection":
        return True
    if mode == "psychoeducation":
        return (
            has_temporal_update
            or has_durable_theme
            or bool(state.get("active_concerns"))
        )
    if mode == "guided_exercise":
        return (
            has_temporal_update
            or has_follow_up_commitment
            or (has_trigger_context and bool(state.get("active_concerns")))
        )
    if mode == "supportive_conversation":
        return (
            has_temporal_update
            or has_trigger_context
            or has_follow_up_commitment
            or has_durable_theme
        )
    return (
        has_temporal_update
        or has_trigger_context
        or has_follow_up_commitment
        or has_durable_theme
    )


def build_graph_episode_payload(
    state: AgentState,
) -> tuple[str, str, str]:
    """Build a curated Graphiti episode payload for one durable turn.

    Args:
        state: Completed post-turn agent state.

    Returns:
        A tuple of `(episode_name, episode_body, source_description)`.
    """

    message = state["message"].strip()
    session_id = _slug_fragment(state.get("session_id") or "session")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    episode_name = f"opencouch-{session_id}-turn-{state['turn_count']}-{timestamp}"

    signals = state.get("semantic_signals") or derive_semantic_signals(state)
    durable_cues: list[str] = []
    normalized_message = _normalize_text(message)
    if signals.get("has_relational_theme", False):
        durable_cues.append("relational theme")
    if signals.get("has_grief_theme", False):
        durable_cues.append("grief theme")
    if signals.get("has_anxiety_theme", False):
        durable_cues.append("anxiety theme")
    if signals.get("has_stress_theme", False):
        durable_cues.append("stress theme")
    if signals.get("wants_pattern_reflection", False):
        durable_cues.append("pattern exploration")
    if _contains_any(normalized_message, TEMPORAL_UPDATE_TERMS):
        durable_cues.append("temporal update")
    if _contains_any(normalized_message, TRIGGER_CONTEXT_TERMS):
        durable_cues.append("trigger or context link")
    if _contains_any(normalized_message, FOLLOW_UP_COMMITMENT_TERMS):
        durable_cues.append("follow-up commitment")

    lines = [f"User shared: {message}"]
    concerns = state.get("active_concerns", [])
    if concerns:
        lines.append(f"Active concerns: {', '.join(concerns[:3])}")

    current_goal = (state.get("current_goal") or "").strip()
    if current_goal and _normalize_text(current_goal) != normalized_message:
        lines.append(f"Current goal: {current_goal}")

    session_intent = state.get("session_intent")
    if session_intent:
        lines.append(f"Session intent: {session_intent.replace('_', ' ')}")

    session_stage = state.get("session_stage")
    if session_stage:
        lines.append(f"Session stage: {session_stage.replace('_', ' ')}")

    lines.append(
        f"Therapeutic mode: {(state.get('mode') or 'supportive_conversation').replace('_', ' ')}"
    )
    if durable_cues:
        lines.append(f"Durable cues: {', '.join(durable_cues)}")

    return (
        episode_name,
        "\n".join(lines),
        "OpenCouch curated therapeutic memory episode",
    )


def create_graph_memory_store_from_env() -> GraphMemoryStore:
    """Build a Graphiti-backed store when required env vars are available."""

    load_runtime_env()

    uri = os.getenv("GRAPHITI_NEO4J_URI")
    user = os.getenv("GRAPHITI_NEO4J_USER")
    password = os.getenv("GRAPHITI_NEO4J_PASSWORD")
    database = os.getenv("GRAPHITI_NEO4J_DATABASE", "neo4j")
    api_key = os.getenv("GRAPHITI_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not all([uri, user, password, api_key]):
        return NullGraphMemoryStore()

    graphiti_model = os.getenv("GRAPHITI_OPENAI_MODEL", "gpt-4.1-mini")
    small_model = os.getenv("GRAPHITI_OPENAI_SMALL_MODEL", graphiti_model)
    embedding_model = os.getenv(
        "GRAPHITI_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    reranker_model = os.getenv("GRAPHITI_OPENAI_RERANKER_MODEL", small_model)

    return GraphitiMemoryStore(
        uri=uri,
        user=user,
        password=password,
        database=database,
        api_key=api_key,
        graphiti_model=graphiti_model,
        small_model=small_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )


class GraphitiMemoryStore:
    """Graphiti-backed episodic memory store keyed by owner/group id."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str,
        api_key: str,
        graphiti_model: str,
        small_model: str,
        embedding_model: str,
        reranker_model: str,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._api_key = api_key
        self._graphiti_model = graphiti_model
        self._small_model = small_model
        self._embedding_model = embedding_model
        self._reranker_model = reranker_model
        self._graphiti = None
        self._initialized = False

    async def _ensure_graphiti(self):
        """Lazily initialize the Graphiti client and database constraints."""

        if self._graphiti is not None:
            return self._graphiti

        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import (
            OpenAIRerankerClient,
        )
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient

        driver = Neo4jDriver(
            uri=self._uri,
            user=self._user,
            password=self._password,
            database=self._database,
        )
        graphiti = Graphiti(
            graph_driver=driver,
            llm_client=OpenAIClient(
                config=LLMConfig(
                    api_key=self._api_key,
                    model=self._graphiti_model,
                    small_model=self._small_model,
                )
            ),
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=self._api_key,
                    embedding_model=self._embedding_model,
                )
            ),
            cross_encoder=OpenAIRerankerClient(
                config=LLMConfig(
                    api_key=self._api_key,
                    model=self._reranker_model,
                    small_model=self._small_model,
                )
            ),
        )
        self._graphiti = graphiti
        if not self._initialized:
            await graphiti.build_indices_and_constraints()
            self._initialized = True
        return graphiti

    async def close(self) -> None:
        """Close the underlying Graphiti client when the runtime shuts down."""

        if self._graphiti is not None:
            await self._graphiti.close()
            self._graphiti = None

    async def retrieve(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 4,
    ) -> list[str]:
        """Retrieve owner-scoped episodic memory facts from Graphiti."""

        try:
            graphiti = await self._ensure_graphiti()
            results = await graphiti.search(
                query=query,
                group_ids=[owner_id],
                num_results=limit,
            )
        except Exception as exc:
            logger.warning("Graphiti retrieval failed: %s", exc)
            return []

        memories: list[str] = []
        seen: set[str] = set()
        for edge in results:
            fact = (edge.fact or edge.name or "").strip()
            if not fact or fact in seen:
                continue
            seen.add(fact)
            memories.append(fact)
            if len(memories) >= limit:
                break
        return memories

    async def record_episode(
        self,
        *,
        owner_id: str,
        state: AgentState,
    ) -> bool:
        """Persist one completed therapeutic turn into Graphiti."""

        if not should_record_graph_episode(state):
            return False

        episode_name, episode_body, source_description = build_graph_episode_payload(
            state
        )
        if not episode_body:
            return False

        try:
            graphiti = await self._ensure_graphiti()
            await graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=datetime.now(UTC),
                group_id=owner_id,
            )
            return True
        except Exception as exc:
            logger.warning("Graphiti episode write failed: %s", exc)
            return False
