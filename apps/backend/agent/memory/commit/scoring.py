"""Support-scoring helpers for session-end memory commit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.memory.store import MemoryStore
from agent.memory.text_tokens import tokenize_meaningful
from agent.state import AgentState

if TYPE_CHECKING:
    from agent.memory.types import StoredSessionArc


def _user_turn_texts(state: AgentState) -> list[str]:
    """Return the user-turn transcript texts for session-end scoring."""
    transcript = state.get("transcript", [])
    return [
        (turn.get("content") or "").strip()
        for turn in transcript
        if turn.get("role") == "user" and (turn.get("content") or "").strip()
    ]


def _count_supported_user_turns(
    candidate_tokens: frozenset[str],
    user_turn_texts: list[str],
    *,
    exact_terms: tuple[str, ...] = (),
) -> int:
    """Count how many user turns materially support this candidate."""
    if not candidate_tokens and not exact_terms:
        return 0

    supported = 0
    for text in user_turn_texts:
        lowered = text.lower()
        if any(term and term in lowered for term in exact_terms):
            supported += 1
            continue

        overlap = candidate_tokens & tokenize_meaningful(text)
        if len(overlap) >= 2:
            supported += 1
    return supported


def _count_supporting_session_texts(
    candidate_tokens: frozenset[str],
    support_texts: list[str],
    *,
    exact_terms: tuple[str, ...] = (),
) -> int:
    """Count how many session-level texts materially support this candidate."""
    if not candidate_tokens and not exact_terms:
        return 0

    supported = 0
    for text in support_texts:
        lowered = text.lower()
        if any(term and term in lowered for term in exact_terms):
            supported += 1
            continue

        overlap = candidate_tokens & tokenize_meaningful(text)
        if len(overlap) >= 2:
            supported += 1
    return supported


def _session_support_text(stored_arc: "StoredSessionArc | None") -> str:
    """Flatten the stored session arc into one support text blob."""
    if stored_arc is None:
        return ""

    parts = [stored_arc.summary]
    parts.extend(stored_arc.primary_themes)
    parts.extend(stored_arc.open_loops)
    parts.extend(stored_arc.resolved_threads)
    return " ".join(part for part in parts if part).strip()


def _arc_support_score(
    candidate_tokens: frozenset[str],
    *,
    stored_arc: "StoredSessionArc | None",
    exact_terms: tuple[str, ...] = (),
) -> int:
    """Return a small support score from the episodic summary fields."""
    support_text = _session_support_text(stored_arc)
    if not support_text:
        return 0

    score = 0
    lowered_support = support_text.lower()
    if any(term and term in lowered_support for term in exact_terms):
        score += 2

    overlap = candidate_tokens & tokenize_meaningful(support_text)
    if len(overlap) >= 2:
        score += 1
    if len(overlap) >= 3:
        score += 1
    return score


async def _load_prior_session_support_texts(
    memory_store: MemoryStore,
    *,
    owner_id: str,
    current_session_ids: set[str],
) -> list[str]:
    """Return support texts from prior episodic arcs for this owner."""
    records = await memory_store.asearch((owner_id, "episodic"), query=None, limit=100)
    prior_texts: list[str] = []
    for record in records:
        value = record.value
        if value.get("session_id") in current_session_ids:
            continue

        parts = [value.get("summary", "")]
        parts.extend(value.get("primary_themes", []))
        parts.extend(value.get("open_loops", []))
        parts.extend(value.get("resolved_threads", []))
        support_text = " ".join(part for part in parts if part).strip()
        if support_text:
            prior_texts.append(support_text)
    return prior_texts
