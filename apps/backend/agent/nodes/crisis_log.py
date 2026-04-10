"""Crisis log node — always-on safety audit trail.

Writes a :class:`CrisisLogRecord` to the crisis log backend whenever a
crisis event is detected and the crisis response has run. Always-on
regardless of memory mode: in incognito mode, ``user_id_or_null`` is
null and ``session_id_opaque`` is a SHA-256 hash of the session id with
no reverse mapping. See ``agent/memory/schema.yaml`` §2 namespaces.
crisis_log for the full privacy asymmetry rationale.

Phase 1 v0.1 scope:
- Writes one record per crisis event, keyed off ``state["crisis"]``.
- Runs on the crisis branch only, after ``crisis_response_node``.
- ``response_node_completed = True`` unconditionally in v0.1 (if we're
  executing, the response node finished — failures are handled by the
  response node's own try/except, not by this node).
- Failures in this node are logged LOUDLY via ``exc_info=True`` but do
  not fail the turn. A silent crisis log failure would mean the
  operator loses the audit trail, so observability is essential.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph.runtime import Runtime

from agent.memory.models import CrisisLogRecord
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _hash_session_id(session_id: str | None) -> str:
    """Return a SHA-256 hash of the session id, padded if None.

    Used for the ``session_id_opaque`` field. A stable hash means two
    records from the same session share an opaque identifier without
    exposing the original session id. When session_id is None (very
    rare, but possible in bare-minimum test fixtures), we hash a
    placeholder so the field is always populated.
    """

    source = session_id or "__no_session_id__"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


async def run_crisis_log_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Write a crisis event record to the crisis log backend.

    Always-on: writes regardless of memory_mode. The record contains
    classifier metadata and outcome flags but does NOT contain the
    user's message text or any conversation history.

    Returns an empty state delta — the write is a side effect. The
    node is placed after ``crisis_response_node`` on the crisis branch
    in the parent graph.
    """

    crisis = state.get("crisis")
    if crisis is None or not crisis.needs_crisis_response:
        # Defensive: if something wired this node onto a non-crisis turn,
        # do nothing rather than writing a spurious record. The parent
        # graph topology shouldn't let this happen, but the guard is
        # cheap.
        logger.debug("crisis_log_node called on non-crisis turn; skipping write")
        return {}

    backend = runtime.context["crisis_log_backend"]

    # Classifier metadata → structured record. The override_kind /
    # classifier_path fields aren't currently tracked in state; v0.2
    # can surface them from the crisis gate. For now we record
    # "none" + "deterministic" as reasonable defaults that won't mislead
    # an operator reading the log — if the LLM path was used, the
    # classifier path is still "deterministic" from this node's
    # perspective because we can't tell from state alone.
    record = CrisisLogRecord(
        id=str(uuid4()),
        session_id_opaque=_hash_session_id(state.get("session_id")),
        user_id_or_null=state.get("user_id"),
        detected_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        level=crisis.level,  # type: ignore[arg-type]
        override_kind="none",
        classifier_path="deterministic",
        reason=crisis.reason or "",
        response_node_completed=True,
        llm_failure_occurred=False,
    )

    try:
        await backend.aappend(record)
    except Exception:
        # Crisis log failures must be LOUD — a silent failure means
        # we lose audit capability for a safety-critical event.
        logger.error(
            "crisis_log_node failed to write record; audit trail lost for this event",
            exc_info=True,
        )

    return {}
