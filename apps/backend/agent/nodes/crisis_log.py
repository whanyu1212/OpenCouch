"""Crisis log node for the always-on safety audit trail.

Writes a :class:`CrisisLogRecord` to the crisis log backend whenever a
crisis event is detected and the crisis response has run. Always-on
regardless of memory mode: in incognito mode, ``user_id_or_null`` is
null and ``session_id_opaque`` is a SHA-256 hash of the session id with
no reverse mapping.

- Writes one record per crisis event, keyed off ``state["crisis"]``.
- Runs on the crisis branch only, after ``crisis_response_node``.
- ``response_node_completed = True`` unconditionally (if we're
  executing, the response node finished — failures are handled by the
  response node's own try/except, not by this node).
- Failures in this node are logged LOUDLY via ``exc_info=True`` but do
  not fail the turn. A silent crisis log failure would mean the
  operator loses the audit trail, so observability is essential.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from langgraph.runtime import Runtime

from agent.memory.hashing import hash_session_id as _hash_session_id
from agent.memory.hashing import iso_now
from agent.audit.models import CrisisLogRecord
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

# Keep ``_hash_session_id`` re-exported for existing imports; the canonical
# implementation lives in ``agent.memory.hashing``.
__all__ = ["_hash_session_id", "run_crisis_log_node"]

logger = logging.getLogger(__name__)


async def run_crisis_log_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Write a crisis event record to the crisis log backend.

    Args:
        state: Current graph state for the turn being processed.
        runtime: LangGraph runtime carrying the crisis log backend.

    Returns:
        An empty state delta. The node only performs the audit-log side effect.
    """

    crisis = state.get("crisis")
    if crisis is None or not crisis.needs_crisis_response:
        # Defensive guard: the parent graph should only route crisis turns here,
        # but writing no record is safer than writing a spurious audit event.
        logger.debug("crisis_log_node called on non-crisis turn; skipping write")
        return {}

    backend = runtime.context.crisis_log_backend

    # Read the crisis debug metadata from the dedicated audit channel.
    #
    # Missing-field defaults keep partial-state fixtures working. In production,
    # a missing field means the crisis gate added a path without audit metadata.
    crisis_audit = state.get("crisis_audit", {})
    override_kind = crisis_audit.get("crisis_override_kind", "none")
    classifier_path = crisis_audit.get("crisis_classifier_path", "deterministic")
    llm_failure_occurred = crisis_audit.get("crisis_llm_failure_occurred", False)

    if "crisis_classifier_path" not in crisis_audit:
        logger.debug(
            "crisis_log_node: no classifier_path in crisis_audit state; "
            "using backward-compat default 'deterministic'"
        )

    record = CrisisLogRecord(
        id=str(uuid4()),
        session_id_opaque=_hash_session_id(state.get("session_id")),
        user_id_or_null=state.get("user_id"),
        detected_at=iso_now(),
        level=crisis.level,  # type: ignore[arg-type]
        override_kind=override_kind,
        classifier_path=classifier_path,
        reason=crisis.reason or "",
        response_node_completed=True,
        llm_failure_occurred=llm_failure_occurred,
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
