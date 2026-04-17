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

import logging
from typing import Any
from uuid import uuid4

from langgraph.runtime import Runtime

from agent.memory.hashing import hash_session_id as _hash_session_id
from agent.memory.hashing import iso_now
from agent.memory.models import CrisisLogRecord
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

# Backward-compat: ``_hash_session_id`` used to live in this module.
# It's now shared across memory subsystems (see agent/memory/hashing.py),
# but existing imports like ``from agent.nodes.crisis_log import
# _hash_session_id`` keep working via the alias above. The semantics
# are identical — None/empty -> ``"__no_session_id__"`` placeholder,
# otherwise SHA-256 of the UTF-8 bytes.
__all__ = ["_hash_session_id", "run_crisis_log_node"]

logger = logging.getLogger(__name__)


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

    backend = runtime.context.crisis_log_backend

    # Read the crisis debug metadata from routing state. The crisis gate
    # writes these three fields in its delta for every crisis-path turn
    # (see ``agent/nodes/crisis_gate.py`` — the five dispatch paths each
    # set all three before falling through to ``_build_crisis_delta``).
    #
    # Missing-field defaults exist as a backward-compat safety net for
    # partial-state test fixtures. If a field is missing in production,
    # it means the crisis gate regressed and a new path was added
    # without setting the metadata. We log a debug breadcrumb so the
    # regression leaves a trace in the audit log.
    routing = state.get("routing", {})
    override_kind = routing.get("crisis_override_kind", "none")
    classifier_path = routing.get("crisis_classifier_path", "deterministic")
    llm_failure_occurred = routing.get("crisis_llm_failure_occurred", False)

    if "crisis_classifier_path" not in routing:
        # Breadcrumb for regressions: if a production crisis turn ever
        # hits this debug line, the crisis gate is missing metadata for
        # one of its dispatch paths.
        logger.debug(
            "crisis_log_node: no classifier_path in routing state; "
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
