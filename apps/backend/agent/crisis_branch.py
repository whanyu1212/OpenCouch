"""Shared execution helpers for crisis branch side effects."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from agent.audit.models import CrisisLogRecord
from agent.memory.hashing import hash_session_id as _hash_session_id
from agent.memory.hashing import iso_now
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded_search import find_crisis_resources

logger = logging.getLogger(__name__)


async def build_crisis_resource_lookup_delta(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Resolve local crisis-resource state for the current crisis turn."""

    llm_client = context.llm_client
    if llm_client is None:
        raise RuntimeError("crisis_resource_lookup_node requires an LLM client.")

    (
        inferred_location,
        found_resources,
        resource_lookup_status,
    ) = await find_crisis_resources(state, llm_client=llm_client)

    return {
        "inferred_location": inferred_location,
        "found_resources": found_resources,
        "resource_lookup_status": resource_lookup_status,
    }


def crisis_response_delta(response_text: str) -> dict[str, Any]:
    """Return the shared response delta for crisis-response turns."""

    return {
        "route": "crisis",
        "response_style": "crisis_response",
        "response_text": response_text,
    }


async def write_crisis_log(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Write a crisis event record to the always-on safety audit log."""

    crisis = state.get("crisis")
    if crisis is None or not crisis.needs_crisis_response:
        logger.debug("crisis log called on non-crisis turn; skipping write")
        return {}

    backend = context.crisis_log_backend
    crisis_audit = state.get("crisis_audit", {})
    override_kind = crisis_audit.get("crisis_override_kind", "none")
    classifier_path = crisis_audit.get("crisis_classifier_path", "llm_primary")
    llm_failure_occurred = crisis_audit.get("crisis_llm_failure_occurred", False)

    if "crisis_classifier_path" not in crisis_audit:
        logger.debug(
            "crisis log: no classifier_path in crisis_audit state; "
            "using default 'llm_primary'"
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
        logger.error(
            "crisis_log_node failed to write record; audit trail lost for this event",
            exc_info=True,
        )

    return {}


__all__ = [
    "_hash_session_id",
    "build_crisis_resource_lookup_delta",
    "crisis_response_delta",
    "write_crisis_log",
]
