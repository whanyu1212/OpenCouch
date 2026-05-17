"""Crisis log node for the always-on safety audit trail.

Writes a :class:`CrisisLogRecord` to the crisis log backend whenever a
crisis event is detected and the crisis response has run. Always-on
regardless of memory mode: in incognito mode, ``user_id_or_null`` is
null and ``session_id_opaque`` is a SHA-256 hash of the session id with
no reverse mapping.

- Writes one record per crisis event, keyed off ``state["crisis"]``.
- Runs on the crisis branch only, after ``crisis_response_node``.
- ``response_node_completed = True`` unconditionally: if this node executes,
  the response node has finished successfully.
- Failures in this node are logged LOUDLY via ``exc_info=True`` but do
  not fail the turn. A silent crisis log failure would mean the
  operator loses the audit trail, so observability is essential.
"""

from __future__ import annotations

from typing import Any

from agent.crisis_branch import _hash_session_id, write_crisis_log
from agent.state import AgentState

# Keep ``_hash_session_id`` re-exported for existing imports; the canonical
# implementation lives in ``agent.memory.hashing``.
__all__ = ["_hash_session_id", "run_crisis_log_node"]


async def run_crisis_log_node(
    state: AgentState,
    runtime: Any,
) -> dict[str, Any]:
    """Write a crisis event record to the crisis log backend.

    Args:
        state: Current graph state for the turn being processed.
        runtime: Runtime object carrying the crisis log backend.

    Returns:
        An empty state delta. The node only performs the audit-log side effect.
    """

    return await write_crisis_log(state, runtime.context)
