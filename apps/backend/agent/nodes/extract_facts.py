"""Semantic fact extraction node — phase 1 v0.1 stub.

This node is a **deliberate stub** in v0.1. Its only job right now is to
prove the wiring: the parent graph includes it in the topology, it gets
called after the response nodes (crisis and therapeutic), and it returns
an empty state delta without crashing.

Real semantic extraction — the LLM call that produces structured
:class:`MemoryWrite` objects and persists them via the memory store's
hot-path deduplication path — lands in v0.3. The full node design is
documented in ``agent/memory/nodes_sketch.py`` under
``extract_semantic_facts_node``.

The name ``extract_semantic_facts_node`` is aspirational: when v0.3
replaces this stub with the real implementation, no call sites need to
change. Only the body of this function is rewritten.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def run_extract_semantic_facts_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """No-op semantic-extraction stub for phase 1 v0.1.

    Logs a debug message and returns an empty state delta. Real
    extraction lands in v0.3 — see ``agent/memory/nodes_sketch.py``.

    The node exists in the parent graph topology from v0.1 onward so
    that the wiring is proven end-to-end before the real extraction
    logic is implemented. Tests that drive the graph through
    ``run_agent`` will observe that this node executes in the right
    order without affecting state.
    """

    logger.debug(
        "extract_semantic_facts_node (v0.1 stub): no-op, real extraction in v0.3"
    )
    return {}
