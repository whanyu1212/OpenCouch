"""Terminal memory extraction node.

This node keeps the top-level graph topology thin by wrapping the semantic and
procedural extraction side-effect nodes behind one terminal graph node. The
individual extractors remain independently testable and preserve their existing
skip, diagnostics, and failure-isolation behavior.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.runtime import Runtime

from agent.nodes.extract_facts import run_extract_semantic_facts_node
from agent.nodes.extract_procedural_rules import run_extract_procedural_rules_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_memory_extraction_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Run terminal memory side effects and merge their diagnostics.

    Args:
        state: Current graph state after turn finalization.
        runtime: LangGraph runtime carrying memory dependencies.

    Returns:
        State delta containing merged extractor diagnostics.
    """

    semantic_delta, procedural_delta = await asyncio.gather(
        run_extract_semantic_facts_node(state, runtime),
        run_extract_procedural_rules_node(state, runtime),
    )

    diagnostics = {
        **semantic_delta.get("diagnostics", {}),
        **procedural_delta.get("diagnostics", {}),
    }
    if not diagnostics:
        return {}
    return {"diagnostics": diagnostics}
