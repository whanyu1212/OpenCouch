"""Memory-control routing gate for explicit user memory commands."""

from __future__ import annotations

import time

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.graph_constants import (
    GROUNDED_LOOKUP_GATE_NODE,
    MEMORY_CONTROL_NODE,
    MemoryControlGateNextNode,
)
from agent.memory_control.router import (
    is_pending_cancellation,
    is_pending_confirmation,
    resolve_memory_control_action,
)
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_memory_control_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[MemoryControlGateNextNode]:
    """Route explicit memory-control turns before normal memory loading.

    Args:
        state (AgentState): Current graph state after crisis classification.
        runtime (Runtime[WorkflowContext]): LangGraph runtime carrying the
            workflow context.

    Returns:
        Command[MemoryControlGateNextNode]: State update plus the next node to
            run.
    """

    start = time.monotonic()
    message = state.get("message", "")
    pending_action = (state.get("memory_control", {}) or {}).get("pending_action")

    if pending_action:
        if is_pending_confirmation(message):
            return Command(
                update={
                    "route": "memory_control",
                    "memory_control": {"action": {"type": "confirm_pending"}},
                    "diagnostics": {"memory_control_gate_ms": elapsed_ms(start)},
                },
                goto=MEMORY_CONTROL_NODE,
            )
        if is_pending_cancellation(message):
            return Command(
                update={
                    "route": "memory_control",
                    "memory_control": {"action": {"type": "cancel_pending"}},
                    "diagnostics": {"memory_control_gate_ms": elapsed_ms(start)},
                },
                goto=MEMORY_CONTROL_NODE,
            )
        return Command(
            update={
                "memory_control": {"pending_action": None, "action": {}},
                "diagnostics": {"memory_control_gate_ms": elapsed_ms(start)},
            },
            goto=GROUNDED_LOOKUP_GATE_NODE,
        )

    (
        action,
        classifier_path,
        llm_failure_occurred,
    ) = await resolve_memory_control_action(
        state,
        llm_client=runtime.context.llm_client,
    )
    diagnostics = {
        "memory_control_gate_ms": elapsed_ms(start),
        "memory_control_classifier_path": classifier_path,
        "memory_control_llm_failure_occurred": llm_failure_occurred,
    }

    if action is None:
        return Command(
            update={
                "memory_control": {"action": {}},
                "diagnostics": diagnostics,
            },
            goto=GROUNDED_LOOKUP_GATE_NODE,
        )

    return Command(
        update={
            "route": "memory_control",
            "memory_control": {"action": action},
            "diagnostics": diagnostics,
        },
        goto=MEMORY_CONTROL_NODE,
    )
