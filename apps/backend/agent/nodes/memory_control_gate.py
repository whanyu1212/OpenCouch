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
from agent.gates.memory_control import (
    is_pending_cancellation,
    is_pending_confirmation,
    resolve_memory_control_action,
)
from agent.observability.routing_trace import append_routing_trace
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
                    "diagnostics": {
                        "memory_control_gate_ms": elapsed_ms(start),
                        **append_routing_trace(
                            state.get("diagnostics"),
                            {
                                "stage": "memory",
                                "decision": "confirm_pending",
                                "source": "pending_action",
                                "reason": "User confirmed the pending memory action.",
                            },
                        ),
                    },
                },
                goto=MEMORY_CONTROL_NODE,
            )
        if is_pending_cancellation(message):
            return Command(
                update={
                    "route": "memory_control",
                    "memory_control": {"action": {"type": "cancel_pending"}},
                    "diagnostics": {
                        "memory_control_gate_ms": elapsed_ms(start),
                        **append_routing_trace(
                            state.get("diagnostics"),
                            {
                                "stage": "memory",
                                "decision": "cancel_pending",
                                "source": "pending_action",
                                "reason": "User cancelled the pending memory action.",
                            },
                        ),
                    },
                },
                goto=MEMORY_CONTROL_NODE,
            )
        return Command(
            update={
                "memory_control": {"pending_action": None, "action": {}},
                "diagnostics": {
                    "memory_control_gate_ms": elapsed_ms(start),
                    **append_routing_trace(
                        state.get("diagnostics"),
                        {
                            "stage": "memory",
                            "decision": "pass",
                            "source": "pending_action",
                            "reason": (
                                "User did not confirm the pending memory action; "
                                "continuing normal routing."
                            ),
                        },
                    ),
                },
            },
            goto=GROUNDED_LOOKUP_GATE_NODE,
        )

    route = await resolve_memory_control_action(
        state,
        llm_client=runtime.context.llm_client,
    )
    decision = "pass"
    if route.action is not None:
        decision = str(route.action.payload.get("type") or "memory_control")

    diagnostics = {
        "memory_control_gate_ms": elapsed_ms(start),
        "memory_control_classifier_path": route.classifier_path,
        "memory_control_llm_failure_occurred": route.llm_failure_occurred,
        **append_routing_trace(
            state.get("diagnostics"),
            {
                "stage": "memory",
                "decision": decision,
                "source": route.classifier_path,
                "reason": route.reason,
                "confidence": route.confidence,
            },
        ),
    }

    if route.action is None:
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
            "memory_control": {"action": route.action.to_state_action()},
            "diagnostics": diagnostics,
        },
        goto=MEMORY_CONTROL_NODE,
    )
