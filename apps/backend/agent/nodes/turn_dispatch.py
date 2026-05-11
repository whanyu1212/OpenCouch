"""Turn-level dispatch node for non-crisis turns."""

from __future__ import annotations

import time

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.graph_constants import (
    GROUNDED_ANSWER_NODE,
    LOAD_MEMORY_NODE,
    MEMORY_CONTROL_NODE,
    TurnDispatchNextNode,
)
from agent.observability.routing_trace import append_routing_trace
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.turn_dispatch import TurnDispatchPlan, plan_turn_route


def _dispatch_update(
    state: AgentState,
    plan: TurnDispatchPlan,
    *,
    duration_ms: float,
) -> dict[str, object]:
    memory_action = (
        plan.memory_action.to_state_action() if plan.memory_action is not None else {}
    )
    decision = plan.route
    if plan.memory_action is not None:
        decision = str(memory_action.get("type") or decision)

    diagnostics = {
        "turn_dispatch_ms": round(duration_ms, 2),
        "turn_dispatch_classifier_path": "llm_primary",
        "turn_dispatch_llm_failure_occurred": False,
        **append_routing_trace(
            state.get("diagnostics"),
            {
                "stage": "turn_dispatch",
                "decision": decision,
                "source": "llm_primary",
                "reason": plan.reason,
                "confidence": plan.confidence,
                "active_flow": plan.active_flow,
                "active_flow_action": plan.active_flow_action,
                "memory_reference_mode": plan.memory_reference_mode,
            },
        ),
    }

    update: dict[str, object] = {
        "route": plan.route,
        "turn_lifecycle": {
            "active_flow": plan.active_flow,
            "action": plan.active_flow_action,
        },
        "memory_reference": {"mode": plan.memory_reference_mode},
        "diagnostics": diagnostics,
    }
    active_flow_delta = dict(plan.active_flow_delta)
    active_flow_memory_delta = active_flow_delta.pop("memory_control", None)
    update.update(active_flow_delta)

    memory_control: dict[str, object | None] = {"action": memory_action}
    if isinstance(active_flow_memory_delta, dict):
        memory_control.update(active_flow_memory_delta)

    if plan.route == "grounded_lookup":
        update["grounded_lookup"] = {
            "query": plan.grounded_lookup_query or "",
            "status": "not_attempted",
        }
        update["memory_control"] = memory_control
        return update

    update["grounded_lookup"] = {"query": "", "status": "not_attempted"}
    update["memory_control"] = memory_control
    return update


async def run_turn_dispatch_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[TurnDispatchNextNode]:
    """Route a safe user turn to the next lifecycle node.

    Args:
        state (AgentState): Current graph state after crisis classification.
        runtime (Runtime[WorkflowContext]): LangGraph runtime carrying workflow
            dependencies.

    Returns:
        Command[TurnDispatchNextNode]: State update plus the next node to run.
    """

    start = time.monotonic()
    plan = await plan_turn_route(
        state,
        llm_client=runtime.context.llm_client,
    )
    update = _dispatch_update(
        state,
        plan,
        duration_ms=elapsed_ms(start),
    )

    if plan.route == "memory_control":
        next_node: TurnDispatchNextNode = MEMORY_CONTROL_NODE
    elif plan.route == "grounded_lookup":
        next_node = GROUNDED_ANSWER_NODE
    else:
        next_node = LOAD_MEMORY_NODE

    return Command(update=update, goto=next_node)
