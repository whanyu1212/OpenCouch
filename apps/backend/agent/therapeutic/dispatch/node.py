"""Therapeutic dispatch node — the routing entry point for the subgraph.

The dispatcher is LLM-primary. Sibling modules own the pieces: ``planner``
holds the framework-agnostic policy that returns a ``DispatchPlan``, this
module turns that plan into a LangGraph ``Command``, ``prompt`` builds the
classifier prompts, and ``constants`` holds the style → node-name map. The
public surface is re-exported by ``agent.therapeutic.dispatch``.

Boundary invariant:
``response_style`` is the routing axis and maps to the subgraph node.
``therapeutic_approach`` is prompt context; it shapes how the selected
node responds but must not choose the node. The only special handling is
active guided-exercise continuity, where the pinned exercise approach is
reused when the existing exercise route continues or clarifies.
"""

from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.observability.routing_trace import append_routing_trace
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.dispatch.constants import (
    TherapeuticNodeName,
    node_for_response_style,
)
from agent.therapeutic.dispatch.planner import DispatchPlan, plan_therapeutic_route
from agent.therapeutic.exercises.state import clear_exercise_delta


def _style_update(response_style: str, approach: str) -> dict:
    """Build the selected-style state delta.

    Args:
        response_style: Therapeutic response style selected for this turn.
        approach: Therapeutic approach selected for this turn.

    Returns:
        State delta carrying the selected response style and therapeutic approach.
    """

    return {
        "response_style": response_style,
        "therapeutic_approach": approach,
    }


def _to_command(
    state: AgentState,
    plan: DispatchPlan,
) -> Command[TherapeuticNodeName]:
    """Convert a dispatch plan into the LangGraph command.

    Args:
        state: Current graph state.
        plan: Internal routing plan.

    Returns:
        LangGraph command for the planned response-style node.
    """

    update = (
        {
            **_style_update(plan.response_style, plan.therapeutic_approach),
            **clear_exercise_delta(state),
        }
        if plan.clear_exercise
        else _style_update(plan.response_style, plan.therapeutic_approach)
    )
    if "diagnostics" in state:
        decision = plan.response_style
        if plan.therapeutic_approach and plan.therapeutic_approach != "none":
            decision = f"{decision}/{plan.therapeutic_approach}"
        update["diagnostics"] = append_routing_trace(
            state.get("diagnostics"),
            {
                "stage": "dispatch",
                "decision": decision,
                "source": plan.source,
                "reason": plan.reason,
                "confidence": plan.confidence,
            },
        )
    return Command(
        update=update,
        goto=node_for_response_style(plan.response_style),
    )


async def run_therapeutic_dispatch_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[TherapeuticNodeName]:
    """Route the current turn to the correct therapeutic response node.

    Args:
        state: The current agent state.
        runtime: The LangGraph runtime carrying injected dependencies.

    Returns:
        A ``Command`` pointing at the next therapeutic response node, with any
        required routing or exercise-state updates.
    """

    plan = await plan_therapeutic_route(state, runtime.context.llm_client)
    return _to_command(state, plan)
