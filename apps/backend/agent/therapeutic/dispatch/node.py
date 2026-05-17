"""Therapeutic dispatch adapter and shared update builder.

The dispatcher is LLM-primary. Sibling modules own the pieces: ``planner``
holds the framework-agnostic policy that returns a ``DispatchPlan``, ``prompt``
builds the classifier prompts, and ``constants`` holds the style → node-name map. The
public surface is re-exported by ``agent.therapeutic.dispatch``.

Boundary invariant:
``response_style`` is the routing axis and maps to the subgraph node.
``therapeutic_approach`` is prompt context; it shapes how the selected
node responds but must not choose the node. The only special handling is
active guided-exercise continuity, where the pinned exercise approach is
reused when the existing exercise route continues or clarifies.
"""

from __future__ import annotations

from typing import Any

from agent.observability.routing_trace import append_routing_trace
from agent.runtime.command import RuntimeCommand
from agent.state import AgentState
from agent.therapeutic.dispatch.constants import (
    TherapeuticNodeName,
    node_for_response_style,
)
from agent.therapeutic.dispatch.planner import DispatchPlan, plan_therapeutic_route
from agent.therapeutic.exercises.state import clear_exercise_delta

_CLOSING_SESSION_ACTION = "suggest_end_session"


def _style_update(plan: DispatchPlan) -> dict:
    """Build the selected-style state delta.

    Args:
        plan: Therapeutic dispatch plan selected for this turn.

    Returns:
        State delta carrying routing, session-arc, and response-guidance fields.
    """

    update = {
        "response_style": plan.response_style,
        "therapeutic_approach": plan.therapeutic_approach,
    }
    session_progress = {}
    if plan.session_intent:
        session_progress["session_intent"] = plan.session_intent
    if plan.session_stage:
        session_progress["session_stage"] = plan.session_stage
    if plan.guidance_permission:
        session_progress["guidance_permission"] = plan.guidance_permission
    if session_progress:
        update["session_progress"] = session_progress
    if plan.response_guidance.strip():
        update["response_guidance"] = plan.response_guidance.strip()
    if plan.response_style == "closing":
        update["session_action"] = _CLOSING_SESSION_ACTION
    return update


def build_therapeutic_dispatch_update(
    state: AgentState,
    plan: DispatchPlan,
) -> dict:
    """Build the state delta for a therapeutic dispatch plan.

    Text runtimes use this to share the dispatch policy without depending on
    routing adapter objects.
    """

    update = (
        {
            **_style_update(plan),
            **clear_exercise_delta(state),
        }
        if plan.clear_exercise
        else _style_update(plan)
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
                "exercise_start_basis": plan.exercise_start_basis,
                "session_intent": plan.session_intent,
                "session_stage": plan.session_stage,
                "guidance_permission": plan.guidance_permission,
            },
        )
    return update


def _to_command(
    state: AgentState,
    plan: DispatchPlan,
) -> RuntimeCommand[TherapeuticNodeName]:
    """Convert a dispatch plan into a compatibility routing command.

    Args:
        state: Current graph state.
        plan: Internal routing plan.

    Returns:
        Routing command for the planned response-style node.
    """

    update = build_therapeutic_dispatch_update(state, plan)
    return RuntimeCommand(
        update=update,
        goto=node_for_response_style(plan.response_style),
    )


async def run_therapeutic_dispatch_node(
    state: AgentState,
    runtime: Any,
) -> RuntimeCommand[TherapeuticNodeName]:
    """Route the current turn to the correct therapeutic response node.

    Args:
        state: The current agent state.
        runtime: The runtime object carrying injected dependencies.

    Returns:
        A command pointing at the next therapeutic response node, with any
        required routing or exercise-state updates.
    """

    plan = await plan_therapeutic_route(state, runtime.context.llm_client)
    return _to_command(state, plan)
