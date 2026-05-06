"""Therapeutic dispatch node — the routing entry point for the subgraph.

The dispatcher uses an LLM-primary classifier with narrow deterministic
guards. Sibling modules in this package own the pieces: ``classifier`` for
the LLM call, ``guards`` for the pre-LLM rule engine, ``router`` for plan
composition, ``fallback`` for the deterministic fallback, ``prompt`` for
prompt construction, and ``regex_catalog`` for the shared regex
definitions. The public surface (the entry point and the constants
callers route on) is re-exported by ``agent.therapeutic.dispatch``.

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

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.dispatch.constants import (
    TherapeuticNodeName,
    _RESPONSE_STYLE_NODE_MAP,
)
from agent.therapeutic.dispatch.router import DispatchPlan, plan_therapeutic_route


def _routing_update(response_style: str, approach: str) -> dict:
    """Build the therapeutic routing state delta.

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


def _clear_active_exercise_update(response_style: str, approach: str) -> dict:
    """Build a state delta that clears active guided-exercise state.

    Args:
        response_style: Therapeutic response style selected for this turn.
        approach: Therapeutic approach selected for this turn.

    Returns:
        State delta carrying routing metadata and a cleared exercise state.
    """

    return {
        **_routing_update(response_style, approach),
        "exercise_state": {
            "exercise_type": None,
            "exercise_step": None,
            "exercise_therapeutic_approach": None,
            "exercise_selection_options": None,
        },
    }


def _command_from_plan(plan: DispatchPlan) -> Command[TherapeuticNodeName]:
    """Convert a dispatch plan into the LangGraph command.

    Args:
        plan: Internal routing plan.

    Returns:
        LangGraph command for the planned response-style node.
    """

    update = (
        _clear_active_exercise_update(plan.response_style, plan.therapeutic_approach)
        if plan.clear_exercise
        else _routing_update(plan.response_style, plan.therapeutic_approach)
    )
    return Command(
        update=update,
        goto=_RESPONSE_STYLE_NODE_MAP[plan.response_style],
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
    return _command_from_plan(plan)
