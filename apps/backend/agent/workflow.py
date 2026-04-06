"""LangGraph workflow assembly for the OpenCouch agent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Awaitable, Callable, TypedDict

from agent.graph import finalize_turn_state, prepare_turn_state
from agent.nodes.crisis_gate import run_crisis_gate
from agent.nodes.session_stage import update_session_stage
from agent.state import AgentState
from agent.subgraphs import run_crisis_subgraph, run_therapeutic_subgraph
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from services.llm.base import BaseLLMClient


class WorkflowContext(TypedDict):
    """Runtime-only dependencies injected into the LangGraph workflow."""

    llm_client: BaseLLMClient | None


def _diff_state_updates(before: AgentState, after: AgentState) -> dict[str, Any]:
    """Return only the keys changed by a node execution.

    Args:
        before: State snapshot before running the node.
        after: State snapshot after running the node.

    Returns:
        A partial state update containing only changed keys.
    """

    updates: dict[str, Any] = {}
    for key, value in after.items():
        if key not in before or before[key] != value:
            updates[key] = value
    return updates


async def _run_async_node_as_update(
    state: AgentState,
    runner: Callable[[AgentState], Awaitable[AgentState]],
) -> dict[str, Any]:
    """Run a legacy async node and convert it into a partial state update.

    Args:
        state: Current workflow state snapshot.
        runner: Async function that returns the updated full state.

    Returns:
        A partial state update for LangGraph.
    """

    before = deepcopy(state)
    after = await runner(deepcopy(state))
    return _diff_state_updates(before, after)


def _run_sync_node_as_update(
    state: AgentState,
    runner: Callable[[AgentState], AgentState],
) -> dict[str, Any]:
    """Run a legacy sync node and convert it into a partial state update.

    Args:
        state: Current workflow state snapshot.
        runner: Sync function that returns the updated full state.

    Returns:
        A partial state update for LangGraph.
    """

    before = deepcopy(state)
    after = runner(deepcopy(state))
    return _diff_state_updates(before, after)


def _route_after_crisis_gate(state: AgentState) -> str:
    """Return the next workflow branch after crisis classification.

    Args:
        state: The shared graph state after the crisis-gate node.

    Returns:
        The next node name for the workflow branch.
    """

    return "crisis_subgraph" if state["route"] == "crisis" else "therapeutic_subgraph"


def build_agent_workflow(
    *,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the LangGraph workflow for the current runtime configuration.

    Args:
        checkpointer: Optional LangGraph checkpointer for thread persistence.

    Returns:
        A compiled LangGraph workflow ready to invoke.
    """

    workflow = StateGraph(AgentState, context_schema=WorkflowContext)

    def prepare_turn_node(state: AgentState) -> dict[str, Any]:
        """Refresh turn-scoped state from the persisted thread before routing.

        Args:
            state: Shared workflow state after applying the latest thread input update.

        Returns:
            A partial update with derived context rebuilt for the current turn.
        """

        return _run_sync_node_as_update(state, prepare_turn_state)

    async def crisis_gate_node(
        state: AgentState,
        runtime: Runtime[WorkflowContext],
    ) -> dict[str, Any]:
        """Run the hybrid crisis gate inside the LangGraph workflow.

        Args:
            state: Shared workflow state for the current turn.
            runtime: LangGraph runtime carrying invocation-scoped dependencies.

        Returns:
            A partial state update after the crisis gate.
        """

        return await _run_async_node_as_update(
            state,
            lambda working_state: run_crisis_gate(
                working_state,
                llm_client=runtime.context.get("llm_client"),
            ),
        )

    async def session_stage_node(
        state: AgentState,
        runtime: Runtime[WorkflowContext],
    ) -> dict[str, Any]:
        """Update the current session stage before response routing.

        Args:
            state: Shared workflow state for the current turn.
            runtime: LangGraph runtime carrying invocation-scoped dependencies.

        Returns:
            A partial state update after session-stage inference.
        """

        return await _run_async_node_as_update(
            state,
            lambda working_state: update_session_stage(
                working_state,
                llm_client=runtime.context.get("llm_client"),
            ),
        )

    async def crisis_subgraph_node(
        state: AgentState,
        runtime: Runtime[WorkflowContext],
    ) -> dict[str, Any]:
        """Run the crisis-response branch inside the workflow.

        Args:
            state: Shared workflow state for the current turn.
            runtime: LangGraph runtime carrying invocation-scoped dependencies.

        Returns:
            A partial state update after the crisis branch completes.
        """

        return await _run_async_node_as_update(
            state,
            lambda working_state: run_crisis_subgraph(
                working_state,
                llm_client=runtime.context.get("llm_client"),
            ),
        )

    async def therapeutic_subgraph_node(
        state: AgentState,
        runtime: Runtime[WorkflowContext],
    ) -> dict[str, Any]:
        """Run the non-crisis branch inside the workflow.

        Args:
            state: Shared workflow state for the current turn.
            runtime: LangGraph runtime carrying invocation-scoped dependencies.

        Returns:
            A partial state update after the therapeutic branch completes.
        """

        return await _run_async_node_as_update(
            state,
            lambda working_state: run_therapeutic_subgraph(
                working_state,
                llm_client=runtime.context.get("llm_client"),
            ),
        )

    def finalize_turn_node(state: AgentState) -> dict[str, Any]:
        """Persist the completed turn into transcript/history state.

        Args:
            state: Shared workflow state after response generation.

        Returns:
            A partial update with the current turn folded into transcript/history.
        """

        return _run_sync_node_as_update(state, finalize_turn_state)

    def compact_persisted_state_node(state: AgentState) -> dict[str, Any]:
        """Strip derived prompt-window fields before checkpoint persistence.

        Args:
            state: Shared workflow state after the turn has been finalized.

        Returns:
            A partial update removing derived `history` persistence.
        """

        return {"history": []}

    workflow.add_node("prepare_turn", prepare_turn_node)
    workflow.add_node("crisis_gate", crisis_gate_node)
    workflow.add_node("session_stage", session_stage_node)
    workflow.add_node("crisis_subgraph", crisis_subgraph_node)
    workflow.add_node("therapeutic_subgraph", therapeutic_subgraph_node)
    workflow.add_node("finalize_turn", finalize_turn_node)
    workflow.add_node("compact_persisted_state", compact_persisted_state_node)

    workflow.add_edge(START, "prepare_turn")
    workflow.add_edge("prepare_turn", "crisis_gate")
    workflow.add_edge("crisis_gate", "session_stage")
    workflow.add_conditional_edges(
        "session_stage",
        _route_after_crisis_gate,
        {
            "crisis_subgraph": "crisis_subgraph",
            "therapeutic_subgraph": "therapeutic_subgraph",
        },
    )
    workflow.add_edge("crisis_subgraph", "finalize_turn")
    workflow.add_edge("therapeutic_subgraph", "finalize_turn")
    workflow.add_edge("finalize_turn", "compact_persisted_state")
    workflow.add_edge("compact_persisted_state", END)

    return workflow.compile(checkpointer=checkpointer)
