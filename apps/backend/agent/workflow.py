"""LangGraph workflow assembly for the OpenCouch agent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict

from agent.graph import _run_turn_events
from agent.state import AgentState
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from services.llm.base import BaseLLMClient


class WorkflowContext(TypedDict):
    """Runtime-only dependencies injected into the LangGraph workflow."""

    llm_client: BaseLLMClient | None


def _diff_state_updates(before: AgentState, after: AgentState) -> dict[str, Any]:
    """Return only the keys changed by a node execution."""

    return {
        key: value
        for key, value in after.items()
        if key not in before or before[key] != value
    }


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

    async def run_turn_node(
        state: AgentState,
        runtime: Runtime[WorkflowContext],
    ) -> dict[str, Any]:
        """Run the full turn pipeline inside the LangGraph workflow.

        Args:
            state: Shared workflow state for the current turn.
            runtime: LangGraph runtime carrying invocation-scoped dependencies.

        Returns:
            A partial state update after the turn completes.
        """

        before = deepcopy(state)
        final_states: list[AgentState] = []
        async for _ in _run_turn_events(
            deepcopy(state),
            llm_client=runtime.context.get("llm_client"),
            prepare_state=True,
            state_sink=final_states,
        ):
            pass
        if not final_states:
            raise RuntimeError("Turn runner completed without producing a final state.")
        return _diff_state_updates(before, final_states[-1])

    def compact_persisted_state_node(state: AgentState) -> dict[str, Any]:
        """Strip derived prompt-window fields before checkpoint persistence.

        Args:
            state: Shared workflow state after the turn has been finalized.

        Returns:
            A partial update removing derived `history` persistence.
        """

        return {"history": [], "semantic_signals": {}}

    workflow.add_node("run_turn", run_turn_node)
    workflow.add_node("compact_persisted_state", compact_persisted_state_node)

    workflow.add_edge(START, "run_turn")
    workflow.add_edge("run_turn", "compact_persisted_state")
    workflow.add_edge("compact_persisted_state", END)

    return workflow.compile(checkpointer=checkpointer)
