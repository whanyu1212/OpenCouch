"""LangGraph workflow and entrypoints for the OpenCouch agent.

This module owns the entire agent execution surface in three layers:

* **State plumbing** — :func:`build_initial_state` and :func:`state_to_output`
  convert between the public ``AgentInput`` / ``AgentOutput`` contract and the
  internal grouped :class:`AgentState` shape used by the graph nodes.
* **Graph assembly** — :func:`build_agent_workflow` constructs and compiles the
  LangGraph ``StateGraph`` that wires ``load_memory -> crisis_gate ->
  (crisis_response | END)`` together. Crisis-gate routing is encoded directly
  in the node via :class:`langgraph.types.Command`, not via a conditional edge.
* **Public entrypoints** — :func:`run_agent` is a one-shot convenience wrapper
  that compiles a fresh workflow, invokes it with sensible defaults, and
  returns a normalized ``AgentOutput``. For thread-persistent execution see
  :class:`agent.persistence.PersistentAgentRuntime`.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.memory_graph import GraphMemoryStore, NullGraphMemoryStore
from agent.memory_profile import SqliteProfileMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    CrisisAssessment,
    ModeType,
    ResponseKind,
)
from agent.nodes.crisis_gate import run_crisis_gate_node
from agent.nodes.crisis_response import run_crisis_response_node
from agent.nodes.load_memory import run_load_memory_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from services.llm.base import BaseLLMClient


# ── State plumbing ───────────────────────────────────────────────────────────


def build_initial_state(agent_input: AgentInput) -> AgentState:
    """Convert external input into the internal state dictionary."""

    transcript = [message.model_dump(mode="json") for message in agent_input.history]
    state = AgentState(
        message=agent_input.message,
        channel=agent_input.channel,
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        installed_skills=list(agent_input.installed_skills),
        history=list(transcript),
        transcript=list(transcript),
        working_memory=list(agent_input.working_memory),
        memory={
            "summary": "",
            "active_concerns": [],
            "open_loops": [],
            "current_goal": None,
        },
        progress={
            "intent": None,
            "intent_source": None,
            "stage": "opening",
            "stage_source": "deterministic",
            "stage_reason": "fresh_graph_memory_scaffold",
            "turn_count": sum(1 for turn in transcript if turn.get("role") == "user")
            + 1,
            "is_guest": False,
        },
        crisis=CrisisAssessment(),
        routing={
            "route": "load_memory_only",
            "mode": "memory_bootstrap",
            "mode_source": "graph_bootstrap",
            "mode_type": ModeType.OPERATIONAL,
            "active_modalities": [],
            "semantic_signals": {},
        },
        response={
            "guidance": "startup_memory_bootstrap",
            "kind": ResponseKind.THERAPEUTIC,
            "text": "",
            "should_persist_memory": False,
        },
    )
    return state


def state_to_output(state: AgentState) -> AgentOutput:
    """Normalize graph state into the public agent output contract."""

    response_state = state.get("response", {})
    routing_state = state.get("routing", {})

    return AgentOutput(
        response_text=response_state.get("text", ""),
        response_type=response_state.get("kind", ResponseKind.THERAPEUTIC),
        crisis=state.get("crisis", CrisisAssessment()),
        mode=routing_state.get("mode"),
        mode_type=routing_state.get("mode_type"),
        mode_source=routing_state.get("mode_source"),
        should_persist_memory=response_state.get("should_persist_memory", False),
    )


# ── Graph assembly ───────────────────────────────────────────────────────────


def build_agent_workflow(
    *,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the LangGraph workflow.

    Args:
        checkpointer: Optional LangGraph checkpointer for thread persistence.

    Returns:
        A compiled LangGraph workflow ready to invoke.
    """

    workflow = StateGraph(AgentState, context_schema=WorkflowContext)

    workflow.add_node("load_memory_node", run_load_memory_node)
    workflow.add_node("crisis_gate_node", run_crisis_gate_node)
    workflow.add_node("crisis_response_node", run_crisis_response_node)

    workflow.add_edge(START, "load_memory_node")
    workflow.add_edge("load_memory_node", "crisis_gate_node")
    # crisis_gate_node returns Command(update=..., goto=...) so its routing
    # decision is encoded in the node itself — no conditional edge needed.
    # The static edge below still fires when crisis_response_node completes.
    workflow.add_edge("crisis_response_node", END)

    return workflow.compile(checkpointer=checkpointer)


# ── Public entrypoints ───────────────────────────────────────────────────────


async def run_agent(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
    profile_memory_store: SqliteProfileMemoryStore | None = None,
    graph_memory_store: GraphMemoryStore | None = None,
    is_guest_mode: bool = True,
) -> AgentOutput:
    """Run the full compiled agent workflow end-to-end for one turn.

    Convenience entrypoint for callers (CLI scripts, tests, evals) that just
    want a one-shot ``AgentInput -> AgentOutput`` invocation. The compiled
    workflow handles routing through ``load_memory -> crisis_gate ->
    (crisis_response | END)``.

    For thread-persistent execution with checkpointed state and a long-lived
    compiled graph, use :class:`agent.persistence.PersistentAgentRuntime`
    instead — that path reuses one workflow across many turns.

    Args:
        agent_input: The user message and conversation context for this turn.
        llm_client: Optional provider client used for LLM-backed gate and
            response nodes. When ``None`` the graph falls back to deterministic
            behavior.
        profile_memory_store: Optional profile-memory store. Defaults to a
            stub :class:`SqliteProfileMemoryStore` rooted at an in-memory path
            so one-shot calls do not require disk I/O.
        graph_memory_store: Optional graph-memory store. Defaults to
            :class:`NullGraphMemoryStore`.
        is_guest_mode: Whether the runtime should treat the session as
            ephemeral (no long-term memory). Defaults to ``True`` so one-shot
            calls do not accidentally pollute persistent stores.
    """

    workflow = build_agent_workflow()
    profile_store = profile_memory_store or SqliteProfileMemoryStore(":memory:")
    graph_store = graph_memory_store or NullGraphMemoryStore()

    final_state = await workflow.ainvoke(
        build_initial_state(agent_input),
        context={
            "llm_client": llm_client,
            "profile_memory_store": profile_store,
            "graph_memory_store": graph_store,
            "is_guest_mode": is_guest_mode,
        },
    )
    return state_to_output(final_state)
