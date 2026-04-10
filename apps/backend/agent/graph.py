"""LangGraph workflow and entrypoints for the OpenCouch agent.

This module owns the entire agent execution surface in three layers:

* **State plumbing** — :func:`build_initial_state` and :func:`state_to_output`
  convert between the public ``AgentInput`` / ``AgentOutput`` contract and the
  internal grouped :class:`AgentState` shape used by the graph nodes.
* **Graph assembly** — :func:`build_agent_workflow` constructs and compiles the
  LangGraph ``StateGraph`` that wires the full topology: load_memory →
  crisis_gate → (crisis_response + crisis_log | therapeutic_subgraph) →
  extract_semantic_facts → finalize_turn → END. Crisis-gate routing is
  encoded directly in the node via :class:`langgraph.types.Command`, not
  via a conditional edge. The terminal ``finalize_turn`` node appends the
  assistant response to the transcript so the next turn's ``get_history``
  call sees both sides of each exchange.
* **Public entrypoints** — :func:`run_agent` is a one-shot convenience wrapper
  that compiles a fresh workflow, invokes it with sensible defaults, and
  returns a normalized ``AgentOutput``. For thread-persistent execution see
  :class:`agent.persistence.PersistentAgentRuntime`.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.memory.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    CrisisAssessment,
    MessageRole,
    ModeType,
    ResponseKind,
)
from agent.nodes.crisis_gate import run_crisis_gate_node
from agent.nodes.crisis_log import run_crisis_log_node
from agent.nodes.crisis_response import run_crisis_response_node
from agent.nodes.extract_facts import run_extract_semantic_facts_node
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.load_memory import run_load_memory_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.graph import build_therapeutic_subgraph
from services.llm.base import BaseLLMClient


# ── State plumbing ───────────────────────────────────────────────────────────


def build_initial_state(agent_input: AgentInput) -> AgentState:
    """Convert external input into the internal state dictionary.

    Transcript handling: the prior exchanges from ``agent_input.history``
    are serialized into the transcript, and then the **current user
    message** is appended so downstream nodes see a transcript ending
    in the user's latest turn. The assistant side is appended later by
    :func:`agent.nodes.finalize_turn.run_finalize_turn_node` once the
    response is ready. This split (user side at init, assistant side
    at finalize) keeps transcript ownership clear and avoids the
    phantom-assistant-turn bug that the pre-refactor ``load_memory_node``
    introduced by trying to append both halves during a single step.

    Routing and response scaffolds are left as empty/placeholder values
    that the dispatcher and response nodes overwrite. They used to carry
    misleading labels like ``memory_bootstrap`` and
    ``startup_memory_bootstrap``, which showed up in CLI diagnostics
    every turn even though they were never real states the graph
    actually occupied. The current labels describe the genuine
    pre-dispatch state: routing unresolved, no response generated yet.
    """

    prior_transcript = [
        message.model_dump(mode="json") for message in agent_input.history
    ]
    current_user_turn = {
        "role": MessageRole.USER.value,
        "content": agent_input.message,
    }
    transcript_with_user = [*prior_transcript, current_user_turn]

    state = AgentState(
        message=agent_input.message,
        channel=agent_input.channel,
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        installed_skills=list(agent_input.installed_skills),
        history=list(transcript_with_user),
        transcript=list(transcript_with_user),
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
            "stage_reason": "turn_start",
            "turn_count": sum(
                1 for turn in transcript_with_user if turn.get("role") == "user"
            ),
            "is_guest": False,
        },
        crisis=CrisisAssessment(),
        routing={
            "route": "pending",
            "mode": "pending",
            "mode_source": "graph_bootstrap",
            "mode_type": ModeType.OPERATIONAL,
            "active_modalities": [],
            "semantic_signals": {},
        },
        response={
            "guidance": "pending",
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

    Topology::

        START
          → load_memory_node
          → crisis_gate_node  (Command routes to one of the branches)
          ├─ crisis_response_node → crisis_log_node → extract_semantic_facts_node
          │                                             → finalize_turn_node → END
          └─ therapeutic_subgraph → extract_semantic_facts_node
                                      → finalize_turn_node → END

    The therapeutic subgraph is embedded as a single node compiled by
    :func:`agent.therapeutic.graph.build_therapeutic_subgraph`. Its
    internal structure (dispatcher + three mode nodes) is hidden from
    the parent topology, keeping the top-level graph small and
    inspectable in LangSmith.

    ``finalize_turn_node`` is a small terminal step that appends the
    assistant response to the transcript before END. It runs on both
    branches via the converge-before-END structure, and pairs with
    :func:`build_initial_state` (which appends the user message) to
    keep transcript ownership explicit and out of the response nodes.

    Args:
        checkpointer: Optional LangGraph checkpointer for thread persistence.

    Returns:
        A compiled LangGraph workflow ready to invoke.
    """

    workflow = StateGraph(AgentState, context_schema=WorkflowContext)

    # Build the therapeutic subgraph once per workflow compile. The
    # subgraph shares AgentState with the parent, so no wrapper function
    # is needed — LangGraph propagates state and runtime context
    # automatically.
    therapeutic_subgraph = build_therapeutic_subgraph()

    # Register all top-level nodes.
    workflow.add_node("load_memory_node", run_load_memory_node)
    workflow.add_node("crisis_gate_node", run_crisis_gate_node)
    workflow.add_node("crisis_response_node", run_crisis_response_node)
    workflow.add_node("crisis_log_node", run_crisis_log_node)
    workflow.add_node("therapeutic_subgraph", therapeutic_subgraph)
    workflow.add_node("extract_semantic_facts_node", run_extract_semantic_facts_node)
    workflow.add_node("finalize_turn_node", run_finalize_turn_node)

    # Spine: entry → memory load → crisis gate (which Command-routes).
    workflow.add_edge(START, "load_memory_node")
    workflow.add_edge("load_memory_node", "crisis_gate_node")

    # Crisis branch: crisis_response → crisis_log → extract_facts → finalize → END.
    # The crisis_log_node runs ONLY on the crisis branch (it's the
    # always-on safety audit trail, not a cross-cutting concern).
    workflow.add_edge("crisis_response_node", "crisis_log_node")
    workflow.add_edge("crisis_log_node", "extract_semantic_facts_node")

    # Therapeutic branch: subgraph → extract_facts → finalize → END.
    workflow.add_edge("therapeutic_subgraph", "extract_semantic_facts_node")

    # Shared terminal: extract_facts → finalize_turn → END. Both branches
    # converge through extract_facts (for semantic memory writes) and
    # then through finalize_turn (for transcript append).
    workflow.add_edge("extract_semantic_facts_node", "finalize_turn_node")
    workflow.add_edge("finalize_turn_node", END)

    return workflow.compile(checkpointer=checkpointer)


# ── Public entrypoints ───────────────────────────────────────────────────────


async def run_agent(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
    memory_store: MemoryStore | None = None,
    crisis_log_backend: CrisisLogBackend | None = None,
    memory_mode: MemoryMode = MemoryMode.INCOGNITO,
) -> AgentOutput:
    """Run the full compiled agent workflow end-to-end for one turn.

    Convenience entrypoint for callers (CLI scripts, tests, evals) that just
    want a one-shot ``AgentInput -> AgentOutput`` invocation. The compiled
    workflow handles routing through ``load_memory -> crisis_gate ->
    (crisis_response + crisis_log | therapeutic_subgraph) ->
    extract_semantic_facts -> END``.

    For thread-persistent execution with checkpointed state and a long-lived
    compiled graph, use :class:`agent.persistence.PersistentAgentRuntime`
    instead — that path reuses one workflow across many turns.

    Args:
        agent_input: The user message and conversation context for this turn.
        llm_client: Optional provider client used for LLM-backed gate and
            response nodes. When ``None`` the graph falls back to deterministic
            behavior.
        memory_store: Optional unified memory store. Defaults to a fresh
            in-memory :class:`OpenCouchMemoryStore`.
        crisis_log_backend: Optional crisis log backend. Defaults to
            :class:`InMemoryCrisisLogBackend`. Always-on regardless of
            memory_mode.
        memory_mode: Persistence tier for this turn. Defaults to
            :attr:`MemoryMode.INCOGNITO` so one-shot calls do not accidentally
            pollute persistent stores.
    """

    workflow = build_agent_workflow()
    store = memory_store or OpenCouchMemoryStore()
    crisis_log = crisis_log_backend or InMemoryCrisisLogBackend()

    final_state = await workflow.ainvoke(
        build_initial_state(agent_input),
        context={
            "llm_client": llm_client,
            "memory_store": store,
            "crisis_log_backend": crisis_log,
            "memory_mode": memory_mode,
        },
    )
    return state_to_output(final_state)
