"""Build the top-level LangGraph workflow for the OpenCouch agent.

This module owns graph assembly, state conversion, and the one-shot execution
path. Thread persistence, runtime-owned stores, and active-session recovery live
in ``agent.persistence``.

Responsibilities:
    State plumbing converts between ``AgentInput`` / ``AgentOutput`` and the
    split LangGraph schemas: ``AgentGraphInputState``, ``AgentState``, and
    ``AgentGraphOutputState``.

    Graph assembly compiles a safety-first ``StateGraph``. ``crisis_gate_node``
    routes with ``Command`` to either the crisis branch or the therapeutic
    branch. Both branches converge at ``finalize_turn_node`` and then END.

    Memory extraction (semantic + procedural) is *not* a graph node. It runs
    as a background task dispatched by the runtime layer
    (:class:`agent.persistence.PersistentAgentRuntime`) after the graph
    terminates, so the user-visible turn does not wait on extraction LLM
    calls. The one-shot ``run_agent`` entrypoint runs extraction
    synchronously after ``ainvoke`` so that callers without a runtime still
    get the side effects they expect.

    The public entrypoint, ``run_agent``, compiles a fresh workflow with
    in-memory defaults for callers that do not need thread-persistent behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agent.audit.crisis_log import CrisisLogBackend, InMemoryCrisisLogBackend
from agent.graph_constants import (
    CRISIS_GATE_NODE,
    CRISIS_LOG_NODE,
    CRISIS_RESOURCE_LOOKUP_NODE,
    CRISIS_RESPONSE_NODE,
    FINALIZE_TURN_NODE,
    GROUNDED_ANSWER_NODE,
    GROUNDED_LOOKUP_GATE_NODE,
    LOAD_MEMORY_NODE,
    MEMORY_CONTROL_GATE_NODE,
    MEMORY_CONTROL_NODE,
    THERAPEUTIC_SUBGRAPH_NODE,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore, OpenCouchMemoryStore
from agent.models import (
    AgentInput,
    AgentOutput,
    CrisisAssessment,
    MessageRole,
    ResponseStyleType,
    ResponseCategory,
)
from agent.memory.extraction_service import (
    extract_procedural_rules,
    extract_semantic_facts,
)
from agent.nodes.crisis_gate import run_crisis_gate_node
from agent.nodes.crisis_log import run_crisis_log_node
from agent.nodes.crisis_resource_lookup import run_crisis_resource_lookup_node
from agent.nodes.crisis_response import run_crisis_response_node
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.grounded_answer import run_grounded_answer_node
from agent.nodes.grounded_lookup_gate import run_grounded_lookup_gate_node
from agent.nodes.load_memory import run_load_memory_node
from agent.nodes.memory_control import run_memory_control_node
from agent.nodes.memory_control_gate import run_memory_control_gate_node
from agent.observability.tracing import apply_graph_tracing
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentGraphOutputState, AgentState
from agent.therapeutic.graph import build_therapeutic_subgraph
from services.llm.base import BaseLLMClient


# ── State plumbing ───────────────────────────────────────────────────────────


def build_initial_state(
    agent_input: AgentInput,
    *,
    prior_turn_count: int | None = None,
    include_input_history: bool = False,
) -> AgentGraphInputState:
    """Convert public input into the internal graph state.

    Args:
        agent_input: The external turn input to seed into graph state.
        prior_turn_count: Optional persisted user-turn count from the prior
            checkpoint. When omitted, the function derives the count from
            ``agent_input.history``.
        include_input_history: Whether to inline ``agent_input.history`` into
            ``transcript`` for one-shot callers without a checkpointer.

    Returns:
        An ``AgentGraphInputState`` seeded for the current turn.
    """

    current_user_turn = {
        "role": MessageRole.USER.value,
        "content": agent_input.message,
    }
    prior_history_turns = [
        message.model_dump(mode="json") for message in agent_input.history
    ]
    visible_history = (
        [*prior_history_turns, current_user_turn]
        if include_input_history
        else [current_user_turn]
    )

    if prior_turn_count is None:
        prior_user_turns = sum(
            1
            for msg in agent_input.history
            if (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            )
            == "user"
        )
        turn_count = prior_user_turns + 1
    else:
        turn_count = prior_turn_count + 1

    state = AgentGraphInputState(
        message=agent_input.message,
        channel=agent_input.channel,
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        installed_skills=list(agent_input.installed_skills),
        transcript=visible_history,
        working_memory=list(agent_input.working_memory),
        session_memory={"summary": ""},
        procedural_profile={
            "procedural_rules": [],
            "proactive_recall_enabled": False,
        },
        session_progress={
            "turn_count": turn_count,
            "is_guest": False,
        },
        exercise_state={},
        memory_control={},
        crisis=CrisisAssessment(),
        therapeutic_approach=None,
        response_style="pending",
        response_style_source=None,
        response_style_type=ResponseStyleType.THERAPEUTIC,
        response_kind=ResponseCategory.THERAPEUTIC,
        response_text="",
        should_persist_memory=False,
        diagnostics={},
        route="",
        crisis_audit={},
        grounded_lookup={"query": "", "status": "not_attempted"},
        inferred_location="",
        found_resources=[],
        resource_lookup_status="not_attempted",
    )
    return state


def state_to_output(state: Mapping[str, Any]) -> AgentOutput:
    """Normalize graph state into the public output contract.

    Args:
        state: The final graph state or graph output payload for the turn.

    Returns:
        An ``AgentOutput`` built from the relevant response and routing fields.
    """

    return AgentOutput(
        response_text=state.get("response_text", ""),
        response_type=state.get("response_kind", ResponseCategory.THERAPEUTIC),
        crisis=state.get("crisis", CrisisAssessment()),
        response_style=state.get("response_style"),
        response_style_type=state.get("response_style_type"),
        response_style_source=state.get("response_style_source"),
        therapeutic_approach=state.get("therapeutic_approach"),
        should_persist_memory=state.get("should_persist_memory", False),
        diagnostics=dict(state.get("diagnostics", {})),
    )


# ── Graph assembly ───────────────────────────────────────────────────────────


def build_agent_workflow(
    *,
    checkpointer: Any | None = None,
) -> CompiledStateGraph[
    AgentState,
    WorkflowContext,
    AgentGraphInputState,
    AgentGraphOutputState,
]:
    """Compile the top-level LangGraph workflow.

    Topology::

        START
          → crisis_gate_node  (Command routes to one of the branches)
             ├─ crisis_resource_lookup_node → crisis_response_node
             │  → crisis_log_node → finalize_turn_node
             └─ memory_control_gate_node
                ├─ memory_control_node → finalize_turn_node
                └─ grounded_lookup_gate_node
                   ├─ grounded_answer_node → finalize_turn_node
                   └─ load_memory_node → therapeutic_subgraph
                      → finalize_turn_node

        finalize_turn_node → END

    Memory extraction (semantic facts + procedural rules) is run by the
    *runtime*, not the graph. ``PersistentAgentRuntime`` schedules
    extraction as a background ``asyncio.Task`` after each ``run_turn``;
    ``run_agent`` runs it synchronously after ``ainvoke``. Either way, the
    graph's responsibility ends at ``finalize_turn_node`` so the
    user-visible turn does not wait on extractor LLM calls.

    Args:
        checkpointer: Optional LangGraph checkpointer for thread persistence.

    Returns:
        A compiled LangGraph workflow ready to invoke.
    """

    workflow = StateGraph(
        AgentState,
        context_schema=WorkflowContext,
        input_schema=AgentGraphInputState,
        output_schema=AgentGraphOutputState,
    )

    therapeutic_subgraph = build_therapeutic_subgraph()

    # Shared retry policy for the top-level I/O nodes.
    _io_retry = RetryPolicy(max_attempts=2)

    workflow.add_node(LOAD_MEMORY_NODE, run_load_memory_node, retry_policy=_io_retry)
    workflow.add_node(CRISIS_GATE_NODE, run_crisis_gate_node, retry_policy=_io_retry)
    workflow.add_node(
        MEMORY_CONTROL_GATE_NODE,
        run_memory_control_gate_node,
        retry_policy=_io_retry,
    )
    workflow.add_node(
        MEMORY_CONTROL_NODE,
        run_memory_control_node,
        retry_policy=_io_retry,
    )
    workflow.add_node(
        GROUNDED_LOOKUP_GATE_NODE,
        run_grounded_lookup_gate_node,
        retry_policy=_io_retry,
    )
    workflow.add_node(
        GROUNDED_ANSWER_NODE,
        run_grounded_answer_node,
        retry_policy=_io_retry,
    )
    workflow.add_node(
        CRISIS_RESOURCE_LOOKUP_NODE,
        run_crisis_resource_lookup_node,
        retry_policy=_io_retry,
    )
    workflow.add_node(
        CRISIS_RESPONSE_NODE, run_crisis_response_node, retry_policy=_io_retry
    )
    workflow.add_node(CRISIS_LOG_NODE, run_crisis_log_node, retry_policy=_io_retry)
    workflow.add_node(THERAPEUTIC_SUBGRAPH_NODE, therapeutic_subgraph)
    workflow.add_node(FINALIZE_TURN_NODE, run_finalize_turn_node)

    # Safety-first entry.
    workflow.add_edge(START, CRISIS_GATE_NODE)

    # Crisis branch skips memory load.
    workflow.add_edge(CRISIS_RESOURCE_LOOKUP_NODE, CRISIS_RESPONSE_NODE)
    workflow.add_edge(CRISIS_RESPONSE_NODE, CRISIS_LOG_NODE)
    workflow.add_edge(CRISIS_LOG_NODE, FINALIZE_TURN_NODE)

    workflow.add_edge(MEMORY_CONTROL_NODE, FINALIZE_TURN_NODE)
    workflow.add_edge(GROUNDED_ANSWER_NODE, FINALIZE_TURN_NODE)
    workflow.add_edge(LOAD_MEMORY_NODE, THERAPEUTIC_SUBGRAPH_NODE)
    workflow.add_edge(THERAPEUTIC_SUBGRAPH_NODE, FINALIZE_TURN_NODE)

    # Memory extraction is dispatched by the runtime after the graph
    # terminates — see ``PersistentAgentRuntime._schedule_extraction``
    # and ``agent.graph.run_agent``.
    workflow.add_edge(FINALIZE_TURN_NODE, END)

    compiled = workflow.compile(checkpointer=checkpointer)
    return apply_graph_tracing(compiled)


# ── Public entrypoints ───────────────────────────────────────────────────────


async def run_agent(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
    memory_store: MemoryStore | None = None,
    crisis_log_backend: CrisisLogBackend | None = None,
    memory_mode: MemoryMode = MemoryMode.INCOGNITO,
) -> AgentOutput:
    """Run one turn through a fresh compiled workflow.

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

    Returns:
        The normalized ``AgentOutput`` for the completed turn.
    """

    workflow = build_agent_workflow()
    store = memory_store or OpenCouchMemoryStore()
    crisis_log = crisis_log_backend or InMemoryCrisisLogBackend()

    initial_state = build_initial_state(agent_input, include_input_history=True)
    final_state = await workflow.ainvoke(
        initial_state,
        context=WorkflowContext(
            llm_client=llm_client,
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=memory_mode,
        ),
    )

    # ``run_agent`` is the one-shot entrypoint used by tests and eval
    # runners — it has no thread lock, no per-thread drain, and no
    # ``__aexit__`` shutdown drain. Running extraction synchronously
    # here preserves the contract that callers without a runtime see
    # extraction's side effects before this function returns.
    # Merge initial_state (carries ``message`` and other input-only fields
    # that fall out of the public output schema) with final_state. The
    # graph's output schema strips private channels including ``route``,
    # so derive it from the public ``crisis`` assessment to preserve the
    # crisis-skip path in extraction policy.
    extraction_state = {**dict(initial_state), **dict(final_state)}
    crisis_assessment = final_state.get("crisis")
    if crisis_assessment is not None and getattr(crisis_assessment, "level", 0) >= 2:
        extraction_state["route"] = "crisis"
    extraction_diagnostics = await _run_extraction_synchronously(
        extraction_state,
        llm_client=llm_client,
        memory_store=store,
        memory_mode=memory_mode,
    )
    if extraction_diagnostics:
        merged_diag = {
            **dict(final_state.get("diagnostics", {})),
            **extraction_diagnostics,
        }
        if isinstance(final_state, dict):
            final_state = {**final_state, "diagnostics": merged_diag}
        else:
            final_state = {**dict(final_state), "diagnostics": merged_diag}
    return state_to_output(final_state)


async def _run_extraction_synchronously(
    state: Mapping[str, Any],
    *,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore,
    memory_mode: MemoryMode,
) -> dict[str, Any]:
    """Run both extractors sequentially and merge their diagnostics.

    Used by ``run_agent`` (no runtime). The runtime path
    (:meth:`PersistentAgentRuntime.run_turn`) dispatches extraction as a
    background task instead — this function exists so that one-shot
    callers without a runtime still see extraction side effects before
    they receive their ``AgentOutput``.

    Args:
        state: The terminal graph state for the turn.
        llm_client: Optional control-plane LLM. ``None`` skips extraction.
        memory_store: Memory store to receive writes.
        memory_mode: Persistence tier for the turn.

    Returns:
        Merged diagnostics dict from both extractors. Empty when
        extraction was skipped (e.g., ``MemoryMode.INCOGNITO``).
    """

    import asyncio

    semantic_outcome, procedural_outcome = await asyncio.gather(
        extract_semantic_facts(
            state,  # type: ignore[arg-type]
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
            embedding_provider=None,
            session_buffer=None,
        ),
        extract_procedural_rules(
            state,  # type: ignore[arg-type]
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
            session_buffer=None,
        ),
    )
    return {
        **semantic_outcome.as_diagnostics(),
        **procedural_outcome.as_diagnostics(),
    }
