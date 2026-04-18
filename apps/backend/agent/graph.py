"""LangGraph workflow and entrypoints for the OpenCouch agent.

This module owns the entire agent execution surface in three layers:

* **State plumbing** — :func:`build_initial_state` and :func:`state_to_output`
  convert between the public ``AgentInput`` / ``AgentOutput`` contract and the
  internal grouped :class:`AgentState` shape used by the graph nodes.
* **Graph assembly** — :func:`build_agent_workflow` constructs and compiles the
  LangGraph ``StateGraph`` that wires the full topology: crisis_gate →
  (crisis_response + crisis_log | load_memory → therapeutic_subgraph) →
  extract_semantic_facts → extract_procedural_rules → finalize_turn → END.
  v0.9 safety reorder: crisis gate runs FIRST so safety-critical routing is
  never blocked by optional memory retrieval. Crisis-gate routing is encoded
  directly in the node via :class:`langgraph.types.Command`, not via a
  conditional edge. The terminal ``finalize_turn`` node appends the assistant
  response to the transcript so the next turn's ``get_history`` call sees both
  sides of each exchange.
* **Public entrypoints** — :func:`run_agent` is a one-shot convenience wrapper
  that compiles a fresh workflow, invokes it with sensible defaults, and
  returns a normalized ``AgentOutput``. For thread-persistent execution see
  :class:`agent.persistence.PersistentAgentRuntime`.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

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
from agent.nodes.extract_procedural_rules import run_extract_procedural_rules_node
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.load_memory import run_load_memory_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.graph import build_therapeutic_subgraph
from services.llm.base import BaseLLMClient


# ── State plumbing ───────────────────────────────────────────────────────────


def build_initial_state(
    agent_input: AgentInput,
    *,
    prior_turn_count: int | None = None,
    include_input_history: bool = False,
) -> AgentState:
    """Convert external input into the internal state dictionary.

    Transcript handling defaults to reducer-backed persistence mode:
    only the **current user message** is emitted into ``history`` and
    ``transcript``. Both fields use an ``operator.add`` reducer, so
    when a checkpointer is active the prior turns are restored from the
    checkpoint and the reducer appends the new user turn automatically.
    The assistant side is appended later by
    :func:`agent.nodes.finalize_turn.run_finalize_turn_node` once the
    response is ready.

    One-shot callers can opt into ``include_input_history=True`` to
    seed ``agent_input.history`` directly into state for classifiers
    and prompt builders that need prior-turn context without a
    checkpointer.

    ``prior_turn_count`` is optional: persistent runtimes can pass the
    previous checkpoint's ``progress.turn_count`` directly and avoid
    reloading the transcript just to count user turns. When omitted,
    the function falls back to counting prior user turns from
    ``agent_input.history``.

    Routing and response scaffolds are left as empty/placeholder values
    that the dispatcher and response nodes overwrite. For persistent
    sessions (``PersistentAgentRuntime``), the checkpointer restores
    prior values of ``progress``, ``routing``, ``response``, etc.
    from the previous turn's checkpoint — those fields are omitted
    from the input on subsequent turns so the checkpoint's values
    are preserved rather than overwritten.
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

    # Compute turn_count from the previous checkpoint when the caller
    # already has it. Falling back to ``agent_input.history`` keeps the
    # one-shot API stable and preserves older tests that still pass
    # prior messages directly.
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

    state = AgentState(
        message=agent_input.message,
        channel=agent_input.channel,
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        installed_skills=list(agent_input.installed_skills),
        # Persistent reducer path emits only the current user turn.
        # One-shot callers can opt into seeding the provided input
        # history directly via ``include_input_history=True``.
        history=visible_history,
        transcript=visible_history,
        working_memory=list(agent_input.working_memory),
        memory={
            "summary": "",
            "active_concerns": [],
            "open_loops": [],
            "current_goal": None,
            # v0.7 Stage C: procedural fields default to empty / recall off.
            # load_memory_node overwrites these with the stored profile on
            # every turn; the empty defaults here exist so that any code
            # reading state["memory"] before load_memory_node has run sees
            # a consistent shape rather than missing keys.
            "procedural_rules": [],
            "proactive_recall_enabled": False,
        },
        progress={
            "intent": None,
            "intent_source": None,
            "stage": "opening",
            "stage_source": "deterministic",
            "stage_reason": "turn_start",
            "turn_count": turn_count,
            "is_guest": False,
        },
        crisis=CrisisAssessment(),
        routing={
            "route": "pending",
            "mode": "pending",
            "mode_source": "graph_bootstrap",
            "mode_type": ModeType.OPERATIONAL,
            # modality is intentionally NOT reset here. With the
            # _merge_dicts reducer on routing, the checkpoint preserves
            # the dispatcher's modality selection across turns. This is
            # critical for multi-turn exercise modality continuity —
            # the exercise continuation fast path reads modality from
            # routing state and carries it forward.
        },
        response={
            "guidance": "pending",
            "kind": ResponseKind.THERAPEUTIC,
            "text": "",
            "should_persist_memory": False,
        },
        # v0.8 observability: start each turn with a fresh, empty
        # diagnostics dict that nodes can write timings and write-
        # counts into. Uses a merge reducer so multiple nodes can
        # write independently.
        diagnostics={},
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
        modality=routing_state.get("modality"),
        should_persist_memory=response_state.get("should_persist_memory", False),
        # v0.8 observability: pass the per-turn diagnostics dict
        # through to the CLI / API caller. Empty dict when no node
        # wrote anything (e.g., pre-observability-instrumentation
        # tests that construct state manually).
        diagnostics=dict(state.get("diagnostics", {})),
    )


# ── Graph assembly ───────────────────────────────────────────────────────────


def build_agent_workflow(
    *,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the LangGraph workflow.

    Topology (v0.9 — crisis gate runs before memory load)::

        START
          → crisis_gate_node  (Command routes to one of the branches)
          ├─ crisis_response_node → crisis_log_node → finalize_turn_node
          │                                             → extract_semantic_facts_node
          │                                             → extract_procedural_rules_node → END
          └─ load_memory_node → therapeutic_subgraph → finalize_turn_node
                                                         → extract_semantic_facts_node
                                                         → extract_procedural_rules_node → END

    v0.9 safety reorder: the crisis gate runs FIRST (directly after
    START), before any memory retrieval. This ensures that a user in
    crisis is never blocked by an optional memory feature.

    v0.9 latency reorder: ``finalize_turn_node`` runs BEFORE extractors.
    This checkpoints the response to transcript/history immediately,
    allowing ``run_turn_stream`` to emit a ``ResponseEvent`` to the
    user while the extractor LLM calls (~6.7s) run in the background.

    The therapeutic subgraph is embedded as a single node compiled by
    :func:`agent.therapeutic.graph.build_therapeutic_subgraph`. Its
    internal structure (dispatcher + mode nodes) is hidden from the
    parent topology, keeping the top-level graph small and inspectable
    in LangSmith.

    ``finalize_turn_node`` appends the assistant response to the
    transcript. It runs on both branches and pairs with
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

    # Retry policy for nodes that perform I/O (LLM calls, store access,
    # embedding API, web search). Acts as defense-in-depth: most nodes
    # already catch expected exceptions internally and fall back to
    # deterministic behavior, so retries fire only for *unexpected*
    # transient failures outside the node's own error handling (e.g.,
    # framework-level deserialization errors, store connection resets,
    # or exceptions raised before the node reaches its try/except).
    # finalize_turn_node is excluded (pure state, no I/O).
    # therapeutic_subgraph is excluded (compiled graph — its child
    # nodes have their own retry policies).
    _io_retry = RetryPolicy(max_attempts=2)

    # Register all top-level nodes.
    workflow.add_node("load_memory_node", run_load_memory_node, retry_policy=_io_retry)
    workflow.add_node("crisis_gate_node", run_crisis_gate_node, retry_policy=_io_retry)
    workflow.add_node(
        "crisis_response_node", run_crisis_response_node, retry_policy=_io_retry
    )
    workflow.add_node("crisis_log_node", run_crisis_log_node, retry_policy=_io_retry)
    workflow.add_node("therapeutic_subgraph", therapeutic_subgraph)
    workflow.add_node(
        "extract_semantic_facts_node",
        run_extract_semantic_facts_node,
        retry_policy=_io_retry,
    )
    workflow.add_node(
        "extract_procedural_rules_node",
        run_extract_procedural_rules_node,
        retry_policy=_io_retry,
    )
    workflow.add_node("finalize_turn_node", run_finalize_turn_node)

    # Spine: entry → crisis gate (safety first, Command-routes).
    workflow.add_edge(START, "crisis_gate_node")

    # Crisis branch: crisis_response → crisis_log → finalize → extractors → END.
    # Memory load is SKIPPED — crisis nodes use zero memory state.
    workflow.add_edge("crisis_response_node", "crisis_log_node")
    workflow.add_edge("crisis_log_node", "finalize_turn_node")

    # Therapeutic branch: memory load → subgraph → finalize → extractors → END.
    workflow.add_edge("load_memory_node", "therapeutic_subgraph")
    workflow.add_edge("therapeutic_subgraph", "finalize_turn_node")

    # Shared terminal: finalize FIRST (checkpoints the response to
    # transcript/history), THEN extractors (side-effect LLM calls that
    # don't affect user-visible output). v0.9 reorder: finalize runs
    # before extractors so the response is persisted and can be emitted
    # to the user immediately via ResponseEvent while extractors run in
    # the background. Both extractors fan out in parallel from finalize
    # — the diagnostics dict uses a merge reducer so they can write
    # independently without racing.
    workflow.add_edge("finalize_turn_node", "extract_semantic_facts_node")
    workflow.add_edge("finalize_turn_node", "extract_procedural_rules_node")
    workflow.add_edge("extract_semantic_facts_node", END)
    workflow.add_edge("extract_procedural_rules_node", END)

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
    extract_semantic_facts -> extract_procedural_rules -> finalize_turn
    -> END``.

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
        build_initial_state(agent_input, include_input_history=True),
        context=WorkflowContext(
            llm_client=llm_client,
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=memory_mode,
        ),
    )
    return state_to_output(final_state)
