"""Therapeutic subgraph assembly.

Builds the compiled ``StateGraph`` that wires the therapeutic dispatcher
+ mode nodes together. The parent graph (``agent/graph.py``) registers
the result as a single node via ``add_node("therapeutic_subgraph",
build_therapeutic_subgraph())``.

Internal topology:

    START(subgraph)
      → therapeutic_dispatch_node
        → Command(goto=<one of the mode nodes>)
      → [supportive_response_node
         | reflective_response_node
         | clarifying_response_node
         | psychoeducation_response_node
         | closing_response_node
         | guided_exercise_response_node]
      → END(subgraph)

The dispatcher returns ``Command(goto=...)`` rather than using a
conditional edge — the same pattern as the top-level ``crisis_gate_node``.
Each mode node terminates at the subgraph's END; LangGraph propagates
state and the runtime context into and out of the subgraph
automatically because both the parent and subgraph share ``AgentState``.

Mode rollout history:
- v0.1: supportive, reflective, clarifying (the MVP three)
- v0.5: LLM-backed dispatcher added with tuned prompts
- v0.6 Stage A: psychoeducation mode node wired up
- v0.6 Stage B: closing mode node wired up (tonal only — session
  termination and summarization remain runtime concerns)
- v0.6 Stage C: guided_exercise mode node wired up. First
  multi-turn mode in the codebase; tracks exercise state via
  ``progress.exercise_type`` + ``progress.exercise_step``, and
  the dispatcher gained an "active-exercise fast-path" that
  keeps the mode engaged across turns without re-classifying.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agent.runtime_context import WorkflowContext
from agent.state import AgentState, ResponseState, RoutingState, SessionProgressState
from agent.therapeutic.clarifying import run_clarifying_response_node
from agent.therapeutic.closing import run_closing_response_node
from agent.therapeutic.dispatcher import (
    CLARIFYING_NODE,
    CLOSING_NODE,
    GUIDED_EXERCISE_NODE,
    PSYCHOEDUCATION_NODE,
    REFLECTIVE_NODE,
    SUPPORTIVE_NODE,
    TECHNIQUE_NODE,
    run_therapeutic_dispatch_node,
)
from agent.therapeutic.guided_exercise import run_guided_exercise_response_node
from agent.therapeutic.psychoeducation import run_psychoeducation_response_node
from agent.therapeutic.reflective import run_reflective_response_node
from agent.therapeutic.supportive import run_supportive_response_node
from agent.therapeutic.technique import run_technique_response_node

# Dispatcher node name exported so the parent graph (or tests) can
# reference it without importing from dispatcher.py directly.
DISPATCH_NODE = "therapeutic_dispatch_node"


class TherapeuticSubgraphOutput(TypedDict):
    """Parent-visible delta emitted by the therapeutic subgraph.

    The compiled subgraph is registered as a single parent-graph node.
    If we let it default to the full ``AgentState`` output schema,
    LangGraph bubbles the entire accumulated state back to the parent
    on subgraph completion. That becomes a problem once ``history`` and
    ``transcript`` are reducer-backed: the parent sees those full lists
    as a node delta and appends them again, duplicating the transcript.

    Restricting the subgraph's output schema to only the fields it owns
    keeps the parent merge semantic correct while preserving the full
    input state inside the subgraph itself.
    """

    routing: NotRequired[RoutingState]
    response: NotRequired[ResponseState]
    progress: NotRequired[SessionProgressState]


def build_therapeutic_subgraph() -> CompiledStateGraph:
    """Build and compile the therapeutic response subgraph.

    Returns a compiled ``StateGraph`` that can be registered as a single
    node in the parent ``build_agent_workflow`` via::

        parent.add_node("therapeutic_subgraph", build_therapeutic_subgraph())

    The subgraph shares the parent's ``AgentState`` schema, so no
    wrapper function is needed — LangGraph propagates state into and
    out of the subgraph automatically.

    The subgraph does NOT do memory writes. Those live at the top level
    (``extract_semantic_facts_node`` and similar, Stage E+) and run
    AFTER the subgraph returns. This keeps memory concerns out of the
    therapeutic package.

    Returns:
        A ``CompiledStateGraph`` ready to register as a node in the
        parent graph.
    """

    subgraph = StateGraph(
        AgentState,
        context_schema=WorkflowContext,
        output_schema=TherapeuticSubgraphOutput,
    )

    # Retry policy for therapeutic nodes that make LLM calls. Acts as
    # defense-in-depth: each mode node catches LLM exceptions internally
    # and falls back to deterministic responses, so retries fire only
    # for unexpected transient failures outside the node's own error
    # handling (framework-level errors, connection resets, etc.).
    _io_retry = RetryPolicy(max_attempts=2)

    subgraph.add_node(
        DISPATCH_NODE, run_therapeutic_dispatch_node, retry_policy=_io_retry
    )
    subgraph.add_node(
        SUPPORTIVE_NODE, run_supportive_response_node, retry_policy=_io_retry
    )
    subgraph.add_node(
        REFLECTIVE_NODE, run_reflective_response_node, retry_policy=_io_retry
    )
    subgraph.add_node(
        CLARIFYING_NODE, run_clarifying_response_node, retry_policy=_io_retry
    )
    subgraph.add_node(
        PSYCHOEDUCATION_NODE, run_psychoeducation_response_node, retry_policy=_io_retry
    )
    subgraph.add_node(CLOSING_NODE, run_closing_response_node, retry_policy=_io_retry)
    subgraph.add_node(
        GUIDED_EXERCISE_NODE, run_guided_exercise_response_node, retry_policy=_io_retry
    )
    subgraph.add_node(
        TECHNIQUE_NODE, run_technique_response_node, retry_policy=_io_retry
    )

    subgraph.add_edge(START, DISPATCH_NODE)
    # therapeutic_dispatch_node returns Command(goto=<style>); no
    # conditional edge needed. Each style node terminates at END.
    subgraph.add_edge(SUPPORTIVE_NODE, END)
    subgraph.add_edge(REFLECTIVE_NODE, END)
    subgraph.add_edge(CLARIFYING_NODE, END)
    subgraph.add_edge(PSYCHOEDUCATION_NODE, END)
    subgraph.add_edge(CLOSING_NODE, END)
    subgraph.add_edge(GUIDED_EXERCISE_NODE, END)
    subgraph.add_edge(TECHNIQUE_NODE, END)

    return subgraph.compile()
