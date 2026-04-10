"""Therapeutic subgraph assembly.

Builds the compiled ``StateGraph`` that wires the therapeutic dispatcher
+ three mode nodes together. The parent graph (``agent/graph.py``)
registers the result as a single node via ``add_node("therapeutic_subgraph",
build_therapeutic_subgraph())``.

Internal topology:

    START(subgraph)
      → therapeutic_dispatch_node
        → Command(goto=<one of the three mode nodes>)
      → [supportive_response_node
         | reflective_response_node
         | clarifying_response_node]
      → END(subgraph)

The dispatcher returns ``Command(goto=...)`` rather than using a
conditional edge — the same pattern as the top-level ``crisis_gate_node``.
Each mode node terminates at the subgraph's END; LangGraph propagates
state and the runtime context into and out of the subgraph
automatically because both the parent and subgraph share ``AgentState``.

Phase 1 v0.1 scope: three modes (supportive, reflective, clarifying)
with keyword-based dispatch. The other three modes (psychoeducation,
guided_exercise, closing) land in v0.6 alongside the LLM-backed
dispatcher in v0.5.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.clarifying import run_clarifying_response_node
from agent.therapeutic.dispatcher import (
    CLARIFYING_NODE,
    REFLECTIVE_NODE,
    SUPPORTIVE_NODE,
    run_therapeutic_dispatch_node,
)
from agent.therapeutic.reflective import run_reflective_response_node
from agent.therapeutic.supportive import run_supportive_response_node

# Dispatcher node name exported so the parent graph (or tests) can
# reference it without importing from dispatcher.py directly.
DISPATCH_NODE = "therapeutic_dispatch_node"


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

    subgraph = StateGraph(AgentState, context_schema=WorkflowContext)

    subgraph.add_node(DISPATCH_NODE, run_therapeutic_dispatch_node)
    subgraph.add_node(SUPPORTIVE_NODE, run_supportive_response_node)
    subgraph.add_node(REFLECTIVE_NODE, run_reflective_response_node)
    subgraph.add_node(CLARIFYING_NODE, run_clarifying_response_node)

    subgraph.add_edge(START, DISPATCH_NODE)
    # therapeutic_dispatch_node returns Command(goto=<mode>); no
    # conditional edge needed. Each mode node terminates at END.
    subgraph.add_edge(SUPPORTIVE_NODE, END)
    subgraph.add_edge(REFLECTIVE_NODE, END)
    subgraph.add_edge(CLARIFYING_NODE, END)

    return subgraph.compile()
