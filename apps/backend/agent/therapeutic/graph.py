"""Therapeutic subgraph assembly.

Builds the compiled ``StateGraph`` that wires the therapeutic dispatcher
+ response-style nodes together. The parent graph (``agent/graph.py``) registers
the result as a single node via ``add_node("therapeutic_subgraph",
build_therapeutic_subgraph())``.

Internal topology:

    START(subgraph)
      → therapeutic_dispatch_node
        → Command(goto=<therapeutic_response_node | guided_exercise_response_node>)
      → [therapeutic_response_node | guided_exercise_response_node]
      → END(subgraph)

The dispatcher returns ``Command(goto=...)`` rather than using a
conditional edge — the same pattern as the top-level ``crisis_gate_node``.
Each response-style node terminates at the subgraph's END; LangGraph propagates
state and the runtime context into and out of the subgraph
automatically.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agent.models import Channel, CrisisAssessment
from agent.runtime_context import WorkflowContext
from agent.state import (
    AgentState,
    ExerciseState,
    ProceduralProfileState,
    SessionMemoryState,
    SessionProgressState,
)
from agent.therapeutic.dispatch import (
    GUIDED_EXERCISE_NODE,
    THERAPEUTIC_RESPONSE_NODE,
    run_therapeutic_dispatch_node,
)
from agent.therapeutic.exercises.node import run_guided_exercise_response_node
from agent.therapeutic.response import run_therapeutic_response_node
from agent.memory.entries import WorkingMemoryEntry

# Dispatcher node name exported so the parent graph (or tests) can
# reference it without importing from dispatcher.py directly.
DISPATCH_NODE = "therapeutic_dispatch_node"


class TherapeuticSubgraphOutput(TypedDict):
    """Parent-visible delta emitted by the therapeutic subgraph.

    The compiled subgraph is registered as a single parent-graph node.
    If we let it default to the full ``AgentState`` output schema,
    LangGraph bubbles the entire accumulated state back to the parent
    on subgraph completion. That becomes a problem when ``transcript`` is
    reducer-backed: the parent sees the full list as a node delta and appends
    it again, duplicating the transcript.

    Restricting the subgraph's output schema to only the fields it owns
    keeps the parent merge semantic correct while preserving the full
    input state inside the subgraph itself.
    """

    response_text: NotRequired[str]
    response_style: NotRequired[str]
    therapeutic_approach: NotRequired[str | None]
    exercise_state: NotRequired[ExerciseState]
    diagnostics: NotRequired[dict[str, Any]]


class TherapeuticSubgraphInput(TypedDict):
    """Subset of parent state consumed by the therapeutic subgraph."""

    message: str
    channel: Channel
    user_id: str | None
    session_id: str | None
    installed_skills: list[str]
    crisis: CrisisAssessment
    transcript: list[dict[str, str]]
    working_memory: list[WorkingMemoryEntry]
    session_memory: SessionMemoryState
    procedural_profile: ProceduralProfileState
    session_progress: SessionProgressState
    exercise_state: ExerciseState
    therapeutic_approach: NotRequired[str | None]
    diagnostics: NotRequired[dict[str, Any]]


def build_therapeutic_subgraph() -> CompiledStateGraph[
    AgentState,
    WorkflowContext,
    TherapeuticSubgraphInput,
    TherapeuticSubgraphOutput,
]:
    """Build and compile the therapeutic response subgraph.

    Returns a compiled ``StateGraph`` that can be registered as a single
    node in the parent ``build_agent_workflow`` via::

        parent.add_node("therapeutic_subgraph", build_therapeutic_subgraph())

    The subgraph keeps ``AgentState`` as its internal schema, but its
    parent-facing contract is narrowed with explicit input and output
    schemas so only the channels it actually reads and writes cross the
    subgraph boundary.

    The subgraph does not own general memory writes. Those live at the
    top level (``extract_semantic_facts_node`` and similar) and run after
    the subgraph returns, which keeps memory concerns out of the
    therapeutic package.

    Returns:
        A ``CompiledStateGraph`` ready to register as a node in the
        parent graph.
    """

    subgraph = StateGraph(
        AgentState,
        context_schema=WorkflowContext,
        input_schema=TherapeuticSubgraphInput,
        output_schema=TherapeuticSubgraphOutput,
    )

    # Retry policy for therapeutic nodes that make LLM calls. Response-generation
    # failures propagate out of the node so transient provider errors can be
    # retried by LangGraph rather than hidden behind canned text.
    _io_retry = RetryPolicy(max_attempts=2)

    subgraph.add_node(
        DISPATCH_NODE, run_therapeutic_dispatch_node, retry_policy=_io_retry
    )
    subgraph.add_node(
        THERAPEUTIC_RESPONSE_NODE,
        run_therapeutic_response_node,
        retry_policy=_io_retry,
    )
    subgraph.add_node(
        GUIDED_EXERCISE_NODE, run_guided_exercise_response_node, retry_policy=_io_retry
    )

    subgraph.add_edge(START, DISPATCH_NODE)
    # therapeutic_dispatch_node returns Command(goto=<node>); no
    # conditional edge needed. Each response node terminates at END.
    subgraph.add_edge(THERAPEUTIC_RESPONSE_NODE, END)
    subgraph.add_edge(GUIDED_EXERCISE_NODE, END)

    return subgraph.compile()
