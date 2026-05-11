"""Shared top-level graph node names and status-stage labels."""

from __future__ import annotations

from typing import Literal, TypeAlias

LOAD_MEMORY_NODE = "load_memory_node"
CRISIS_GATE_NODE = "crisis_gate_node"
CRISIS_RESOURCE_LOOKUP_NODE = "crisis_resource_lookup_node"
CRISIS_RESPONSE_NODE = "crisis_response_node"
CRISIS_LOG_NODE = "crisis_log_node"
TURN_DISPATCH_NODE = "turn_dispatch_node"
MEMORY_CONTROL_NODE = "memory_control_node"
GROUNDED_ANSWER_NODE = "grounded_answer_node"
THERAPEUTIC_SUBGRAPH_NODE = "therapeutic_subgraph"
FINALIZE_TURN_NODE = "finalize_turn_node"

CrisisGateNextNode: TypeAlias = Literal[
    "crisis_resource_lookup_node",
    "turn_dispatch_node",
]
TurnDispatchNextNode: TypeAlias = Literal[
    "memory_control_node",
    "grounded_answer_node",
    "load_memory_node",
]

GRAPH_NODE_TO_STATUS_STAGE = {
    LOAD_MEMORY_NODE: "load_memory",
    CRISIS_GATE_NODE: "crisis_gate",
    CRISIS_RESOURCE_LOOKUP_NODE: "crisis_resource_lookup",
    CRISIS_RESPONSE_NODE: "crisis_response",
    CRISIS_LOG_NODE: "crisis_log",
    TURN_DISPATCH_NODE: "turn_dispatch",
    MEMORY_CONTROL_NODE: "memory_control",
    GROUNDED_ANSWER_NODE: "grounded_lookup",
    THERAPEUTIC_SUBGRAPH_NODE: "therapeutic",
    FINALIZE_TURN_NODE: "finalize",
}
