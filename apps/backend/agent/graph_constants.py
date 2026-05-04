"""Shared top-level graph node names and status-stage labels."""

from __future__ import annotations

from typing import Literal, TypeAlias

LOAD_MEMORY_NODE = "load_memory_node"
CRISIS_GATE_NODE = "crisis_gate_node"
CRISIS_RESOURCE_LOOKUP_NODE = "crisis_resource_lookup_node"
CRISIS_RESPONSE_NODE = "crisis_response_node"
CRISIS_LOG_NODE = "crisis_log_node"
MEMORY_CONTROL_GATE_NODE = "memory_control_gate_node"
MEMORY_CONTROL_NODE = "memory_control_node"
GROUNDED_LOOKUP_GATE_NODE = "grounded_lookup_gate_node"
GROUNDED_ANSWER_NODE = "grounded_answer_node"
THERAPEUTIC_SUBGRAPH_NODE = "therapeutic_subgraph"
MEMORY_EXTRACTION_NODE = "memory_extraction_node"
FINALIZE_TURN_NODE = "finalize_turn_node"

CrisisGateNextNode: TypeAlias = Literal[
    "crisis_resource_lookup_node",
    "memory_control_gate_node",
]
MemoryControlGateNextNode: TypeAlias = Literal[
    "memory_control_node",
    "grounded_lookup_gate_node",
]
GroundedLookupGateNextNode: TypeAlias = Literal[
    "grounded_answer_node",
    "load_memory_node",
]

GRAPH_NODE_TO_STATUS_STAGE = {
    LOAD_MEMORY_NODE: "load_memory",
    CRISIS_GATE_NODE: "crisis_gate",
    CRISIS_RESOURCE_LOOKUP_NODE: "crisis_resource_lookup",
    CRISIS_RESPONSE_NODE: "crisis_response",
    CRISIS_LOG_NODE: "crisis_log",
    MEMORY_CONTROL_GATE_NODE: "memory_control_gate",
    MEMORY_CONTROL_NODE: "memory_control",
    GROUNDED_LOOKUP_GATE_NODE: "grounded_lookup_gate",
    GROUNDED_ANSWER_NODE: "grounded_lookup",
    THERAPEUTIC_SUBGRAPH_NODE: "therapeutic",
    MEMORY_EXTRACTION_NODE: "memory_extraction",
    FINALIZE_TURN_NODE: "finalize",
}
