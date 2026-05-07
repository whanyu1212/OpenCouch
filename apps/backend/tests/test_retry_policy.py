"""Guard tests for RetryPolicy on I/O-bound graph nodes.

Phase A of the LangGraph best-practice alignment plan. These tests
verify that all nodes making network / LLM calls have a RetryPolicy
configured, and that adding retry policies does not alter the graph
topology.

Codex review feedback addressed:
- Node sets are now dynamically discovered from the compiled graph
  with an explicit exclusion list, so adding a new node without retry
  will fail this test automatically.

Note: these tests verify retry policy *presence*, not behavioral
retry (which would require restructuring nodes to let transient
exceptions propagate). The retry policies act as defense-in-depth
for unexpected failures outside each node's own error handling.
"""

from __future__ import annotations

from typing import Any


from agent.graph import build_agent_workflow
from agent.therapeutic.graph import build_therapeutic_subgraph

# Nodes in the parent graph that are intentionally EXCLUDED from retry
# because they do no I/O. Every other user-defined node should have a
# retry policy. If you add a new non-I/O node, add it here.
_PARENT_NO_RETRY_NODES = {
    "finalize_turn_node",  # pure state manipulation, no I/O
    "therapeutic_subgraph",  # compiled subgraph — child nodes have their own retry
}

# LangGraph internal nodes that never have retry.
_INTERNAL_NODES = {"__start__", "__end__"}


def _get_retry_max_attempts(node: Any) -> int | None:
    """Extract max_attempts from a compiled PregelNode's retry_policy.

    Returns None if no retry policy is set. Handles both single
    RetryPolicy and tuple-of-RetryPolicy storage formats.
    """
    rp = node.retry_policy
    if rp is None:
        return None
    policies = rp if isinstance(rp, tuple) else (rp,)
    return max(p.max_attempts for p in policies)


def test_parent_graph_all_io_nodes_have_retry_policy() -> None:
    """Every user-defined I/O node in the parent graph should have retry.

    Uses dynamic discovery: any node NOT in the exclusion list must
    have a retry policy. This catches newly added nodes automatically.
    """

    graph = build_agent_workflow()
    for name, node in graph.nodes.items():
        if name in _INTERNAL_NODES or name in _PARENT_NO_RETRY_NODES:
            continue
        max_attempts = _get_retry_max_attempts(node)
        assert max_attempts is not None, (
            f"Node {name!r} has no retry_policy. Either add "
            f"retry_policy=RetryPolicy(max_attempts=2) to its add_node call, "
            f"or add it to _PARENT_NO_RETRY_NODES if it does no I/O."
        )
        assert max_attempts >= 2, (
            f"Node {name!r} retry_policy.max_attempts is {max_attempts}, expected >= 2."
        )


def test_excluded_parent_nodes_have_no_retry() -> None:
    """Nodes in the exclusion list should NOT have retry policies.

    Guards against accidentally adding retry to a node that was
    explicitly excluded (e.g., finalize_turn_node).
    """

    graph = build_agent_workflow()
    for name in _PARENT_NO_RETRY_NODES:
        if name not in graph.nodes:
            continue
        node = graph.nodes[name]
        assert node.retry_policy is None, (
            f"Node {name!r} is in the no-retry exclusion list but has a "
            f"retry_policy. Remove it from _PARENT_NO_RETRY_NODES or "
            f"remove the retry_policy."
        )


def test_therapeutic_subgraph_all_nodes_have_retry_policy() -> None:
    """Every node in the therapeutic subgraph should have retry.

    The subgraph has no non-I/O nodes (every node either dispatches
    or generates LLM responses), so all user-defined nodes should
    have retry.
    """

    subgraph = build_therapeutic_subgraph()
    for name, node in subgraph.nodes.items():
        if name in _INTERNAL_NODES:
            continue
        max_attempts = _get_retry_max_attempts(node)
        assert max_attempts is not None, (
            f"Subgraph node {name!r} has no retry_policy. "
            f"Add retry_policy=RetryPolicy(max_attempts=2) to its add_node call."
        )
        assert max_attempts >= 2, (
            f"Subgraph node {name!r} retry_policy.max_attempts is "
            f"{max_attempts}, expected >= 2."
        )


def test_retry_policy_does_not_alter_graph_topology() -> None:
    """Adding retry policies must not change the node set or edge set.

    Captures the current expected topology so that a future change that
    accidentally adds/removes nodes or edges while editing add_node
    calls is caught immediately.
    """

    graph = build_agent_workflow()
    graph_def = graph.get_graph()

    node_ids = set(graph_def.nodes.keys())
    edge_tuples = {(e.source, e.target) for e in graph_def.edges}

    # Expected nodes (parent-level view — subgraph internals are opaque).
    # Memory extraction is no longer in the graph — it runs as a runtime-
    # managed background task post-finalize, so finalize_turn_node is now
    # the terminal graph node.
    expected_nodes = {
        "__start__",
        "__end__",
        "crisis_gate_node",
        "crisis_resource_lookup_node",
        "memory_control_gate_node",
        "memory_control_node",
        "grounded_lookup_gate_node",
        "grounded_answer_node",
        "load_memory_node",
        "crisis_response_node",
        "crisis_log_node",
        "therapeutic_subgraph",
        "finalize_turn_node",
    }
    assert node_ids == expected_nodes, (
        f"Node set mismatch.\n"
        f"  Extra: {node_ids - expected_nodes}\n"
        f"  Missing: {expected_nodes - node_ids}"
    )

    # Expected edges. Includes the two Command-based routing edges from
    # crisis_gate_node and finalize_turn_node terminating the graph.
    expected_edges = {
        ("__start__", "crisis_gate_node"),
        ("crisis_gate_node", "crisis_resource_lookup_node"),
        ("crisis_gate_node", "memory_control_gate_node"),
        ("crisis_resource_lookup_node", "crisis_response_node"),
        ("crisis_response_node", "crisis_log_node"),
        ("crisis_log_node", "finalize_turn_node"),
        ("memory_control_gate_node", "memory_control_node"),
        ("memory_control_gate_node", "grounded_lookup_gate_node"),
        ("memory_control_node", "finalize_turn_node"),
        ("grounded_lookup_gate_node", "grounded_answer_node"),
        ("grounded_lookup_gate_node", "load_memory_node"),
        ("grounded_answer_node", "finalize_turn_node"),
        ("load_memory_node", "therapeutic_subgraph"),
        ("therapeutic_subgraph", "finalize_turn_node"),
        ("finalize_turn_node", "__end__"),
    }
    assert edge_tuples == expected_edges, (
        f"Edge set mismatch.\n"
        f"  Extra: {edge_tuples - expected_edges}\n"
        f"  Missing: {expected_edges - edge_tuples}"
    )
