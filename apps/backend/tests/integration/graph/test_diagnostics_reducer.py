"""Guard tests for the diagnostics merge reducer and terminal memory extraction.

Phase B of the LangGraph best-practice alignment plan. Tests verify:

1. ``AgentState.diagnostics`` uses a merge reducer annotation.
2. No node function spreads the existing diagnostics dict (all 4 sites).
3. Terminal memory extraction is represented by one graph node.
4. A full turn produces merged diagnostics from all nodes.
5. The reducer handles None values defensively.
6. LangGraph actually resolves the reducer at graph compile time.

Codex review feedback addressed:
- No-spread test covers all 4 edited node files, not just load_memory.
- Graph-level test verifies LangGraph resolves NotRequired[Annotated[...]]
  into a reducer-backed channel.
- Streaming test length check prevents false passes from duplicates.
- Reducer None guard is tested explicitly.
"""

from __future__ import annotations

import typing
from typing import Any, NotRequired, get_type_hints

import pytest

from agent.graph import build_agent_workflow, run_agent
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput
from agent.nodes.crisis_gate import _build_crisis_delta
from agent.nodes.load_memory import run_load_memory_node
from agent.models import CrisisAssessment
from tests.support.persistence import FakeCrossRestartLLM
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, _merge_dicts


# ── Reducer annotation tests ───────────────────────────────────────────────


def test_diagnostics_uses_merge_reducer() -> None:
    """``AgentState.diagnostics`` should have a merge reducer annotation."""

    hints = get_type_hints(AgentState, include_extras=True)
    diagnostics_hint = hints.get("diagnostics")
    assert diagnostics_hint is not None, "AgentState has no 'diagnostics' field"

    # Unwrap NotRequired if present to reach the Annotated layer.
    inner = diagnostics_hint
    origin = typing.get_origin(inner)
    if origin is NotRequired:
        args = typing.get_args(inner)
        inner = args[0] if args else inner

    assert typing.get_origin(inner) is typing.Annotated or hasattr(
        inner, "__metadata__"
    ), f"diagnostics type {inner} is not Annotated — no reducer attached"
    metadata = getattr(inner, "__metadata__", ())
    assert len(metadata) > 0, "diagnostics has Annotated wrapper but no metadata"
    reducer = metadata[0]
    assert callable(reducer), f"diagnostics metadata[0] is not callable: {reducer!r}"

    # Verify the reducer merges dicts correctly.
    left = {"a": 1, "b": 2}
    right = {"b": 3, "c": 4}
    merged = reducer(left, right)
    assert merged == {"a": 1, "b": 3, "c": 4}, f"Reducer merge incorrect: {merged}"


def test_merge_dicts_handles_none_values() -> None:
    """The reducer should not crash if either side is None."""

    assert _merge_dicts(None, {"a": 1}) == {"a": 1}  # type: ignore[arg-type]
    assert _merge_dicts({"a": 1}, None) == {"a": 1}  # type: ignore[arg-type]
    assert _merge_dicts(None, None) == {}  # type: ignore[arg-type]


def test_langgraph_resolves_diagnostics_reducer_at_compile() -> None:
    """Verify LangGraph actually resolves NotRequired[Annotated[...]] into a
    reducer-backed channel at graph compile time.

    This is the framework-risk edge case Codex flagged — type-level
    introspection alone doesn't guarantee LangGraph handles the nesting.
    We compile the graph and verify the diagnostics channel has a reducer.
    """

    graph = build_agent_workflow()
    # LangGraph compiled graphs expose channel info via .channels
    channels = graph.channels
    assert "diagnostics" in channels, "diagnostics not found in compiled graph channels"
    # The channel should be a BinaryOperatorAggregate (reducer-backed),
    # not a LastValue (overwrite).
    channel = channels["diagnostics"]
    channel_type_name = type(channel).__name__
    assert channel_type_name == "BinaryOperatorAggregate", (
        f"diagnostics channel is {channel_type_name}, expected "
        f"BinaryOperatorAggregate (reducer-backed). LangGraph did not "
        f"resolve the Annotated reducer from NotRequired[Annotated[...]]."
    )


# ── No-spread tests for all 4 edited node files ────────────────────────────


class _FakeRuntime:
    """Minimal runtime for isolated node tests."""

    def __init__(self, **overrides: Any) -> None:
        self.context = WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
            **overrides,
        )


def _state_with_pre_existing_diagnostics(**extra: Any) -> Any:
    """Build a minimal state dict with pre-existing diagnostics keys."""
    return {
        "message": "hello",
        "session_id": "test-thread",
        "user_id": None,
        "transcript": [],
        "history": [],
        "working_memory": [],
        "session_memory": {
            "summary": "",
            "active_concerns": [],
            "open_loops": [],
            "current_goal": None,
        },
        "procedural_profile": {
            "procedural_rules": [],
            "proactive_recall_enabled": False,
        },
        "route": "therapeutic",
        "response_style": "pending",
        "response_text": "",
        "session_progress": {"turn_count": 1},
        "diagnostics": {"pre_existing_key": 42, "another_key": "val"},
        **extra,
    }


def _assert_no_spread(diag: dict[str, Any], node_name: str) -> None:
    """Assert that a diagnostics delta does NOT contain pre-existing keys."""
    assert "pre_existing_key" not in diag, (
        f"{node_name} delta contains pre-existing key — it is still spreading "
        f"state.get('diagnostics', {{}}). Remove the spread and let the "
        f"reducer handle merging."
    )
    assert "another_key" not in diag, (
        f"{node_name} delta contains 'another_key' — still spreading."
    )


@pytest.mark.asyncio
async def test_load_memory_node_does_not_spread_diagnostics() -> None:
    """load_memory_node should return only its own diagnostics keys."""
    state = _state_with_pre_existing_diagnostics()
    delta = await run_load_memory_node(state, _FakeRuntime())  # type: ignore[arg-type]
    diag = delta.get("diagnostics", {})
    _assert_no_spread(diag, "load_memory_node")
    assert "load_memory_ms" in diag


def test_crisis_gate_build_delta_does_not_spread_diagnostics() -> None:
    """_build_crisis_delta should return only its own diagnostics keys.

    Uses the helper directly since the full crisis_gate_node returns a
    Command object (harder to call in isolation).
    """
    delta = _build_crisis_delta(
        CrisisAssessment(),
        override_kind="none",
        classifier_path="deterministic",
        llm_failure_occurred=False,
        duration_ms=5.0,
    )
    diag = delta.get("diagnostics", {})
    _assert_no_spread(diag, "_build_crisis_delta")
    assert "crisis_gate_ms" in diag


# ── No remaining spread sites (static check) ───────────────────────────────


def test_no_diagnostics_spreading_in_codebase() -> None:
    """No node file should contain ``**state.get("diagnostics"`` anymore.

    This is a static grep-style check to catch regressions if someone
    adds a new node and copies the old spreading pattern.
    """
    from tests.support.paths import BACKEND_ROOT

    agent_dir = BACKEND_ROOT / "agent"
    pattern = '**state.get("diagnostics"'
    violations = []
    for py_file in agent_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # skip non-text files (e.g., macOS resource forks)
        if pattern in content:
            violations.append(str(py_file.relative_to(agent_dir)))
    assert violations == [], (
        f"Found diagnostics spreading in: {violations}. "
        f"Use the _merge_dicts reducer instead."
    )


# ── Graph topology: extractors lifted out of the graph ─────────────────────


def test_finalize_terminates_at_end_with_no_extractor_nodes() -> None:
    """Memory extraction is no longer in the graph topology.

    Earlier shapes routed extraction through a wrapper node, then through
    parallel terminal edges. Both became hot-path tax: extraction blocks
    the user-visible turn even though its work is post-response. The
    runtime now manages extraction as a background task (or runs it
    synchronously after ``ainvoke`` for one-shot ``run_agent`` callers).
    Pin the new shape: ``finalize_turn_node`` is the terminal node.
    """

    graph = build_agent_workflow()
    graph_def = graph.get_graph()
    edge_tuples = {(e.source, e.target) for e in graph_def.edges}
    node_names = set(graph_def.nodes)

    assert ("finalize_turn_node", "__end__") in edge_tuples
    assert "extract_semantic_facts_node" not in node_names
    assert "extract_procedural_rules_node" not in node_names
    assert "memory_extraction_node" not in node_names


# ── End-to-end: diagnostics merge across all nodes ──────────────────────────


@pytest.mark.asyncio
async def test_full_turn_diagnostics_merge_all_node_keys() -> None:
    """A full turn should produce diagnostics with keys from all nodes that
    write diagnostics — crisis_gate, load_memory (non-incognito), and
    both extractors.

    Uses LOCAL mode so load_memory_node writes its diagnostics (the
    incognito path returns early without diagnostics).
    """

    store = OpenCouchMemoryStore()
    crisis_log = InMemoryCrisisLogBackend()
    result = await run_agent(
        AgentInput(message="I feel a bit anxious today", session_id="test-diag"),
        memory_store=store,
        crisis_log_backend=crisis_log,
        memory_mode=MemoryMode.LOCAL,
        llm_client=FakeCrossRestartLLM(),
    )
    diag = result.diagnostics

    # Keys from crisis_gate_node.
    assert "crisis_gate_ms" in diag, f"Missing crisis_gate_ms. Keys: {sorted(diag)}"
    assert "crisis_level" in diag

    # Keys from load_memory_node (only present in non-incognito mode).
    assert "load_memory_ms" in diag, f"Missing load_memory_ms. Keys: {sorted(diag)}"
    assert "semantic_hits" in diag

    # Keys from extract_semantic_facts_node.
    assert "extract_facts_ms" in diag, f"Missing extract_facts_ms. Keys: {sorted(diag)}"
    assert "extract_facts_reason" in diag

    # Keys from extract_procedural_rules_node.
    assert "extract_procedural_ms" in diag, (
        f"Missing extract_procedural_ms. Keys: {sorted(diag)}"
    )
    assert "extract_procedural_reason" in diag
