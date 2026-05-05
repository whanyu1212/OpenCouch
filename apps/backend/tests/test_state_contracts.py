"""Guard tests for node-level state channel ownership.

These tests lock in the post-schema-split contract: each node may write only
the top-level channels it explicitly owns. If a node starts mutating an
unrelated channel, the failure should be immediate and local.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.graph import build_agent_workflow, build_initial_state
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.modes import MemoryMode
from agent.memory.models import DispatchDecision
from agent.memory.store import OpenCouchMemoryStore
from agent.models import AgentInput, CrisisAssessment
from agent.nodes.crisis_gate import run_crisis_gate_node
from agent.nodes.crisis_log import run_crisis_log_node
from agent.nodes.crisis_resource_lookup import run_crisis_resource_lookup_node
from agent.nodes.crisis_response import run_crisis_response_node
from agent.nodes.extract_semantic_facts import run_extract_semantic_facts_node
from agent.nodes.extract_procedural_rules import run_extract_procedural_rules_node
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.grounded_answer import run_grounded_answer_node
from agent.nodes.grounded_lookup_gate import run_grounded_lookup_gate_node
from agent.nodes.load_memory import run_load_memory_node
from agent.nodes.memory_control import run_memory_control_node
from agent.nodes.memory_control_gate import run_memory_control_gate_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentGraphOutputState, AgentState
from agent.therapeutic.dispatcher import run_therapeutic_dispatch_node
from agent.therapeutic.graph import (
    TherapeuticSubgraphInput,
    TherapeuticSubgraphOutput,
    build_therapeutic_subgraph,
)
from agent.therapeutic.guided_exercise import run_guided_exercise_response_node
from agent.therapeutic.response import run_therapeutic_response_node
from services.llm.base import BaseLLMClient, StructuredResponseT


class _FakeRuntime:
    """Minimal runtime wrapper exposing ``runtime.context`` for node tests."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
        response_llm: BaseLLMClient | None = None,
        memory_mode: MemoryMode = MemoryMode.LOCAL,
        session_memory_buffer: SessionMemoryBuffer | None = None,
    ) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            response_llm=response_llm,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=memory_mode,
            session_memory_buffer=session_memory_buffer,
        )


class _FakeDispatchLLM(BaseLLMClient):
    """Structured-output fake for dispatcher contract tests."""

    def __init__(
        self,
        *,
        response_style: str,
        therapeutic_approach: str,
    ) -> None:
        self.response_style = response_style
        self.therapeutic_approach = therapeutic_approach

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "unused"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        return cast(
            StructuredResponseT,
            DispatchDecision(
                response_style=self.response_style,  # type: ignore[arg-type]
                therapeutic_approach=self.therapeutic_approach,  # type: ignore[arg-type]
                reasoning="contract test",
                confidence="high",
            ),
        )


def _build_state(
    message: str = "I had a rough day today.",
    *,
    user_id: str | None = "user-1",
    session_id: str | None = "thread-1",
) -> AgentState:
    """Build a minimal seeded graph state for isolated node tests.

    Args:
        message: The current user message.
        user_id: The optional user identifier.
        session_id: The optional session identifier.

    Returns:
        A seeded ``AgentState`` suitable for direct node invocation.
    """

    state = build_initial_state(
        AgentInput(
            message=message,
            user_id=user_id,
            session_id=session_id,
            history=[],
            working_memory=[],
        )
    )
    return cast(AgentState, dict(state))


def _assert_exact_keys(delta: dict[str, Any], expected: set[str]) -> None:
    """Assert that a node delta writes exactly the expected channels.

    Args:
        delta: The node delta to inspect.
        expected: The expected top-level channel keys.
    """

    actual = set(delta.keys())
    assert actual == expected, (
        f"delta keys mismatch\n"
        f"  expected: {sorted(expected)!r}\n"
        f"  actual:   {sorted(actual)!r}"
    )


def _assert_allowed_keys(delta: dict[str, Any], allowed: set[str]) -> None:
    """Assert that a node delta writes only allowed channels.

    Args:
        delta: The node delta to inspect.
        allowed: The allowed top-level channel keys.
    """

    actual = set(delta.keys())
    assert actual.issubset(allowed), (
        f"delta writes unexpected channels\n"
        f"  allowed: {sorted(allowed)!r}\n"
        f"  actual:  {sorted(actual)!r}"
    )


def test_parent_graph_schema_contract() -> None:
    """The top-level graph should keep explicit input/output schemas."""

    graph = build_agent_workflow()
    assert graph.builder.input_schema is AgentGraphInputState
    assert graph.builder.output_schema is AgentGraphOutputState


def test_therapeutic_subgraph_schema_contract() -> None:
    """The therapeutic subgraph should keep explicit input/output schemas."""

    subgraph = build_therapeutic_subgraph()
    assert subgraph.builder.input_schema is TherapeuticSubgraphInput
    assert subgraph.builder.output_schema is TherapeuticSubgraphOutput


@pytest.mark.asyncio
async def test_crisis_gate_crisis_path_channel_contract() -> None:
    """Crisis gate should write only crisis-owned channels on crisis turns."""

    command = await run_crisis_gate_node(
        _build_state("I want to kill myself tonight."),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    assert command.goto == "crisis_resource_lookup_node"
    _assert_exact_keys(
        command.update,
        {
            "crisis",
            "route",
            "crisis_audit",
            "diagnostics",
            "response_style",
            "response_style_source",
            "response_style_type",
            "response_kind",
        },
    )


@pytest.mark.asyncio
async def test_crisis_gate_therapeutic_path_channel_contract() -> None:
    """Crisis gate should not write response channels on safe turns."""

    command = await run_crisis_gate_node(
        _build_state("I had a hard day at work."),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    assert command.goto == "memory_control_gate_node"
    _assert_exact_keys(
        command.update,
        {"crisis", "route", "crisis_audit", "diagnostics"},
    )


@pytest.mark.asyncio
async def test_memory_control_gate_passthrough_channel_contract() -> None:
    """Memory-control gate should write only action/diagnostic channels normally."""

    command = await run_memory_control_gate_node(
        _build_state("I had a hard day at work."),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    assert command.goto == "grounded_lookup_gate_node"
    _assert_exact_keys(command.update, {"memory_control", "diagnostics"})


@pytest.mark.asyncio
async def test_memory_control_gate_action_channel_contract() -> None:
    """Memory-control gate should route explicit memory commands to the node."""

    command = await run_memory_control_gate_node(
        _build_state("What do you remember about me?"),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    assert command.goto == "memory_control_node"
    _assert_exact_keys(
        command.update,
        {"route", "memory_control", "diagnostics"},
    )


@pytest.mark.asyncio
async def test_memory_control_node_channel_contract() -> None:
    """Memory-control node should write only operational response channels."""

    state = _build_state("What do you remember about me?")
    state["route"] = "memory_control"
    state["memory_control"] = {"action": {"type": "list"}}

    delta = await run_memory_control_node(
        state,
        cast(Any, _FakeRuntime(memory_mode=MemoryMode.LOCAL)),
    )

    _assert_allowed_keys(
        delta,
        {
            "route",
            "response_style",
            "response_style_source",
            "response_style_type",
            "response_kind",
            "response_text",
            "diagnostics",
            "memory_control",
            "procedural_profile",
        },
    )


@pytest.mark.asyncio
async def test_grounded_lookup_gate_passthrough_channel_contract() -> None:
    """Grounded lookup gate should write only lookup scratch channels normally."""

    command = await run_grounded_lookup_gate_node(
        _build_state("I had a hard day at work."),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    assert command.goto == "load_memory_node"
    _assert_exact_keys(
        command.update,
        {"grounded_lookup", "diagnostics"},
    )


@pytest.mark.asyncio
async def test_grounded_lookup_gate_action_channel_contract() -> None:
    """Grounded lookup gate should route explicit lookup requests to the node."""

    command = await run_grounded_lookup_gate_node(
        _build_state("Can you look up the current 988 rules?"),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    assert command.goto == "grounded_answer_node"
    _assert_exact_keys(
        command.update,
        {"route", "grounded_lookup", "diagnostics"},
    )


@pytest.mark.asyncio
async def test_grounded_answer_node_channel_contract() -> None:
    """Grounded answer node should write only operational response channels."""

    state = _build_state("Can you look up the current 988 rules?")
    state["route"] = "grounded_lookup"
    state["grounded_lookup"] = {"query": "Can you look up the current 988 rules?"}

    delta = await run_grounded_answer_node(
        state,
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(
        delta,
        {
            "route",
            "grounded_lookup",
            "response_style",
            "response_style_source",
            "response_style_type",
            "response_kind",
            "response_text",
            "diagnostics",
        },
    )


@pytest.mark.asyncio
async def test_crisis_resource_lookup_channel_contract() -> None:
    """Crisis resource lookup should write only resource lookup channels."""

    state = _build_state("I need help right now.")
    state["crisis"] = CrisisAssessment(
        level=3,
        reason="imminent_risk",
        needs_crisis_response=True,
    )

    delta = await run_crisis_resource_lookup_node(
        state,
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(
        delta,
        {
            "inferred_location",
            "found_resources",
            "resource_lookup_status",
        },
    )


@pytest.mark.asyncio
async def test_crisis_response_channel_contract() -> None:
    """Crisis response should write only response channels it owns."""

    state = _build_state("I need help right now.")
    state["crisis"] = CrisisAssessment(
        level=3,
        reason="imminent_risk",
        needs_crisis_response=True,
    )

    delta = await run_crisis_response_node(
        state,
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(
        delta,
        {
            "route",
            "response_style",
            "response_style_source",
            "response_style_type",
            "response_kind",
            "response_text",
        },
    )


@pytest.mark.asyncio
async def test_crisis_log_channel_contract() -> None:
    """Crisis log should not write any graph state."""

    state = _build_state("I need help right now.")
    state["crisis"] = CrisisAssessment(
        level=3,
        reason="imminent_risk",
        needs_crisis_response=True,
    )
    state["crisis_audit"] = {
        "crisis_override_kind": "none",
        "crisis_classifier_path": "deterministic",
        "crisis_llm_failure_occurred": False,
    }

    delta = await run_crisis_log_node(
        state,
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(delta, set())


@pytest.mark.asyncio
async def test_load_memory_channel_contract() -> None:
    """Load-memory should write only memory-loading channels."""

    delta = await run_load_memory_node(
        _build_state("Work has been stressful lately."),
        cast(Any, _FakeRuntime(memory_mode=MemoryMode.LOCAL)),
    )

    _assert_exact_keys(
        delta,
        {"working_memory", "session_memory", "procedural_profile", "diagnostics"},
    )


@pytest.mark.asyncio
async def test_dispatch_default_channel_contract() -> None:
    """Default therapeutic dispatch should only write response routing."""

    command = await run_therapeutic_dispatch_node(
        _build_state("I had a rough day at work."),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(command.update, {"response_style", "therapeutic_approach"})


@pytest.mark.asyncio
async def test_dispatch_active_exercise_exit_channel_contract() -> None:
    """Exercise opt-out should clear exercise continuity and write response routing."""

    state = _build_state("This isn't helping, can we just talk?")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_therapeutic_approach": "cbt",
    }
    state["therapeutic_approach"] = "cbt"

    command = await run_therapeutic_dispatch_node(
        state,
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(
        command.update,
        {"response_style", "therapeutic_approach", "exercise_state"},
    )


@pytest.mark.asyncio
async def test_dispatch_llm_mid_exercise_clarifying_channel_contract() -> None:
    """Mid-exercise side-turns should not write unrelated channels."""

    state = _build_state("Wait, what do you mean by grounding?")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_therapeutic_approach": "act",
    }
    state["therapeutic_approach"] = "act"

    command = await run_therapeutic_dispatch_node(
        state,
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="clarifying",
                    therapeutic_approach="cbt",
                )
            ),
        ),
    )

    assert command.goto == "therapeutic_response_node"
    _assert_exact_keys(command.update, {"response_style", "therapeutic_approach"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_style", "message", "extra_state"),
    [
        ("supportive", "I had a rough day.", {}),
        (
            "reflective",
            "Why do I keep ending up in the same fight?",
            {},
        ),
        ("clarifying", "What do you mean?", {}),
        ("psychoeducation", "Why am I reacting like this?", {}),
        ("closing", "Thanks, I should go.", {}),
        (
            "technique",
            "Can you help me examine this thought step by step?",
            {"therapeutic_approach": "cbt"},
        ),
    ],
)
async def test_fixed_shape_therapeutic_mode_channel_contract(
    response_style: str,
    message: str,
    extra_state: dict[str, Any],
) -> None:
    """Fixed-shape therapeutic response node should only write response channels."""

    state = _build_state(message)
    state["response_style"] = response_style
    state.update(extra_state)

    delta = await run_therapeutic_response_node(
        state,
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(
        delta,
        {
            "response_kind",
            "response_text",
            "response_style",
            "response_style_source",
            "response_style_type",
        },
    )


@pytest.mark.asyncio
async def test_guided_exercise_start_channel_contract() -> None:
    """Starting an exercise should only write exercise and response channels."""

    delta = await run_guided_exercise_response_node(
        _build_state("Can you guide me through a grounding exercise?"),
        cast(Any, _FakeRuntime(memory_mode=MemoryMode.INCOGNITO)),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_kind",
            "response_text",
            "response_style",
            "response_style_source",
            "response_style_type",
        },
    )


@pytest.mark.asyncio
async def test_guided_exercise_hold_channel_contract() -> None:
    """Hold responses should not mutate exercise continuity."""

    state = _build_state("Okay, I'm trying.")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_therapeutic_approach": "none",
    }

    delta = await run_guided_exercise_response_node(
        state,
        cast(Any, _FakeRuntime(memory_mode=MemoryMode.INCOGNITO)),
    )

    _assert_exact_keys(
        delta,
        {
            "response_kind",
            "response_text",
            "response_style",
            "response_style_source",
            "response_style_type",
        },
    )


@pytest.mark.asyncio
async def test_guided_exercise_advance_channel_contract() -> None:
    """Advancing an exercise should only write exercise and response channels."""

    state = _build_state("lamp, window, desk")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_therapeutic_approach": "none",
    }

    delta = await run_guided_exercise_response_node(
        state,
        cast(Any, _FakeRuntime(memory_mode=MemoryMode.INCOGNITO)),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_kind",
            "response_text",
            "response_style",
            "response_style_source",
            "response_style_type",
        },
    )


@pytest.mark.asyncio
async def test_guided_exercise_exit_channel_contract() -> None:
    """Exercise exits should clear only exercise continuity plus response fields."""

    state = _build_state("Let's stop for now.")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_therapeutic_approach": "none",
    }

    delta = await run_guided_exercise_response_node(
        state,
        cast(Any, _FakeRuntime(memory_mode=MemoryMode.INCOGNITO)),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_kind",
            "response_text",
            "response_style",
            "response_style_source",
            "response_style_type",
        },
    )


@pytest.mark.asyncio
async def test_guided_exercise_completion_channel_contract() -> None:
    """Exercise completion may additionally request memory persistence."""

    state = _build_state("coffee")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 4,
        "exercise_therapeutic_approach": "none",
    }

    delta = await run_guided_exercise_response_node(
        state,
        cast(Any, _FakeRuntime(memory_mode=MemoryMode.INCOGNITO)),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_kind",
            "response_text",
            "should_persist_memory",
            "response_style",
            "response_style_source",
            "response_style_type",
        },
    )


@pytest.mark.asyncio
async def test_finalize_turn_channel_contract() -> None:
    """Finalize should append transcript only."""

    state = _build_state("Hello")
    state["response_text"] = "Thanks for sharing that."
    state["response_style"] = "supportive"

    delta = await run_finalize_turn_node(
        state,
        cast(Any, _FakeRuntime()),
    )

    _assert_exact_keys(delta, {"transcript"})


@pytest.mark.asyncio
async def test_extract_facts_channel_contract() -> None:
    """Semantic extractor should only return diagnostics deltas."""

    delta = await run_extract_semantic_facts_node(
        _build_state("Hello there."),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(delta, {"diagnostics"})


@pytest.mark.asyncio
async def test_extract_procedural_channel_contract() -> None:
    """Procedural extractor should only return diagnostics deltas."""

    delta = await run_extract_procedural_rules_node(
        _build_state("Hello there."),
        cast(Any, _FakeRuntime(llm_client=None)),
    )

    _assert_exact_keys(delta, {"diagnostics"})
