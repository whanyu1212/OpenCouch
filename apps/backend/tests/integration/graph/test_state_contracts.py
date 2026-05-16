"""Guard tests for node-level state channel ownership.

These tests lock in the post-schema-split contract: each node may write only
the top-level channels it explicitly owns. If a node starts mutating an
unrelated channel, the failure should be immediate and local.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

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
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.grounded_answer import run_grounded_answer_node
from agent.nodes.load_memory import run_load_memory_node
from agent.nodes.memory_control import run_memory_control_node
from agent.nodes.turn_dispatch import run_turn_dispatch_node
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentGraphOutputState, AgentState
from agent.therapeutic.dispatch import run_therapeutic_dispatch_node
from agent.therapeutic.graph import (
    TherapeuticSubgraphInput,
    TherapeuticSubgraphOutput,
    build_therapeutic_subgraph,
)
from agent.therapeutic.exercises.node import (
    run_guided_exercise_response_node as _run_guided_exercise_response_node,
)
from agent.therapeutic.exercises.types import ExerciseStepDecision
from agent.therapeutic.response import run_therapeutic_response_node
from llm.base import BaseLLMClient, StructuredResponseT


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


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the guided exercise node with a no-op test stream writer.

    Args:
        state (AgentState): Test graph state.
        runtime (Any): Runtime stub.
        **kwargs (Any): Optional node keyword overrides.

    Returns:
        dict[str, Any]: Node delta.
    """

    kwargs.setdefault("stream_writer_factory", lambda: lambda _: None)
    return await _run_guided_exercise_response_node(state, runtime, **kwargs)


class _FakeDispatchLLM(BaseLLMClient):
    """Structured-output fake for dispatcher contract tests."""

    def __init__(
        self,
        *,
        response_style: str,
        therapeutic_approach: str,
        exercise_start_basis: str = "ambiguous_or_none",
        turn_route: str = "therapeutic",
        memory_action_type: str | None = None,
        lookup_query: str | None = None,
        active_flow_action: str = "none",
        memory_reference_mode: str = "none",
        step_state: str = "hold",
        crisis_level: int = 0,
        stream_text: str = "unused",
    ) -> None:
        self.response_style = response_style
        self.therapeutic_approach = therapeutic_approach
        self.exercise_start_basis = exercise_start_basis
        self.turn_route = turn_route
        self.memory_action_type = memory_action_type
        self.lookup_query = lookup_query
        self.active_flow_action = active_flow_action
        self.memory_reference_mode = memory_reference_mode
        self.step_state = step_state
        self.crisis_level = crisis_level
        self.stream_text = stream_text

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return self.stream_text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield self.stream_text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        if response_schema.__name__ == "CrisisAssessmentSchema":
            return response_schema(  # type: ignore[call-arg,return-value]
                level=self.crisis_level,
                confidence="high",
                reason="contract test crisis assessment",
                needs_crisis_response=self.crisis_level >= 2,
                needs_clarification=self.crisis_level == 1,
            )
        if response_schema.__name__ == "ExerciseSelectionDecision":
            return cast(
                StructuredResponseT,
                response_schema(
                    exercise_type="grounding_5_4_3_2_1",
                    reasoning="contract test exercise selection",
                    confidence="high",
                ),
            )
        if response_schema.__name__ == "ExerciseStepDecision":
            return cast(
                StructuredResponseT,
                ExerciseStepDecision(
                    step_state=self.step_state,  # type: ignore[arg-type]
                    reasoning="contract test step state",
                    confidence="high",
                ),
            )
        if response_schema.__name__ == "TurnDispatchDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                route=self.turn_route,
                memory_action_type=self.memory_action_type,
                query=self.lookup_query,
                active_flow_action=self.active_flow_action,
                memory_reference_mode=self.memory_reference_mode,
                reasoning="contract test turn dispatch",
                confidence="high",
            )
        return cast(
            StructuredResponseT,
            DispatchDecision(
                response_style=self.response_style,  # type: ignore[arg-type]
                therapeutic_approach=self.therapeutic_approach,  # type: ignore[arg-type]
                exercise_start_basis=self.exercise_start_basis,  # type: ignore[arg-type]
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
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                    crisis_level=2,
                )
            ),
        ),
    )

    assert command.goto == "crisis_resource_lookup_node"
    _assert_exact_keys(
        command.update,
        {
            "crisis",
            "route",
            "crisis_audit",
            "diagnostics",
            "exercise_state",
            "memory_control",
            "turn_lifecycle",
        },
    )
    assert command.update["exercise_state"] == {
        "exercise_type": None,
        "exercise_step": None,
        "exercise_step_id": None,
        "exercise_version": None,
        "exercise_therapeutic_approach": None,
    }
    assert command.update["memory_control"] == {
        "action": {},
        "pending_action": None,
    }
    assert command.update["turn_lifecycle"] == {
        "active_flow": "none",
        "action": "none",
    }


@pytest.mark.asyncio
async def test_crisis_gate_therapeutic_path_channel_contract() -> None:
    """Crisis gate should not write response channels on safe turns."""

    command = await run_crisis_gate_node(
        _build_state("I had a hard day at work."),
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                    crisis_level=0,
                )
            ),
        ),
    )

    assert command.goto == "turn_dispatch_node"
    _assert_exact_keys(
        command.update,
        {"crisis", "route", "crisis_audit", "diagnostics"},
    )


@pytest.mark.asyncio
async def test_turn_dispatch_therapeutic_channel_contract() -> None:
    """Turn dispatch should route ordinary support to memory loading."""

    command = await run_turn_dispatch_node(
        _build_state("I had a hard day at work."),
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                )
            ),
        ),
    )

    assert command.goto == "load_memory_node"
    _assert_exact_keys(
        command.update,
        {
            "route",
            "turn_lifecycle",
            "memory_control",
            "grounded_lookup",
            "memory_reference",
            "diagnostics",
        },
    )
    assert command.update["memory_reference"] == {"mode": "none"}


@pytest.mark.asyncio
async def test_turn_dispatch_marks_explicit_memory_reference_turns() -> None:
    """Turn dispatch should carry one-turn permission for recall requests."""

    command = await run_turn_dispatch_node(
        _build_state("What did we work out for the presentation?"),
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                    memory_reference_mode="explicit",
                )
            ),
        ),
    )

    assert command.goto == "load_memory_node"
    assert command.update["memory_reference"] == {"mode": "explicit"}


@pytest.mark.asyncio
async def test_turn_dispatch_memory_control_channel_contract() -> None:
    """Turn dispatch should route explicit memory commands to the node."""

    command = await run_turn_dispatch_node(
        _build_state("What do you remember about me?"),
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                    turn_route="memory_control",
                    memory_action_type="list",
                )
            ),
        ),
    )

    assert command.goto == "memory_control_node"
    _assert_exact_keys(
        command.update,
        {
            "route",
            "turn_lifecycle",
            "memory_control",
            "grounded_lookup",
            "memory_reference",
            "diagnostics",
        },
    )


@pytest.mark.asyncio
async def test_turn_dispatch_memory_mutation_clears_exercise_state() -> None:
    """Destructive memory commands should explicitly end active exercises."""

    state = _build_state("Forget what you saved about presentations.")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_step_id": "see",
        "exercise_version": 1,
        "exercise_therapeutic_approach": "dbt_skills",
    }

    command = await run_turn_dispatch_node(
        state,
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                    turn_route="memory_control",
                    memory_action_type="forget_by_query",
                    lookup_query="presentations",
                    active_flow_action="clear",
                )
            ),
        ),
    )

    assert command.goto == "memory_control_node"
    _assert_exact_keys(
        command.update,
        {
            "route",
            "turn_lifecycle",
            "memory_control",
            "grounded_lookup",
            "memory_reference",
            "diagnostics",
            "exercise_state",
        },
    )
    assert command.update["exercise_state"] == {
        "exercise_type": None,
        "exercise_step": None,
        "exercise_step_id": None,
        "exercise_version": None,
        "exercise_therapeutic_approach": None,
    }


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
            "response_text",
            "diagnostics",
            "memory_control",
            "procedural_profile",
        },
    )


@pytest.mark.asyncio
async def test_turn_dispatch_grounded_lookup_channel_contract() -> None:
    """Turn dispatch should route factual lookup requests to the lookup node."""

    command = await run_turn_dispatch_node(
        _build_state("Can you look up the current 988 rules?"),
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                    turn_route="grounded_lookup",
                    lookup_query="current 988 rules",
                )
            ),
        ),
    )

    assert command.goto == "grounded_answer_node"
    _assert_exact_keys(
        command.update,
        {
            "route",
            "turn_lifecycle",
            "memory_control",
            "grounded_lookup",
            "memory_reference",
            "diagnostics",
        },
    )


@pytest.mark.asyncio
async def test_grounded_answer_node_channel_contract() -> None:
    """Grounded answer node should write only operational response channels."""

    state = _build_state("Can you look up the current 988 rules?")
    state["route"] = "grounded_lookup"
    state["grounded_lookup"] = {"query": "Can you look up the current 988 rules?"}

    async def _lookup(
        state: AgentState,
        *,
        llm_client: BaseLLMClient,
        query: str,
    ) -> tuple[str, str]:
        _ = (state, llm_client, query)
        return "Verified answer.\n\nSources:\n- Official source", "answered"

    with patch("agent.turn_branches.answer_factual_lookup", _lookup):
        delta = await run_grounded_answer_node(
            state,
            cast(
                Any,
                _FakeRuntime(
                    llm_client=_FakeDispatchLLM(
                        response_style="supportive",
                        therapeutic_approach="none",
                    )
                ),
            ),
        )

    _assert_exact_keys(
        delta,
        {
            "route",
            "grounded_lookup",
            "response_style",
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

    async def _lookup(
        state: AgentState,
        *,
        llm_client: BaseLLMClient,
    ) -> tuple[str, list[dict[str, str]], str]:
        return "", [], "no_location"

    with patch("agent.nodes.crisis_resource_lookup.find_crisis_resources", _lookup):
        delta = await run_crisis_resource_lookup_node(
            state,
            cast(
                Any,
                _FakeRuntime(
                    llm_client=_FakeDispatchLLM(
                        response_style="supportive",
                        therapeutic_approach="pfa",
                    )
                ),
            ),
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

    with patch(
        "agent.nodes.crisis_response.get_stream_writer",
        return_value=lambda _: None,
    ):
        delta = await run_crisis_response_node(
            state,
            cast(
                Any,
                _FakeRuntime(
                    llm_client=_FakeDispatchLLM(
                        response_style="supportive",
                        therapeutic_approach="pfa",
                    )
                ),
            ),
        )

    _assert_exact_keys(
        delta,
        {
            "route",
            "response_style",
            "response_text",
        },
    )


@pytest.mark.asyncio
async def test_crisis_response_requires_llm_client() -> None:
    """Crisis response should fail loudly when no response LLM is configured."""

    state = _build_state("I need help but I won't share where I am.")
    state["crisis"] = CrisisAssessment(
        level=3,
        reason="imminent_risk",
        needs_crisis_response=True,
    )
    state["resource_lookup_status"] = "location_refused"

    with pytest.raises(RuntimeError, match="requires an LLM client"):
        await run_crisis_response_node(
            state,
            cast(Any, _FakeRuntime(llm_client=None)),
        )


@pytest.mark.asyncio
async def test_crisis_response_prefers_response_llm() -> None:
    """Crisis response should use the response writer when one is configured."""

    state = _build_state("I need help right now.")
    state["crisis"] = CrisisAssessment(
        level=3,
        reason="imminent_risk",
        needs_crisis_response=True,
    )

    with patch(
        "agent.nodes.crisis_response.get_stream_writer",
        return_value=lambda _: None,
    ):
        delta = await run_crisis_response_node(
            state,
            cast(
                Any,
                _FakeRuntime(
                    llm_client=_FakeDispatchLLM(
                        response_style="supportive",
                        therapeutic_approach="pfa",
                        stream_text="control response",
                    ),
                    response_llm=_FakeDispatchLLM(
                        response_style="supportive",
                        therapeutic_approach="pfa",
                        stream_text="writer response",
                    ),
                ),
            ),
        )

    assert delta["response_text"] == "writer response"


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
        "crisis_classifier_path": "llm_primary",
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
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                )
            ),
        ),
    )

    _assert_allowed_keys(
        command.update,
        {
            "response_style",
            "therapeutic_approach",
            "session_progress",
            "response_guidance",
            "diagnostics",
        },
    )


@pytest.mark.asyncio
async def test_dispatch_closing_channel_contract() -> None:
    """Closing dispatch may add the session-finalization suggestion signal."""

    command = await run_therapeutic_dispatch_node(
        _build_state("Thanks, I need to go."),
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="closing",
                    therapeutic_approach="none",
                )
            ),
        ),
    )

    _assert_allowed_keys(
        command.update,
        {
            "response_style",
            "therapeutic_approach",
            "session_action",
            "diagnostics",
        },
    )
    assert command.update["session_action"] == "suggest_end_session"


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
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="none",
                )
            ),
        ),
    )

    _assert_allowed_keys(
        command.update,
        {"response_style", "therapeutic_approach", "exercise_state", "diagnostics"},
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
    _assert_allowed_keys(
        command.update,
        {"response_style", "therapeutic_approach", "diagnostics"},
    )


@pytest.mark.asyncio
async def test_dispatch_preserves_exercise_on_active_flow_side_turn() -> None:
    """Active-flow preserve should keep exercise state through support side-turns."""

    state = _build_state("I feel silly before I answer the step.")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_therapeutic_approach": "dbt_skills",
    }
    state["therapeutic_approach"] = "dbt_skills"
    state["turn_lifecycle"] = {
        "active_flow": "guided_exercise",
        "action": "preserve",
    }

    command = await run_therapeutic_dispatch_node(
        state,
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="pfa",
                )
            ),
        ),
    )

    assert command.goto == "therapeutic_response_node"
    assert command.update["response_style"] == "supportive"
    assert "exercise_state" not in command.update


@pytest.mark.asyncio
async def test_dispatch_active_flow_continue_forces_guided_exercise() -> None:
    """Active-flow continue should keep continuation inside the exercise runner."""

    state = _build_state("I can see my desk and window.")
    state["exercise_state"] = {
        "exercise_type": "grounding_5_4_3_2_1",
        "exercise_step": 0,
        "exercise_therapeutic_approach": "dbt_skills",
    }
    state["therapeutic_approach"] = "pfa"
    state["turn_lifecycle"] = {
        "active_flow": "guided_exercise",
        "action": "continue",
    }

    command = await run_therapeutic_dispatch_node(
        state,
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="supportive",
                    therapeutic_approach="pfa",
                )
            ),
        ),
    )

    assert command.goto == "guided_exercise_response_node"
    assert command.update["response_style"] == "guided_exercise"
    assert command.update["therapeutic_approach"] == "dbt_skills"
    assert "exercise_state" not in command.update


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed-shape therapeutic response node should only write response channels."""

    from agent.therapeutic import response

    monkeypatch.setattr(response, "get_stream_writer", lambda: lambda _: None)

    state = _build_state(message)
    state["response_style"] = response_style
    state.update(extra_state)

    delta = await run_therapeutic_response_node(
        state,
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style=response_style,
                    therapeutic_approach=extra_state.get(
                        "therapeutic_approach", "none"
                    ),
                )
            ),
        ),
    )

    _assert_exact_keys(
        delta,
        {
            "response_text",
            "response_style",
        },
    )


@pytest.mark.asyncio
async def test_guided_exercise_start_channel_contract() -> None:
    """Starting an exercise should only write exercise and response channels."""

    delta = await run_guided_exercise_response_node(
        _build_state("Can you guide me through a grounding exercise?"),
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="guided_exercise",
                    therapeutic_approach="none",
                ),
                memory_mode=MemoryMode.INCOGNITO,
            ),
        ),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_text",
            "response_style",
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
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="guided_exercise",
                    therapeutic_approach="none",
                    step_state="hold",
                ),
                memory_mode=MemoryMode.INCOGNITO,
            ),
        ),
    )

    _assert_exact_keys(
        delta,
        {
            "response_text",
            "response_style",
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
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="guided_exercise",
                    therapeutic_approach="none",
                    step_state="complete",
                ),
                memory_mode=MemoryMode.INCOGNITO,
            ),
        ),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_text",
            "response_style",
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
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="guided_exercise",
                    therapeutic_approach="none",
                    step_state="exit",
                ),
                memory_mode=MemoryMode.INCOGNITO,
            ),
        ),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_text",
            "response_style",
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
        cast(
            Any,
            _FakeRuntime(
                llm_client=_FakeDispatchLLM(
                    response_style="guided_exercise",
                    therapeutic_approach="none",
                    step_state="complete",
                ),
                memory_mode=MemoryMode.INCOGNITO,
            ),
        ),
    )

    _assert_exact_keys(
        delta,
        {
            "exercise_state",
            "response_text",
            "response_style",
        },
    )


@pytest.mark.asyncio
async def test_finalize_turn_channel_contract() -> None:
    """Finalize should append the transcript and stamp a finalize-time
    marker into diagnostics. The marker is the boundary signal
    ``stamp_turn_total_ms`` reads to compute ``post_finalize_ms``; it
    is stripped from the public diagnostics before the turn returns,
    but it's a legitimate channel write at the node level."""

    state = _build_state("Hello")
    state["response_text"] = "Thanks for sharing that."
    state["response_style"] = "supportive"

    delta = await run_finalize_turn_node(
        state,
        cast(Any, _FakeRuntime()),
    )

    _assert_exact_keys(delta, {"transcript", "diagnostics"})
    assert set(delta["diagnostics"].keys()) == {"finalize_done_at_monotonic"}
