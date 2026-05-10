"""Guard tests for reducer-backed transcript state.

Phase C of the LangGraph best-practice alignment plan. These tests
verify that:

1. ``transcript`` uses an ``operator.add`` reducer so LangGraph accumulates
   turns via the checkpointer instead of manual reconstruction every turn.
2. ``build_initial_state`` emits only the current user turn (not the
   full prior history) — the checkpointer restores prior turns.
3. ``run_finalize_turn_node`` returns a single-element delta (not a
   full reconstructed list) — the reducer appends it.
4. Multi-turn sessions accumulate correctly via the checkpointer.
5. Exercise state persists across turns via the checkpoint without
   the manual carry-forward hack in persistence.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import operator
import typing
from typing import Annotated, Any, NotRequired, get_type_hints

import pytest

from agent.conversation import format_recent_history, get_recent_history, get_transcript
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.models import DispatchDecision
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.graph import build_agent_workflow, build_initial_state
from agent.models import AgentInput, MessageRole
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.persistence import PersistentAgentRuntime
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentGraphOutputState, AgentState
from llm.base import BaseLLMClient, StructuredResponseT


# ── Reducer annotation tests ───────────────────────────────────────────────


def _get_reducer(state_class: type, field: str) -> Any:
    """Extract the reducer function from a TypedDict field's Annotated metadata.

    Unwraps NotRequired if present.
    """
    hints = get_type_hints(state_class, include_extras=True)
    hint = hints.get(field)
    if hint is None:
        return None

    inner = hint
    origin = typing.get_origin(inner)
    if origin is NotRequired:
        args = typing.get_args(inner)
        inner = args[0] if args else inner

    if typing.get_origin(inner) is not Annotated:
        return None
    metadata = getattr(inner, "__metadata__", ())
    return metadata[0] if metadata else None


def test_session_progress_uses_merge_reducer() -> None:
    """``AgentState.session_progress`` should have a plain merge reducer."""
    from agent.state import _merge_dicts

    reducer = _get_reducer(AgentState, "session_progress")
    assert reducer is _merge_dicts, (
        f"session_progress reducer is {reducer!r}, expected _merge_dicts. "
    )


def test_exercise_state_uses_merge_reducer() -> None:
    """``AgentState.exercise_state`` should have a merge reducer."""

    from agent.state import _merge_dicts

    reducer = _get_reducer(AgentState, "exercise_state")
    assert reducer is _merge_dicts, (
        f"exercise_state reducer is {reducer!r}, expected _merge_dicts. "
        f"Without a merge reducer, build_initial_state overwrites the "
        f"checkpoint's exercise_state dict and destroys exercise continuity."
    )


def test_therapeutic_approach_uses_default_overwrite_semantics() -> None:
    """``AgentState.therapeutic_approach`` should use overwrite semantics."""

    reducer = _get_reducer(AgentState, "therapeutic_approach")
    assert reducer is None, (
        f"therapeutic_approach reducer is {reducer!r}, expected None. "
        f"Turn-scoped channels should overwrite by default so stale values "
        f"do not survive checkpoint merges into later turns."
    )


def test_transcript_uses_add_reducer() -> None:
    """``AgentState.transcript`` should have ``operator.add`` as its reducer."""
    reducer = _get_reducer(AgentState, "transcript")
    assert reducer is operator.add, (
        f"transcript reducer is {reducer!r}, expected operator.add"
    )


def test_conversation_helpers_prefer_transcript_over_history() -> None:
    """Conversation helpers should read transcript before legacy history."""

    state = {
        "transcript": [
            {"role": "user", "content": "new user turn"},
            {"role": "assistant", "content": "new assistant turn"},
        ],
        "history": [{"role": "user", "content": "legacy user turn"}],
    }

    assert get_transcript(state) == state["transcript"]
    assert get_recent_history(state, limit=1) == [state["transcript"][1]]
    assert format_recent_history(state, limit=2) == (
        "user: new user turn\nassistant: new assistant turn"
    )


def test_conversation_helpers_fall_back_to_history() -> None:
    """Conversation helpers should support older state without transcript."""

    state = {"history": [{"role": "user", "content": "legacy user turn"}]}

    assert get_transcript(state) == state["history"]
    assert format_recent_history(state) == "user: legacy user turn"


def test_langgraph_compiles_therapeutic_approach_as_last_value_channel() -> None:
    """LangGraph should compile ``therapeutic_approach`` as an overwrite channel."""

    graph = build_agent_workflow()
    channels = graph.channels
    assert "therapeutic_approach" in channels, (
        "therapeutic_approach not found in compiled graph channels"
    )
    channel = channels["therapeutic_approach"]
    assert type(channel).__name__ == "LastValue", (
        f"therapeutic_approach channel is {type(channel).__name__}, "
        f"expected LastValue. Turn-scoped channels should overwrite across "
        f"checkpoints instead of accumulating."
    )


def test_parent_graph_declares_explicit_input_and_output_schemas() -> None:
    """The top-level graph should expose split LangGraph I/O schemas."""

    graph = build_agent_workflow()
    assert graph.builder.input_schema is AgentGraphInputState
    assert graph.builder.output_schema is AgentGraphOutputState


# ── build_initial_state tests ──────────────────────────────────────────────


def test_build_initial_state_emits_only_current_user_turn_to_transcript() -> None:
    """build_initial_state should emit only the current user message to transcript.

    The checkpointer is responsible for restoring prior turns via the
    reducer. Emitting the full history would cause duplication when
    the reducer appends to the checkpoint.
    """

    agent_input = AgentInput(
        message="How are you?",
        history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "I feel sad"},
            {"role": "assistant", "content": "Tell me more"},
        ],
    )
    state = build_initial_state(agent_input)

    assert "history" not in state
    assert len(state.get("transcript", [])) == 1
    assert state["transcript"][0]["content"] == "How are you?"


def test_build_initial_state_turn_count_still_correct() -> None:
    """turn_count should still be accurate even though the full history
    is no longer in the transcript.

    The count should reflect the total number of user turns including
    the current one — derived from AgentInput.history, not from
    the transcript.
    """

    agent_input = AgentInput(
        message="Third message",
        history=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Reply 2"},
        ],
    )
    state = build_initial_state(agent_input)

    # 2 prior user turns + 1 current = 3
    assert state["session_progress"]["turn_count"] == 3


def test_build_initial_state_session_progress_contract_is_minimal() -> None:
    """build_initial_state should seed only owned session-progress fields."""

    state = build_initial_state(
        AgentInput(message="Fourth message"),
        prior_turn_count=3,
    )

    assert state["session_progress"] == {
        "turn_count": 4,
        "is_guest": False,
    }


# ── finalize_turn_node tests ──────────────────────────────────────────────


class _FakeRuntime:
    def __init__(self) -> None:
        self.context = WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


class _GuidedExerciseLLM(BaseLLMClient):
    """Fake control LLM that routes safe therapeutic turns to guided exercise."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "fake text"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "Good. Now remind yourself that you're not alone in this."

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        if response_schema.__name__ == "CrisisAssessmentSchema":
            return response_schema(  # type: ignore[call-arg,return-value]
                level=0,
                confidence="high",
                reason="safe therapeutic support request",
                needs_crisis_response=False,
                needs_clarification=False,
            )
        if response_schema.__name__ == "ExerciseStepDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                step_state="complete",
                reasoning="The user confirmed completing the previous step.",
                confidence="high",
            )
        if response_schema.__name__ == "ExerciseSelectionDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                exercise_type="self_compassion_break",
                reasoning="Self-critical language maps to self-compassion.",
                confidence="high",
            )
        if response_schema.__name__ == "TurnDispatchDecision":
            active_flow_action = (
                "continue" if "Active flow: guided_exercise" in prompt else "none"
            )
            return response_schema(  # type: ignore[call-arg,return-value]
                route="therapeutic",
                active_flow_action=active_flow_action,
                reasoning="safe guided-exercise test turn",
                confidence="high",
            )
        return typing.cast(
            StructuredResponseT,
            DispatchDecision(
                response_style="guided_exercise",
                therapeutic_approach="act",
                exercise_start_basis="explicit_user_request",
                reasoning="guided exercise requested or active",
                confidence="high",
            ),
        )


class _SupportiveLLM(BaseLLMClient):
    """Fake LLM that keeps normal turns on the supportive route."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "fake supportive text"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "fake "
        yield "supportive text"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__
        if schema_name == "CrisisAssessmentSchema":
            return response_schema(  # type: ignore[call-arg,return-value]
                level=0,
                confidence="high",
                reason="safe reducer transcript test turn",
                needs_crisis_response=False,
                needs_clarification=False,
            )
        if schema_name == "DispatchDecision":
            return typing.cast(
                StructuredResponseT,
                DispatchDecision(
                    response_style="supportive",
                    therapeutic_approach="none",
                    exercise_start_basis="ambiguous_or_none",
                    reasoning="normal supportive test turn",
                    confidence="high",
                ),
            )
        if schema_name == "ExtractionResult":
            return response_schema(  # type: ignore[call-arg,return-value]
                facts=[],
                reason="no semantic facts in reducer transcript test",
            )
        if schema_name == "ProceduralExtractionResult":
            return response_schema(  # type: ignore[call-arg,return-value]
                rules=[],
                reason="no procedural rules in reducer transcript test",
            )
        if schema_name == "TurnDispatchDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                route="therapeutic",
                reasoning="ordinary supportive turn",
                confidence="high",
            )
        raise RuntimeError(f"_SupportiveLLM unexpected schema {schema_name}")


@pytest.mark.asyncio
async def test_finalize_returns_single_element_delta() -> None:
    """finalize_turn_node should return a 1-element transcript delta."""

    state: dict[str, Any] = {
        "transcript": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Help me"},
        ],
        "response_text": "Of course, what's on your mind?",
        "response_style": "supportive",
    }

    delta = await run_finalize_turn_node(state, _FakeRuntime())  # type: ignore[arg-type]

    # Delta should contain exactly 1 entry — the new assistant turn.
    assert len(delta["transcript"]) == 1, (
        f"Expected 1-element transcript delta, got {len(delta['transcript'])}. "
        f"finalize_turn should return only the new assistant turn; "
        f"the reducer appends it to the checkpoint."
    )
    assert delta["transcript"][0]["role"] == MessageRole.ASSISTANT.value
    assert delta["transcript"][0]["content"] == "Of course, what's on your mind?"
    assert delta["transcript"][0]["response_style"] == "supportive"


# ── Multi-turn accumulation via checkpointer ───────────────────────────────


@pytest.mark.asyncio
async def test_multi_turn_transcript_accumulates_via_checkpointer() -> None:
    """Running 3 turns on the same thread should produce a transcript
    with 6 entries (3 user + 3 assistant), accumulated by the reducer
    and checkpointer — not by manual reconstruction.
    """

    llm = _SupportiveLLM()
    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        await runtime.run_turn(thread_id="t-accum", message="Turn 1", llm_client=llm)
        await runtime.run_turn(thread_id="t-accum", message="Turn 2", llm_client=llm)
        result = await runtime.run_turn(
            thread_id="t-accum",
            message="Turn 3",
            llm_client=llm,
        )

    transcript = result.state.get("transcript", [])
    user_turns = [t for t in transcript if t.get("role") == "user"]
    assistant_turns = [t for t in transcript if t.get("role") == "assistant"]

    assert len(user_turns) == 3, f"Expected 3 user turns, got {len(user_turns)}"
    assert len(assistant_turns) == 3, (
        f"Expected 3 assistant turns, got {len(assistant_turns)}"
    )
    assert len(transcript) == 6, f"Expected 6 total entries, got {len(transcript)}"

    # Verify ordering: alternating user/assistant pairs.
    for i in range(3):
        assert transcript[i * 2]["role"] == "user"
        assert transcript[i * 2 + 1]["role"] == "assistant"

    # Verify content of user turns.
    assert transcript[0]["content"] == "Turn 1"
    assert transcript[2]["content"] == "Turn 2"
    assert transcript[4]["content"] == "Turn 3"


# ── Exercise state persistence without manual carry-forward ────────────────


@pytest.mark.asyncio
async def test_exercise_state_persists_across_turns() -> None:
    """exercise_type and exercise_step set during turn N must be visible
    in turn N+1 without any manual carry-forward in persistence.py.

    The exercise_state field uses a merge reducer so build_initial_state's
    fresh session_progress dict (with turn_count) coexists with the
    checkpoint's exercise_state (which carries exercise_type/exercise_step)
    rather than overwriting it.
    """

    supportive_llm = _SupportiveLLM()
    guided_llm = _GuidedExerciseLLM()
    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        # Turn 1: run a normal turn to create a checkpoint.
        await runtime.run_turn(
            thread_id="t-exercise",
            message="Hello",
            llm_client=supportive_llm,
        )

        # Inject exercise state into the checkpoint by reading the
        # current state and updating exercise_state directly. This simulates
        # what the guided_exercise_response_node does during a real
        # exercise turn.
        state1 = await runtime.get_state("t-exercise")
        assert state1 is not None
        state1["exercise_state"]["exercise_type"] = "grounding_5_4_3_2_1"
        state1["exercise_state"]["exercise_step"] = 1

        # Write the modified state back via the graph's update_state.
        graph = runtime._get_graph()
        config = runtime._config_for_thread("t-exercise")
        await graph.aupdate_state(
            config,
            {"exercise_state": state1["exercise_state"]},
            as_node="finalize_turn_node",
        )

        # Turn 2: run another turn — exercise state must survive.
        result2 = await runtime.run_turn(
            thread_id="t-exercise",
            message="Continue the exercise",
            llm_client=guided_llm,
        )
        state2 = result2.state

        exercise_state2 = state2.get("exercise_state", {})
        assert exercise_state2.get("exercise_type") == "grounding_5_4_3_2_1", (
            f"exercise_type lost across turns. exercise_state={exercise_state2}"
        )
        # exercise_step may advance during the turn (the guided exercise
        # node processes a step), so assert it's present rather than
        # checking an exact value.
        assert exercise_state2.get("exercise_step") is not None, (
            f"exercise_step lost across turns. exercise_state={exercise_state2}"
        )
        session_progress2 = state2.get("session_progress", {})
        assert session_progress2.get("turn_count", 0) >= 2, (
            f"turn_count should be >= 2, got {session_progress2.get('turn_count')}"
        )


@pytest.mark.asyncio
async def test_self_compassion_exercise_continues_across_turns() -> None:
    """A self-compassion exercise should not restart as generic grounding.

    This covers the real user-facing streaming path: turn 1 starts a
    self-compassion break, then a short confirmation on turn 2 should advance
    that same exercise instead of falling back to the default 5-4-3-2-1
    grounding exercise.
    """

    llm = _GuidedExerciseLLM()
    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
        default_llm_client=llm,
    ) as runtime:
        events1 = [
            event
            async for event in runtime.run_turn_stream(
                thread_id="t-self-compassion",
                message=(
                    "I'm being really hard on myself right now. "
                    "Is there something we can do about that?"
                ),
                llm_client=llm,
            )
        ]

        state1 = await runtime.get_state("t-self-compassion")
        assert state1 is not None
        exercise_state1 = state1.get("exercise_state", {})
        assert exercise_state1.get("exercise_type") == "self_compassion_break"
        assert exercise_state1.get("exercise_step") == 0

        events2 = [
            event
            async for event in runtime.run_turn_stream(
                thread_id="t-self-compassion",
                message="done that",
                llm_client=llm,
            )
        ]

        state2 = await runtime.get_state("t-self-compassion")

    done1 = next(event for event in events1 if event.type == "done")
    done2 = next(event for event in events2 if event.type == "done")
    assert done1.output.response_style == "guided_exercise"
    assert state2 is not None
    exercise_state2 = state2.get("exercise_state", {})
    assert exercise_state2.get("exercise_type") == "self_compassion_break"
    assert exercise_state2.get("exercise_step") == 1
    assert "not alone" in done2.output.response_text.lower()
    assert "5-4-3-2-1" not in done2.output.response_text


@pytest.mark.asyncio
async def test_stale_turn_scoped_keys_do_not_survive_next_turn() -> None:
    """Checkpointed turn-scoped keys without carry-forward should be cleared."""

    llm = _SupportiveLLM()
    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        await runtime.run_turn(thread_id="t-routing", message="Hello", llm_client=llm)

        state1 = await runtime.get_state("t-routing")
        assert state1 is not None
        graph = runtime._get_graph()
        config = runtime._config_for_thread("t-routing")
        await graph.aupdate_state(
            config,
            {
                "therapeutic_approach": "cbt",
                "inferred_location": "Singapore",
                "found_resources": [{"name": "Hotline"}],
                "resource_lookup_status": "found",
            },
            as_node="finalize_turn_node",
        )

        result2 = await runtime.run_turn(
            thread_id="t-routing",
            message="I had a hard day at work.",
            llm_client=llm,
        )

    assert result2.state.get("therapeutic_approach") == "none"
    assert result2.state.get("inferred_location") == ""
    assert result2.state.get("found_resources") == []
    assert result2.state.get("resource_lookup_status") == "not_attempted"


@pytest.mark.asyncio
async def test_run_turn_does_not_deserialize_full_history_for_turn_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent turns should derive turn_count from checkpoint session_progress,
    not by calling ``get_history()`` and rebuilding the transcript.
    """

    llm = _SupportiveLLM()
    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        await runtime.run_turn(
            thread_id="t-no-history",
            message="First",
            llm_client=llm,
        )

        async def _fail_get_history(thread_id: str) -> list[Any]:
            raise AssertionError(f"get_history should not be called for {thread_id}")

        monkeypatch.setattr(runtime, "get_history", _fail_get_history)
        result = await runtime.run_turn(
            thread_id="t-no-history",
            message="Second",
            llm_client=llm,
        )

    assert result.state["session_progress"]["turn_count"] == 2


@pytest.mark.asyncio
async def test_run_turn_stream_does_not_deserialize_full_history_for_turn_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming turns should use the same turn-count path as ``run_turn``."""

    llm = _SupportiveLLM()
    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        await runtime.run_turn(
            thread_id="t-no-history-stream",
            message="First",
            llm_client=llm,
        )

        async def _fail_get_history(thread_id: str) -> list[Any]:
            raise AssertionError(f"get_history should not be called for {thread_id}")

        monkeypatch.setattr(runtime, "get_history", _fail_get_history)

        done_event = None
        async for event in runtime.run_turn_stream(
            thread_id="t-no-history-stream",
            message="Second",
            llm_client=llm,
        ):
            if hasattr(event, "output"):
                done_event = event
        state = await runtime.get_state("t-no-history-stream")
        assert state is not None
        assert state["session_progress"]["turn_count"] == 2

    assert done_event is not None
