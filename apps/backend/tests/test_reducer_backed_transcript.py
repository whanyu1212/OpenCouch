"""Guard tests for reducer-backed transcript/history state.

Phase C of the LangGraph best-practice alignment plan. These tests
verify that:

1. ``history`` and ``transcript`` use ``operator.add`` reducers so
   LangGraph accumulates turns via the checkpointer instead of
   manual reconstruction every turn.
2. ``build_initial_state`` emits only the current user turn (not the
   full prior history) — the checkpointer restores prior turns.
3. ``run_finalize_turn_node`` returns a single-element delta (not a
   full reconstructed list) — the reducer appends it.
4. Multi-turn sessions accumulate correctly via the checkpointer.
5. Exercise state persists across turns via the checkpoint without
   the manual carry-forward hack in persistence.py.
"""

from __future__ import annotations

import operator
import typing
from typing import Annotated, Any, NotRequired, get_type_hints

import pytest

from agent.memory.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.graph import build_initial_state
from agent.models import AgentInput, MessageRole
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.persistence import PersistentAgentRuntime
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


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


def test_history_uses_add_reducer() -> None:
    """``AgentState.history`` should have ``operator.add`` as its reducer."""
    reducer = _get_reducer(AgentState, "history")
    assert reducer is operator.add, (
        f"history reducer is {reducer!r}, expected operator.add"
    )


def test_progress_uses_merge_reducer() -> None:
    """``AgentState.progress`` should have a merge reducer so that exercise
    state persists across turns without manual carry-forward."""
    from agent.state import _merge_dicts

    reducer = _get_reducer(AgentState, "progress")
    assert reducer is _merge_dicts, (
        f"progress reducer is {reducer!r}, expected _merge_dicts. "
        f"Without a merge reducer, build_initial_state overwrites the "
        f"checkpoint's progress dict and destroys exercise state."
    )


def test_transcript_uses_add_reducer() -> None:
    """``AgentState.transcript`` should have ``operator.add`` as its reducer."""
    reducer = _get_reducer(AgentState, "transcript")
    assert reducer is operator.add, (
        f"transcript reducer is {reducer!r}, expected operator.add"
    )


# ── build_initial_state tests ──────────────────────────────────────────────


def test_build_initial_state_emits_only_current_user_turn() -> None:
    """build_initial_state should emit only the current user message in
    history/transcript — not the full prior history.

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

    # history and transcript should contain ONLY the current user turn.
    assert len(state["history"]) == 1, (
        f"Expected 1 entry in history, got {len(state['history'])}. "
        f"build_initial_state should not reconstruct the full prior history."
    )
    assert state["history"][0]["content"] == "How are you?"
    assert state["history"][0]["role"] == MessageRole.USER.value

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
    assert state["progress"]["turn_count"] == 3


# ── finalize_turn_node tests ──────────────────────────────────────────────


class _FakeRuntime:
    def __init__(self) -> None:
        self.context = WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_response_style=MemoryMode.LOCAL,
        )


@pytest.mark.asyncio
async def test_finalize_returns_single_element_delta() -> None:
    """finalize_turn_node should return a 1-element list for transcript
    and history — the reducer handles appending to the accumulated state.
    """

    state: dict[str, Any] = {
        "transcript": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Help me"},
        ],
        "history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Help me"},
        ],
        "response": {"text": "Of course, what's on your mind?"},
        "routing": {"response_style": "supportive"},
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

    assert len(delta["history"]) == 1
    assert delta["history"] == delta["transcript"]


# ── Multi-turn accumulation via checkpointer ───────────────────────────────


@pytest.mark.asyncio
async def test_multi_turn_transcript_accumulates_via_checkpointer() -> None:
    """Running 3 turns on the same thread should produce a transcript
    with 6 entries (3 user + 3 assistant), accumulated by the reducer
    and checkpointer — not by manual reconstruction.
    """

    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        await runtime.run_turn(thread_id="t-accum", message="Turn 1")
        await runtime.run_turn(thread_id="t-accum", message="Turn 2")
        result = await runtime.run_turn(thread_id="t-accum", message="Turn 3")

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

    The progress field uses a merge reducer so build_initial_state's
    fresh progress dict (with turn_count, stage, etc.) merges with
    the checkpoint's progress (which carries exercise_type/exercise_step)
    rather than overwriting it.
    """

    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        # Turn 1: run a normal turn to create a checkpoint.
        await runtime.run_turn(thread_id="t-exercise", message="Hello")

        # Inject exercise state into the checkpoint by reading the
        # current state and updating progress directly. This simulates
        # what the guided_exercise_response_node does during a real
        # exercise turn.
        state1 = await runtime.get_state("t-exercise")
        assert state1 is not None
        state1["progress"]["exercise_type"] = "grounding_5_4_3_2_1"
        state1["progress"]["exercise_step"] = 1

        # Write the modified state back via the graph's update_state.
        graph = runtime._get_graph()
        config = runtime._config_for_thread("t-exercise")
        await graph.aupdate_state(
            config,
            {"progress": state1["progress"]},
            as_node="finalize_turn_node",
        )

        # Turn 2: run another turn — exercise state must survive.
        result2 = await runtime.run_turn(
            thread_id="t-exercise", message="Continue the exercise"
        )
        state2 = result2.state

        progress2 = state2.get("progress", {})
        assert progress2.get("exercise_type") == "grounding_5_4_3_2_1", (
            f"exercise_type lost across turns. progress={progress2}"
        )
        # exercise_step may advance during the turn (the guided exercise
        # node processes a step), so assert it's present rather than
        # checking an exact value.
        assert progress2.get("exercise_step") is not None, (
            f"exercise_step lost across turns. progress={progress2}"
        )
        assert progress2.get("turn_count", 0) >= 2, (
            f"turn_count should be >= 2, got {progress2.get('turn_count')}"
        )


@pytest.mark.asyncio
async def test_run_turn_does_not_deserialize_full_history_for_turn_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent turns should derive turn_count from checkpoint progress,
    not by calling ``get_history()`` and rebuilding the transcript.
    """

    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        await runtime.run_turn(thread_id="t-no-history", message="First")

        async def _fail_get_history(thread_id: str) -> list[Any]:
            raise AssertionError(f"get_history should not be called for {thread_id}")

        monkeypatch.setattr(runtime, "get_history", _fail_get_history)
        result = await runtime.run_turn(thread_id="t-no-history", message="Second")

    assert result.state["progress"]["turn_count"] == 2


@pytest.mark.asyncio
async def test_run_turn_stream_does_not_deserialize_full_history_for_turn_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming turns should use the same turn-count path as ``run_turn``."""

    async with PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
    ) as runtime:
        await runtime.run_turn(thread_id="t-no-history-stream", message="First")

        async def _fail_get_history(thread_id: str) -> list[Any]:
            raise AssertionError(f"get_history should not be called for {thread_id}")

        monkeypatch.setattr(runtime, "get_history", _fail_get_history)

        done_event = None
        async for event in runtime.run_turn_stream(
            thread_id="t-no-history-stream",
            message="Second",
        ):
            if hasattr(event, "output"):
                done_event = event
        state = await runtime.get_state("t-no-history-stream")
        assert state is not None
        assert state["progress"]["turn_count"] == 2

    assert done_event is not None
