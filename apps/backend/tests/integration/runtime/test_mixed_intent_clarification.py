"""Regression coverage for mixed-intent clarification dispatch."""

from __future__ import annotations

import pytest

import agent.runtime.openai_text_runtime as openai_runtime
from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from tests.support.openai_text import FakeOpenAISDKRunner, ScriptedOpenAITextRouteLLM
from tests.support.persistence import (
    in_memory_audit_feedback_dependencies,
    runtime_persistence_config,
    runtime_storage_paths,
)


def _runtime(**kwargs) -> PersistentAgentRuntime:
    return PersistentAgentRuntime(
        dependencies=in_memory_audit_feedback_dependencies(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_blocking_clarification_uses_therapeutic_clarifying_route(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit blocking clarification should be carried as routing metadata."""

    runner = FakeOpenAISDKRunner("Which would help more right now?")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with _runtime(
        storage_paths=runtime_storage_paths(tmp_path),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-mixed-intent-blocking",
            user_id="user-1",
            message="Can you look up grounding techniques, or maybe talk me through one?",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="grounded_lookup",
                triage_confidence="medium",
                clarification_needed=True,
                clarification_kind="blocking",
                secondary_route="guided_exercise",
                intent_summary="User is choosing between lookup and guided practice.",
                clarification_question="Would you prefer a lookup or a guided exercise?",
            ),
        )

    assert result.state["route"] == "therapeutic"
    assert result.output.response_style == "clarifying"
    assert (
        result.output.diagnostics["openai_triage_tentative_route"] == "grounded_lookup"
    )
    assert result.output.diagnostics["openai_triage_clarification_needed"] is True
    assert result.output.diagnostics["openai_triage_clarification_kind"] == "blocking"
    assert (
        result.output.diagnostics["openai_triage_secondary_route"] == "guided_exercise"
    )

    lifecycle = result.state["turn_lifecycle"]
    assert lifecycle["tentative_route"] == "grounded_lookup"
    assert lifecycle["triage_confidence"] == "medium"
    assert lifecycle["clarification_needed"] is True
    assert lifecycle["clarification_kind"] == "blocking"
    assert lifecycle["secondary_route"] == "guided_exercise"
    assert lifecycle["intent_summary"] == (
        "User is choosing between lookup and guided practice."
    )
    assert lifecycle["clarification_question"] == (
        "Would you prefer a lookup or a guided exercise?"
    )


@pytest.mark.asyncio
async def test_legacy_low_confidence_still_clarifies(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older low-confidence triage outputs keep the existing clarifying fallback."""

    runner = FakeOpenAISDKRunner("Could you say a little more about what you want?")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with _runtime(
        storage_paths=runtime_storage_paths(tmp_path),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-mixed-intent-legacy-low",
            user_id="user-1",
            message="Maybe check this, or maybe just help me with it.",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="grounded_lookup",
                triage_confidence="low",
            ),
        )

    assert result.state["route"] == "therapeutic"
    assert result.output.response_style == "clarifying"
    lifecycle = result.state["turn_lifecycle"]
    assert lifecycle["tentative_route"] == "grounded_lookup"
    assert lifecycle["triage_confidence"] == "low"
    assert lifecycle["clarification_needed"] is True
    assert lifecycle["clarification_kind"] == "none"


@pytest.mark.asyncio
async def test_soft_clarification_preserves_primary_route(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soft clarification is metadata, not a forced blocking clarification route."""

    runner = FakeOpenAISDKRunner("grounded answer")
    monkeypatch.setattr(openai_runtime, "_DEFAULT_OPENAI_RUNNER", runner)

    async with _runtime(
        storage_paths=runtime_storage_paths(tmp_path),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
    ) as runtime:
        result = await runtime.run_turn(
            thread_id="thread-mixed-intent-soft",
            user_id="user-1",
            message="Can you check whether this is common? I might just need reassurance.",
            llm_client=ScriptedOpenAITextRouteLLM(
                route="grounded_lookup",
                triage_confidence="medium",
                clarification_needed=True,
                clarification_kind="soft",
                secondary_route="therapeutic",
                intent_summary="User asked for lookup while also seeking reassurance.",
            ),
        )

    assert result.state["route"] == "grounded_lookup"
    assert result.state["turn_lifecycle"]["clarification_needed"] is True
    assert result.state["turn_lifecycle"]["clarification_kind"] == "soft"
    assert result.state["turn_lifecycle"]["secondary_route"] == "therapeutic"
    assert "tentative_route" not in result.state["turn_lifecycle"]
