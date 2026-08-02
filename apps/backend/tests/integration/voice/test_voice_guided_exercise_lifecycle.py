"""Integration coverage for app-owned voice guided-exercise lifecycle effects."""

from __future__ import annotations

import asyncio

import pytest

import agent.voice.runtime_facade as voice_runtime_facade
from agent.memory.modes import MemoryMode
from agent.memory.operations.semantic_writes import fetch_existing_semantic_records
from agent.memory.retrieval.service import load_memory_for_turn
from agent.models import Channel
from agent.runtime import PersistentAgentRuntime, RuntimeBehaviorConfig
from agent.skills.guided_exercises.catalog.registry import (
    EXERCISE_BOX_BREATHING,
    get_exercise_definition,
)
from agent.voice.tools import execute_voice_tool_call
from tests.support.persistence import (
    in_memory_audit_feedback_dependencies,
    in_memory_runtime_storage_paths,
    runtime_persistence_config,
)


def _runtime() -> PersistentAgentRuntime:
    return PersistentAgentRuntime(
        dependencies=in_memory_audit_feedback_dependencies(),
        storage_paths=in_memory_runtime_storage_paths(),
        persistence_config=runtime_persistence_config(MemoryMode.LOCAL),
        behavior_config=RuntimeBehaviorConfig(
            finalize_active_sessions_on_close=False,
        ),
    )


async def _dispatch(
    runtime: PersistentAgentRuntime,
    *,
    thread_id: str,
    user_id: str,
    tool_name: str,
    arguments: dict[str, object],
    memory_mode: str,
) -> dict[str, object]:
    message = "Let's continue the exercise."
    return await execute_voice_tool_call(
        runtime=runtime,
        tool_name=tool_name,
        arguments=arguments,
        thread_id=thread_id,
        user_id=user_id,
        current_user_message=message,
        transcript=[{"role": "user", "content": message}],
        llm_client=None,
        memory_mode=memory_mode,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_mode", "expects_completion_fact"),
    [
        ("persistent", True),
        ("incognito", False),
    ],
)
async def test_voice_start_to_completion_persists_shared_lifecycle_effects(
    memory_mode: str,
    expects_completion_fact: bool,
) -> None:
    runtime = _runtime()
    thread_id = f"voice-exercise-{memory_mode}"
    user_id = f"voice-owner-{memory_mode}"
    definition = get_exercise_definition(EXERCISE_BOX_BREATHING)
    assert definition is not None

    async with runtime:
        start = await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="start_guided_exercise",
            arguments={
                "exercise_type": EXERCISE_BOX_BREATHING,
                "therapeutic_approach": "dbt_skills",
            },
            memory_mode=memory_mode,
        )

        assert start["status"] == "active"
        assert start["runtime_action"] == "start"
        assert start["current_step_id"] == definition.steps[0].id
        started_state = await runtime.get_state(thread_id)
        assert started_state is not None
        assert started_state["exercise_state"] == {
            "exercise_type": EXERCISE_BOX_BREATHING,
            "exercise_step": 0,
            "exercise_step_id": definition.steps[0].id,
            "exercise_version": definition.version,
            "exercise_therapeutic_approach": "dbt_skills",
        }

        duplicate_start = await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="start_guided_exercise",
            arguments={"exercise_type": EXERCISE_BOX_BREATHING},
            memory_mode=memory_mode,
        )
        assert duplicate_start["status"] == "conflict"
        state_after_conflict = await runtime.get_state(thread_id)
        assert state_after_conflict is not None
        assert state_after_conflict["exercise_state"] == started_state["exercise_state"]

        for index, step in enumerate(definition.steps):
            progress = await _dispatch(
                runtime,
                thread_id=thread_id,
                user_id=user_id,
                tool_name="record_guided_exercise_progress",
                arguments={
                    "expected_skill_id": EXERCISE_BOX_BREATHING,
                    "expected_step_id": step.id,
                    "outcome": "complete",
                    "user_response_summary": "The user completed this step.",
                },
                memory_mode=memory_mode,
            )
            if index == len(definition.steps) - 1:
                assert progress["status"] == "completed"
                assert progress["runtime_action"] == "complete"
            else:
                assert progress["status"] == "active"
                assert progress["runtime_action"] == "advance"

        completed_state = await runtime.get_state(thread_id)
        assert completed_state is not None
        assert completed_state["exercise_state"] == {
            "exercise_type": None,
            "exercise_step": None,
            "exercise_step_id": None,
            "exercise_version": None,
            "exercise_therapeutic_approach": None,
        }
        records = await fetch_existing_semantic_records(
            runtime.memory_store,
            owner_id=user_id,
        )

        if expects_completion_fact:
            assert len(records) == 1
            fact = records[0].value
            assert fact["category"] == "coping_strategy"
            assert fact["predicate"] == "USES"
            assert fact["object"]["identifier"] == EXERCISE_BOX_BREATHING
            assert fact["subject"]["identifier"] == user_id
            assert fact["source_session_id"] == thread_id

            text_memory = await load_memory_for_turn(
                memory_store=runtime.memory_store,
                embedding_provider=None,
                owner_id=user_id,
                query="Can we do the box breathing exercise again?",
                is_first_turn=False,
            )
            assert any(
                entry.get("object") == EXERCISE_BOX_BREATHING
                for entry in text_memory.working_memory
            )
        else:
            assert records == []


@pytest.mark.asyncio
async def test_concurrent_stale_voice_completion_persists_once() -> None:
    runtime = _runtime()
    thread_id = "voice-exercise-concurrent-completion"
    user_id = "voice-owner-concurrent-completion"
    definition = get_exercise_definition(EXERCISE_BOX_BREATHING)
    assert definition is not None

    async with runtime:
        await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="start_guided_exercise",
            arguments={"exercise_type": EXERCISE_BOX_BREATHING},
            memory_mode="persistent",
        )
        for step in definition.steps[:-1]:
            progress = await _dispatch(
                runtime,
                thread_id=thread_id,
                user_id=user_id,
                tool_name="record_guided_exercise_progress",
                arguments={
                    "expected_skill_id": EXERCISE_BOX_BREATHING,
                    "expected_step_id": step.id,
                    "outcome": "complete",
                    "user_response_summary": "The user completed this step.",
                },
                memory_mode="persistent",
            )
            assert progress["runtime_action"] == "advance"

        terminal_result = {
            "status": "completed",
            "runtime_action": "complete",
            "skill_id": EXERCISE_BOX_BREATHING,
            "previous_step_id": definition.steps[-1].id,
            "exercise_state_delta": {
                "exercise_state": {
                    "exercise_type": None,
                    "exercise_step": None,
                    "exercise_step_id": None,
                    "exercise_version": None,
                    "exercise_therapeutic_approach": None,
                }
            },
            "side_effect": "active_skill_state_update",
            "retry_safe": False,
        }

        async def persist_terminal_result() -> dict[str, object]:
            return await runtime.voice.persist_voice_guided_exercise_result(
                thread_id=thread_id,
                user_id=user_id,
                current_user_message="The user completed the exercise.",
                transcript=[
                    {
                        "role": "user",
                        "content": "The user completed the exercise.",
                    }
                ],
                result=terminal_result,
                memory_mode="persistent",
            )

        results = await asyncio.gather(
            persist_terminal_result(),
            persist_terminal_result(),
        )
        assert sorted(result["status"] for result in results) == [
            "completed",
            "conflict",
        ]

        state = await runtime.get_state(thread_id)
        assert state is not None
        assert state["exercise_state"]["exercise_type"] is None
        records = await fetch_existing_semantic_records(
            runtime.memory_store,
            owner_id=user_id,
        )

    assert len(records) == 1


@pytest.mark.asyncio
async def test_cancelled_voice_completion_keeps_active_state_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    thread_id = "voice-exercise-cancelled-completion"
    user_id = "voice-owner-cancelled-completion"
    definition = get_exercise_definition(EXERCISE_BOX_BREATHING)
    assert definition is not None

    async with runtime:
        await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="start_guided_exercise",
            arguments={"exercise_type": EXERCISE_BOX_BREATHING},
            memory_mode="persistent",
        )
        for step in definition.steps[:-1]:
            progress = await _dispatch(
                runtime,
                thread_id=thread_id,
                user_id=user_id,
                tool_name="record_guided_exercise_progress",
                arguments={
                    "expected_skill_id": EXERCISE_BOX_BREATHING,
                    "expected_step_id": step.id,
                    "outcome": "complete",
                    "user_response_summary": "The user completed this step.",
                },
                memory_mode="persistent",
            )
            assert progress["runtime_action"] == "advance"

        terminal_result = {
            "status": "completed",
            "runtime_action": "complete",
            "skill_id": EXERCISE_BOX_BREATHING,
            "previous_step_id": definition.steps[-1].id,
            "exercise_state_delta": {
                "exercise_state": {
                    "exercise_type": None,
                    "exercise_step": None,
                    "exercise_step_id": None,
                    "exercise_version": None,
                    "exercise_therapeutic_approach": None,
                }
            },
            "side_effect": "active_skill_state_update",
            "retry_safe": False,
        }
        write_started = asyncio.Event()
        write_cancelled = asyncio.Event()
        original_write = voice_runtime_facade.write_exercise_completion_fact

        async def stalled_completion_write(**_: object) -> None:
            write_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                write_cancelled.set()
                raise

        monkeypatch.setattr(
            voice_runtime_facade,
            "write_exercise_completion_fact",
            stalled_completion_write,
        )

        completion_task = asyncio.create_task(
            runtime.voice.persist_voice_guided_exercise_result(
                thread_id=thread_id,
                user_id=user_id,
                current_user_message="The user completed the exercise.",
                transcript=[
                    {
                        "role": "user",
                        "content": "The user completed the exercise.",
                    }
                ],
                result=terminal_result,
                memory_mode="persistent",
            )
        )
        await write_started.wait()
        completion_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await completion_task
        assert write_cancelled.is_set()

        state_after_cancellation = await runtime.get_state(thread_id)
        assert state_after_cancellation is not None
        assert state_after_cancellation["exercise_state"]["exercise_type"] == (
            EXERCISE_BOX_BREATHING
        )
        assert state_after_cancellation["exercise_state"]["exercise_step_id"] == (
            definition.steps[-1].id
        )

        monkeypatch.setattr(
            voice_runtime_facade,
            "write_exercise_completion_fact",
            original_write,
        )
        retried_completion = await runtime.voice.persist_voice_guided_exercise_result(
            thread_id=thread_id,
            user_id=user_id,
            current_user_message="The user completed the exercise.",
            transcript=[
                {
                    "role": "user",
                    "content": "The user completed the exercise.",
                }
            ],
            result=terminal_result,
            memory_mode="persistent",
        )
        assert retried_completion["status"] == "completed"

        state = await runtime.get_state(thread_id)
        assert state is not None
        assert state["exercise_state"]["exercise_type"] is None
        records = await fetch_existing_semantic_records(
            runtime.memory_store,
            owner_id=user_id,
        )

    assert len(records) == 1


@pytest.mark.asyncio
async def test_failed_voice_completion_keeps_active_state_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    thread_id = "voice-exercise-failed-completion"
    user_id = "voice-owner-failed-completion"
    definition = get_exercise_definition(EXERCISE_BOX_BREATHING)
    assert definition is not None
    final_step_index = len(definition.steps) - 1
    terminal_result = {
        "status": "completed",
        "runtime_action": "complete",
        "skill_id": EXERCISE_BOX_BREATHING,
        "previous_step_id": definition.steps[-1].id,
        "exercise_state_delta": {
            "exercise_state": {
                "exercise_type": None,
                "exercise_step": None,
                "exercise_step_id": None,
                "exercise_version": None,
                "exercise_therapeutic_approach": None,
            }
        },
        "side_effect": "active_skill_state_update",
        "retry_safe": False,
    }

    async with runtime:
        await runtime._state_store.save_state(  # noqa: SLF001
            thread_id,
            {
                "thread_id": thread_id,
                "channel": Channel.VOICE,
                "user_id": user_id,
                "session_id": thread_id,
                "transcript": [],
                "session_progress": {"turn_count": 1},
                "exercise_state": {
                    "exercise_type": EXERCISE_BOX_BREATHING,
                    "exercise_step": final_step_index,
                    "exercise_step_id": definition.steps[-1].id,
                    "exercise_version": definition.version,
                    "exercise_therapeutic_approach": "dbt_skills",
                },
            },
        )
        original_write = voice_runtime_facade.write_exercise_completion_fact

        async def failed_completion_write(**_: object) -> bool:
            return False

        monkeypatch.setattr(
            voice_runtime_facade,
            "write_exercise_completion_fact",
            failed_completion_write,
        )
        failed_completion = await runtime.voice.persist_voice_guided_exercise_result(
            thread_id=thread_id,
            user_id=user_id,
            current_user_message="The user completed the exercise.",
            transcript=[],
            result=terminal_result,
            memory_mode="persistent",
        )

        assert failed_completion["status"] == "active"
        assert failed_completion["runtime_action"] == "hold"
        assert failed_completion["side_effect"] == "none"
        state_after_failure = await runtime.get_state(thread_id)
        assert state_after_failure is not None
        assert state_after_failure["exercise_state"]["exercise_type"] == (
            EXERCISE_BOX_BREATHING
        )
        assert state_after_failure["exercise_state"]["exercise_step_id"] == (
            definition.steps[-1].id
        )

        monkeypatch.setattr(
            voice_runtime_facade,
            "write_exercise_completion_fact",
            original_write,
        )
        retried_completion = await runtime.voice.persist_voice_guided_exercise_result(
            thread_id=thread_id,
            user_id=user_id,
            current_user_message="The user completed the exercise.",
            transcript=[],
            result=terminal_result,
            memory_mode="persistent",
        )

        assert retried_completion["status"] == "completed"
        state = await runtime.get_state(thread_id)
        assert state is not None
        assert state["exercise_state"]["exercise_type"] is None
        records = await fetch_existing_semantic_records(
            runtime.memory_store,
            owner_id=user_id,
        )

    assert len(records) == 1


@pytest.mark.asyncio
async def test_voice_exit_and_unsafe_do_not_write_completion_facts() -> None:
    runtime = _runtime()
    thread_id = "voice-exercise-non-completion"
    user_id = "voice-owner-non-completion"

    async with runtime:
        await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="start_guided_exercise",
            arguments={"exercise_type": EXERCISE_BOX_BREATHING},
            memory_mode="persistent",
        )
        exited = await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="record_guided_exercise_progress",
            arguments={
                "expected_skill_id": EXERCISE_BOX_BREATHING,
                "expected_step_id": "inhale",
                "outcome": "exit",
                "user_response_summary": "The user wants to stop.",
            },
            memory_mode="persistent",
        )
        assert exited["runtime_action"] == "cancel"

        await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="start_guided_exercise",
            arguments={"exercise_type": EXERCISE_BOX_BREATHING},
            memory_mode="persistent",
        )
        unsafe = await _dispatch(
            runtime,
            thread_id=thread_id,
            user_id=user_id,
            tool_name="record_guided_exercise_progress",
            arguments={
                "expected_skill_id": EXERCISE_BOX_BREATHING,
                "expected_step_id": "inhale",
                "outcome": "unsafe",
                "user_response_summary": "The user may not be safe.",
            },
            memory_mode="persistent",
        )
        assert unsafe["runtime_action"] == "crisis"

        state = await runtime.get_state(thread_id)
        assert state is not None
        assert state["exercise_state"]["exercise_type"] == EXERCISE_BOX_BREATHING
        records = await fetch_existing_semantic_records(
            runtime.memory_store,
            owner_id=user_id,
        )

    assert records == []
