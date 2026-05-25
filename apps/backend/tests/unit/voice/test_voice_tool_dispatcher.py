from __future__ import annotations

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from agent.voice.tools import execute_voice_tool_call


class _RuntimeThatMustNotBuildContext:
    memory_mode = MemoryMode.LOCAL

    async def build_voice_tool_context(self, **kwargs: object) -> object:
        raise AssertionError("incognito memory status must not read persistent runtime")


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_executes_memory_status() -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
    )
    async with runtime:
        output = await execute_voice_tool_call(
            runtime=runtime,
            tool_name="show_memory_status",
            arguments={},
            thread_id="voice-thread",
            user_id=None,
            current_user_message="Is memory on?",
            transcript=[{"role": "user", "content": "Is memory on?"}],
            llm_client=None,
        )

    assert output["side_effect"] == "none"
    assert output["retry_safe"] is True
    assert "response_text" in output


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_answers_incognito_memory_status_without_store() -> (
    None
):
    output = await execute_voice_tool_call(
        runtime=_RuntimeThatMustNotBuildContext(),
        tool_name="show_memory_status",
        arguments={},
        thread_id="voice-thread",
        user_id="user-1",
        current_user_message="Is memory on?",
        transcript=[],
        llm_client=None,
        memory_mode="incognito",
    )

    assert output["side_effect"] == "none"
    assert output["retry_safe"] is True
    assert output["memory_mode"] == "incognito"
    assert "off for this incognito voice session" in str(output["response_text"])


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_rejects_persistent_memory_tool_in_incognito() -> (
    None
):
    with pytest.raises(ValueError, match="not available in incognito"):
        await execute_voice_tool_call(
            runtime=_RuntimeThatMustNotBuildContext(),
            tool_name="show_saved_memory",
            arguments={},
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="What do you remember?",
            transcript=[],
            llm_client=None,
            memory_mode="incognito",
        )


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_uses_transcript_when_user_message_is_blank() -> (
    None
):
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
    )
    async with runtime:
        output = await execute_voice_tool_call(
            runtime=runtime,
            tool_name="show_memory_status",
            arguments={},
            thread_id="voice-thread",
            user_id=None,
            current_user_message="",
            transcript=[{"role": "user", "content": "Is memory on?"}],
            llm_client=None,
        )

    assert output["side_effect"] == "none"
    assert "response_text" in output


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_executes_grounded_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
    )

    async def fake_execute_grounded_lookup_tool(context, *, query: str):
        return {
            "response_text": f"Verified answer for {query}",
            "grounded_lookup": {"query": query, "status": "answered"},
            "status": "answered",
            "side_effect": "none",
            "retry_safe": True,
        }

    monkeypatch.setattr(
        "agent.voice.tools.execute_grounded_lookup_tool",
        fake_execute_grounded_lookup_tool,
    )

    async with runtime:
        output = await execute_voice_tool_call(
            runtime=runtime,
            tool_name="answer_grounded_lookup",
            arguments={"query": "current public source query"},
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="Can you look this up?",
            transcript=[{"role": "user", "content": "Can you look this up?"}],
            llm_client=None,
        )

    assert output["response_text"] == "Verified answer for current public source query"
    assert output["grounded_lookup"] == {
        "query": "current public source query",
        "status": "answered",
    }


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_sets_proactive_memory_recall() -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.LOCAL,
    )
    async with runtime:
        output = await execute_voice_tool_call(
            runtime=runtime,
            tool_name="set_proactive_memory_recall",
            arguments={"enabled": True},
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="Remember proactively when useful.",
            transcript=[],
            llm_client=None,
        )

    assert output["side_effect"] == "procedural_profile_update"
    assert output["retry_safe"] is True


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_loads_therapeutic_response_skill() -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
    )
    async with runtime:
        output = await execute_voice_tool_call(
            runtime=runtime,
            tool_name="load_therapeutic_response_skill",
            arguments={
                "response_style": "supportive",
                "therapeutic_approach": "none",
            },
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="I had a rough day.",
            transcript=[],
            llm_client=None,
        )

    assert output["response_style"] == "supportive"
    assert output["side_effect"] == "none"
    assert "skill_context" in output


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_rejects_unknown_tool() -> None:
    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
    )
    async with runtime:
        with pytest.raises(ValueError, match="Unsupported voice tool"):
            await execute_voice_tool_call(
                runtime=runtime,
                tool_name="delete_everything",
                arguments={},
                thread_id="voice-thread",
                user_id=None,
                current_user_message="",
                transcript=[],
                llm_client=None,
            )


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_rejects_recall_in_incognito() -> None:
    """recall_saved_memory is persistent-only; incognito refuses pre-context."""

    with pytest.raises(ValueError, match="not available in incognito"):
        await execute_voice_tool_call(
            runtime=_RuntimeThatMustNotBuildContext(),
            tool_name="recall_saved_memory",
            arguments={"query": "work stress"},
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="",
            transcript=[],
            llm_client=None,
            memory_mode="incognito",
        )


@pytest.mark.asyncio
async def test_voice_tool_dispatcher_runtime_incognito_overrides_persistent_body() -> (
    None
):
    """Runtime mode is the floor: a 'persistent' body cannot escalate.

    Regression for the memory_mode override hole. Even when the client
    body claims ``memory_mode='persistent'``, an incognito runtime must
    cause persistent-only tools to refuse before any context is built.
    """

    class _IncognitoRuntimeThatMustNotBuildContext:
        memory_mode = MemoryMode.INCOGNITO

        async def build_voice_tool_context(self, **kwargs: object) -> object:
            raise AssertionError(
                "incognito runtime must refuse persistent-only tools "
                "regardless of request body"
            )

    with pytest.raises(ValueError, match="not available in incognito"):
        await execute_voice_tool_call(
            runtime=_IncognitoRuntimeThatMustNotBuildContext(),
            tool_name="recall_saved_memory",
            arguments={"query": "anything"},
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="",
            transcript=[],
            llm_client=None,
            # The client claims persistent; the runtime is incognito.
            # Incognito must win.
            memory_mode="persistent",
        )
