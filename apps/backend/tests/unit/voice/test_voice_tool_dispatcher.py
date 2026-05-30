from __future__ import annotations

import pytest

from agent.memory.modes import MemoryMode
from agent.runtime import PersistentAgentRuntime
from agent.runtime.context import CrisisResourceToolCallRecord
from agent.voice.tools import _execute_crisis_support_template, execute_voice_tool_call
from tests.support.persistence import FakeCrossRestartLLM


class _RuntimeThatMustNotBuildContext:
    memory_mode = MemoryMode.LOCAL

    async def build_voice_tool_context(self, **kwargs: object) -> object:
        raise AssertionError("incognito memory status must not read persistent runtime")


_MUTATOR_CASES = (
    (
        "save_response_preference",
        {"preference_text": "Please keep replies concise."},
    ),
    ("set_proactive_memory_recall", {"enabled": False}),
    (
        "prepare_memory_deletion_by_index",
        {"target_kind": "fact", "target_index": 1},
    ),
    (
        "prepare_memory_deletion_by_query",
        {"query": "old job anxiety"},
    ),
)


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


@pytest.mark.parametrize(("tool_name", "arguments"), _MUTATOR_CASES)
@pytest.mark.asyncio
async def test_voice_mutator_refuses_without_verified_user_quote(
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    output = await execute_voice_tool_call(
        runtime=_RuntimeThatMustNotBuildContext(),
        tool_name=tool_name,
        arguments=arguments,
        thread_id="voice-thread",
        user_id="user-1",
        current_user_message="Please keep replies concise.",
        transcript=[],
        llm_client=None,
        memory_mode="persistent",
    )

    assert output["refused"] is True
    assert output["reason"] == "user_intent_not_verified"
    assert output["side_effect"] == "none"
    assert output["retry_safe"] is True


@pytest.mark.asyncio
async def test_voice_mutator_refuses_without_owner_or_session_id() -> None:
    output = await execute_voice_tool_call(
        runtime=_RuntimeThatMustNotBuildContext(),
        tool_name="set_proactive_memory_recall",
        arguments={
            "enabled": False,
            "user_quote": "Please turn off proactive memory recall.",
        },
        thread_id="",
        user_id=None,
        current_user_message="Please turn off proactive memory recall.",
        transcript=[],
        llm_client=None,
        memory_mode="persistent",
    )

    assert output["refused"] is True
    assert output["reason"] == "owner_or_session_missing"
    assert output["side_effect"] == "none"
    assert output["retry_safe"] is True


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
            arguments={
                "enabled": True,
                "user_quote": "Remember proactively when useful.",
            },
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="Remember proactively when useful.",
            transcript=[],
            llm_client=None,
        )

    assert output["side_effect"] == "procedural_profile_update"
    assert output["retry_safe"] is True


@pytest.mark.asyncio
async def test_voice_mutator_verifies_quote_from_recent_user_transcript() -> None:
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
            arguments={
                "enabled": False,
                "user_quote": "please turn off proactive recall",
            },
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="",
            transcript=[
                {"role": "user", "content": "Small talk first."},
                {"role": "assistant", "content": "I'm listening."},
                {
                    "role": "user",
                    "content": "Please   turn OFF proactive recall for now.",
                },
            ],
            llm_client=None,
        )

    assert output["side_effect"] == "procedural_profile_update"
    assert output["retry_safe"] is True


@pytest.mark.asyncio
async def test_voice_mutator_verifies_quote_with_punctuation_difference() -> None:
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
            arguments={
                "enabled": False,
                "user_quote": "turn it off now",
            },
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="Could you turn it off, now?",
            transcript=[],
            llm_client=None,
        )

    assert output["side_effect"] == "procedural_profile_update"
    assert output["retry_safe"] is True


@pytest.mark.asyncio
async def test_voice_mutator_refuses_too_short_user_quote() -> None:
    output = await execute_voice_tool_call(
        runtime=_RuntimeThatMustNotBuildContext(),
        tool_name="set_proactive_memory_recall",
        arguments={"enabled": True, "user_quote": "yes"},
        thread_id="voice-thread",
        user_id="user-1",
        current_user_message="Yes, I want help with that.",
        transcript=[],
        llm_client=None,
        memory_mode="persistent",
    )

    assert output["refused"] is True
    assert output["reason"] == "user_intent_not_verified"
    assert output["side_effect"] == "none"
    assert output["retry_safe"] is True


@pytest.mark.asyncio
async def test_voice_mutator_verifies_eligible_user_quote() -> None:
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
            arguments={"enabled": False, "user_quote": "turn it off"},
            thread_id="voice-thread",
            user_id="user-1",
            current_user_message="Please turn it off for memory recall.",
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
async def test_voice_tool_dispatcher_loads_crisis_support_template() -> None:
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
            tool_name="get_crisis_support_template",
            arguments={"risk_level": "imminent"},
            thread_id="voice-thread",
            user_id=None,
            current_user_message="I might hurt myself tonight.",
            transcript=[{"role": "user", "content": "I might hurt myself tonight."}],
            llm_client=None,
        )

    assert output["risk_level"] == "imminent"
    assert output["side_effect"] == "none"
    # No prior lookup ran, so the scaffold must steer toward emergency help
    # without inventing any phone number.
    assert "emergency services" in output["response_text"]


class _ContextWithPriorLookup:
    """Minimal context exposing one recorded crisis-resource lookup."""

    def __init__(self, record: CrisisResourceToolCallRecord) -> None:
        self._record = record

    def latest_crisis_resource_tool_result(self) -> CrisisResourceToolCallRecord:
        return self._record


@pytest.mark.asyncio
async def test_crisis_support_template_reuses_prior_lookup_resources() -> None:
    """The scaffold threads verified resources from a prior lookup through.

    Each voice tool call builds a fresh context, but within one context a
    prior ``lookup_crisis_resources`` result should flow into the scaffold so
    the model is handed verified numbers it must not restate or invent.
    """

    context = _ContextWithPriorLookup(
        CrisisResourceToolCallRecord(
            tool_name="lookup_crisis_resources",
            response_text="Verified resource.",
            inferred_location="Singapore",
            found_resources=[
                {
                    "name": "Samaritans of Singapore",
                    "phone": "1767",
                    "url": "https://www.sos.org.sg",
                    "region": "Singapore",
                }
            ],
            resource_lookup_status="found",
        )
    )

    output = await _execute_crisis_support_template(context, {"risk_level": "imminent"})

    assert "Samaritans of Singapore: 1767" in output["response_text"]
    assert "Do not modify phone numbers" in output["response_text"]


@pytest.mark.asyncio
async def test_crisis_support_template_reuses_lookup_across_separate_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior lookup must reach the template across two Realtime requests.

    OpenAI Realtime dispatches each tool call as its own ``/realtime/tools``
    request, so ``lookup_crisis_resources`` and ``get_crisis_support_template``
    build separate contexts. This is the exact flow the per-context recording
    misses: without persisting the lookup to thread state, the template degrades
    to ``not_attempted`` and can tell the model no verified resource exists right
    after one was found. Both calls run against one real runtime so the bridge
    is the persisted state, not an in-memory context shared by the test.
    """

    runtime = PersistentAgentRuntime(
        sqlite_path=":memory:",
        memory_sqlite_path=":memory:",
        crisis_log_sqlite_path=":memory:",
        feedback_sqlite_path=":memory:",
        memory_mode=MemoryMode.INCOGNITO,
    )

    async def fake_find_crisis_resources_for_request(request, *, llm_client):
        return (
            "Singapore",
            [
                {
                    "name": "Samaritans of Singapore",
                    "phone": "1767",
                    "url": "https://www.sos.org.sg",
                    "region": "Singapore",
                }
            ],
            "found",
        )

    monkeypatch.setattr(
        "agent.tools.crisis.find_crisis_resources_for_request",
        fake_find_crisis_resources_for_request,
    )

    llm_client = FakeCrossRestartLLM()
    async with runtime:
        lookup_output = await execute_voice_tool_call(
            runtime=runtime,
            tool_name="lookup_crisis_resources",
            arguments={},
            thread_id="voice-thread",
            user_id=None,
            current_user_message="I might hurt myself tonight.",
            transcript=[{"role": "user", "content": "I might hurt myself tonight."}],
            llm_client=llm_client,
        )
        # Second, separate request: a brand-new context is built internally.
        template_output = await execute_voice_tool_call(
            runtime=runtime,
            tool_name="get_crisis_support_template",
            arguments={"risk_level": "imminent"},
            thread_id="voice-thread",
            user_id=None,
            current_user_message="I might hurt myself tonight.",
            transcript=[{"role": "user", "content": "I might hurt myself tonight."}],
            llm_client=llm_client,
        )

    assert lookup_output["resource_lookup_status"] == "found"
    # The verified resource threads through to the second call's scaffold.
    assert "Samaritans of Singapore: 1767" in template_output["response_text"]
    assert "Do not modify phone numbers" in template_output["response_text"]


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
