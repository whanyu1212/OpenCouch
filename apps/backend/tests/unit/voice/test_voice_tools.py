from __future__ import annotations

import pytest

from agent.voice.tools import build_voice_realtime_tools, execute_voice_tool_call


_MUTATOR_TOOL_NAMES = {
    "save_response_preference",
    "set_proactive_memory_recall",
    "prepare_memory_deletion_by_index",
    "prepare_memory_deletion_by_query",
}


def _tool_by_name(name: str) -> dict[str, object]:
    tools = build_voice_realtime_tools(memory_mode="persistent")
    return next(tool for tool in tools if tool["name"] == name)


def test_voice_tool_surface_is_narrow_for_incognito() -> None:
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="incognito")
    }

    assert "wait_for_user" in names
    assert "show_memory_status" in names
    assert "show_saved_memory" not in names
    assert "save_response_preference" not in names
    assert "lookup_crisis_resources" in names


def test_voice_tool_surface_includes_persistent_memory_tools() -> None:
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="persistent")
    }

    assert "wait_for_user" in names
    assert "show_saved_memory" in names
    assert "show_memory_status" in names
    assert "save_response_preference" in names


def test_voice_tool_surface_includes_grounded_lookup() -> None:
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="persistent")
    }

    assert "answer_grounded_lookup" in names


def test_voice_persistent_tool_surface_includes_text_memory_controls() -> None:
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="persistent")
    }

    assert "set_proactive_memory_recall" in names
    assert "prepare_memory_deletion_by_index" in names
    assert "prepare_memory_deletion_by_query" in names
    assert "confirm_memory_deletion" in names
    assert "cancel_memory_deletion" in names


def test_voice_tool_surface_includes_therapeutic_response_skill_loader() -> None:
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="persistent")
    }

    assert "load_therapeutic_response_skill" in names


def test_voice_tool_surface_includes_guided_exercise_progress_tool() -> None:
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="persistent")
    }

    assert "record_guided_exercise_progress" in names
    assert "load_guided_exercise_skill" in names


def test_guided_exercise_progress_tool_requires_voice_progress_schema() -> None:
    tool = _tool_by_name("record_guided_exercise_progress")
    parameters = tool["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    required = parameters["required"]
    assert isinstance(required, list)

    assert required == [
        "expected_skill_id",
        "expected_step_id",
        "outcome",
        "user_response_summary",
    ]
    assert properties["outcome"]["enum"] == [
        "complete",
        "partial",
        "hold",
        "stuck",
        "exit",
        "unsafe",
    ]


@pytest.mark.parametrize("memory_mode", ["incognito", "persistent"])
def test_voice_tool_surface_includes_crisis_support_template(memory_mode: str) -> None:
    # The crisis scaffold is safety-critical and must not depend on memory mode.
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode=memory_mode)
    }

    assert "get_crisis_support_template" in names


def test_crisis_support_template_tool_requires_risk_level_enum() -> None:
    tool = _tool_by_name("get_crisis_support_template")
    parameters = tool["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    required = parameters["required"]
    assert isinstance(required, list)

    assert required == ["risk_level"]
    assert properties["risk_level"]["enum"] == ["moderate", "high", "imminent"]


def test_voice_tools_are_realtime_function_schemas() -> None:
    tools = build_voice_realtime_tools(memory_mode="persistent")

    assert all(tool["type"] == "function" for tool in tools)
    assert all(tool["parameters"]["type"] == "object" for tool in tools)
    assert all(tool["parameters"]["additionalProperties"] is False for tool in tools)


def test_voice_mutator_tools_require_user_quote_evidence() -> None:
    for tool_name in _MUTATOR_TOOL_NAMES:
        tool = _tool_by_name(tool_name)
        parameters = tool["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        required = parameters["required"]
        assert isinstance(required, list)

        assert "user_quote" in properties
        assert "user_quote" in required


def test_voice_confirmation_tools_do_not_require_user_quote() -> None:
    for tool_name in {"confirm_memory_deletion", "cancel_memory_deletion"}:
        tool = _tool_by_name(tool_name)
        parameters = tool["parameters"]
        assert isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        required = parameters["required"]
        assert isinstance(required, list)

        assert "user_quote" not in properties
        assert "user_quote" not in required


@pytest.mark.asyncio
async def test_wait_for_user_tool_is_noop_signal() -> None:
    result = await execute_voice_tool_call(
        runtime=object(),
        tool_name="wait_for_user",
        arguments={},
        thread_id="voice-thread",
        user_id=None,
        current_user_message="",
        transcript=[],
        memory_mode="incognito",
        llm_client=None,
    )

    assert result == {
        "response_text": "",
        "should_respond": False,
        "side_effect": "none",
    }
