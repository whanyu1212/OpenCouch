from __future__ import annotations

from agent.voice.tools import build_voice_realtime_tools


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

    assert "show_memory_status" in names
    assert "show_saved_memory" not in names
    assert "save_response_preference" not in names
    assert "lookup_crisis_resources" in names


def test_voice_tool_surface_includes_persistent_memory_tools() -> None:
    names = {
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="persistent")
    }

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
