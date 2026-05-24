from __future__ import annotations

from agent.voice.tools import build_voice_realtime_tools


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
