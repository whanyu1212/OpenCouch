from __future__ import annotations

import pytest

import agent.voice.tools as voice_tools
from agent.voice import runtime_facade
from agent.voice.tools.schemas import build_voice_realtime_tools
from agent.voice.tools.specs import (
    VOICE_TOOL_SPECS,
    VOICE_TOOL_SPECS_BY_NAME,
    VoiceToolSpec,
)


def test_voice_tool_specs_preserve_realtime_order() -> None:
    assert tuple(spec.name for spec in VOICE_TOOL_SPECS) == (
        "wait_for_user",
        "show_saved_memory",
        "recall_saved_memory",
        "save_response_preference",
        "set_proactive_memory_recall",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
        "confirm_memory_deletion",
        "cancel_memory_deletion",
        "show_memory_status",
        "load_therapeutic_response_skill",
        "answer_grounded_lookup",
        "lookup_crisis_resources",
        "get_crisis_support_template",
        "list_guided_exercise_skills",
        "load_guided_exercise_skill",
        "record_guided_exercise_progress",
    )


def test_voice_tool_specs_drive_supported_surface_and_policies() -> None:
    assert set(VOICE_TOOL_SPECS_BY_NAME) == {spec.name for spec in VOICE_TOOL_SPECS}
    assert len(VOICE_TOOL_SPECS_BY_NAME) == len(VOICE_TOOL_SPECS)

    persistent_names = {spec.name for spec in VOICE_TOOL_SPECS if spec.persistent_only}
    assert persistent_names == {
        "show_saved_memory",
        "recall_saved_memory",
        "save_response_preference",
        "set_proactive_memory_recall",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
        "confirm_memory_deletion",
        "cancel_memory_deletion",
    }

    memory_mutator_names = {
        spec.name for spec in VOICE_TOOL_SPECS if spec.memory_mutator
    }
    assert memory_mutator_names == {
        "set_proactive_memory_recall",
        "save_response_preference",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
        "confirm_memory_deletion",
        "cancel_memory_deletion",
    }

    intent_gated_names = {
        spec.name for spec in VOICE_TOOL_SPECS if spec.intent_gated_mutator
    }
    assert intent_gated_names == {
        "set_proactive_memory_recall",
        "save_response_preference",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
    }


def test_voice_tool_specs_build_isolated_realtime_schemas() -> None:
    schema = VOICE_TOOL_SPECS_BY_NAME[
        "answer_grounded_lookup"
    ].as_realtime_function_tool()
    schema["parameters"]["properties"]["query"]["description"] = "changed"

    assert (
        VOICE_TOOL_SPECS_BY_NAME["answer_grounded_lookup"].properties["query"][
            "description"
        ]
        != "changed"
    )
    assert [
        tool["name"] for tool in build_voice_realtime_tools(memory_mode="incognito")
    ] == [spec.name for spec in VOICE_TOOL_SPECS if not spec.persistent_only]


def test_voice_public_exports_exclude_private_implementations() -> None:
    assert set(voice_tools.__all__) == {
        "VOICE_TOOL_SPECS",
        "VoiceToolSpec",
        "build_voice_realtime_tools",
        "execute_voice_tool_call",
    }
    assert all(not name.startswith("_") for name in voice_tools.__all__)
    assert runtime_facade.__all__ == ["VoiceRuntimeFacade"]


async def _noop_handler(context: object, arguments: dict[str, object]) -> object:
    del context, arguments
    return {}


def test_voice_tool_spec_rejects_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="requires undefined properties"):
        VoiceToolSpec(
            name="invalid",
            description="",
            properties={},
            required=("missing",),
            handler=_noop_handler,
        )

    with pytest.raises(ValueError, match="route_priority"):
        VoiceToolSpec(
            name="invalid-route",
            description="",
            properties={},
            required=(),
            handler=_noop_handler,
            route="crisis",
            response_style="crisis_response",
        )
