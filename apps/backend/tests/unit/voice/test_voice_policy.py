"""Unit tests for build_voice_instructions policy text."""

from __future__ import annotations

import pytest

from agent.voice.policy import build_voice_instructions


def test_persistent_instructions_include_recall_tool_guidance() -> None:
    instructions = build_voice_instructions(
        thread_id="voice-thread",
        user_id="alice",
        memory_mode="persistent",
    )

    assert "recall_saved_memory" in instructions
    # The guidance should warn against calling it every turn.
    assert "every turn" in instructions


def test_incognito_instructions_omit_recall_tool_guidance() -> None:
    instructions = build_voice_instructions(
        thread_id="voice-thread",
        user_id=None,
        memory_mode="incognito",
    )

    assert "recall_saved_memory" not in instructions
    # Incognito should still surface the durable-memory restriction.
    assert "incognito" in instructions.lower()


@pytest.mark.parametrize("memory_mode", ["incognito", "persistent"])
def test_instructions_contain_session_metadata(memory_mode: str) -> None:
    instructions = build_voice_instructions(
        thread_id="voice-thread-42",
        user_id="alice",
        memory_mode=memory_mode,
    )

    assert "thread_id=voice-thread-42" in instructions
    assert f"memory_mode={memory_mode}" in instructions
