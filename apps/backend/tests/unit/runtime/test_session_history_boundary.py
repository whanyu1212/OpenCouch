"""Tests for the transcript and SDK-session history boundary."""

from __future__ import annotations

from agent.models import MessageRole
from agent.runtime.session.history import (
    include_prompt_history,
    messages_from_sdk_session_items,
    messages_from_transcript,
    session_conversation_from_transcript,
    state_without_prompt_history,
    strip_recent_history_from_prompt,
)


def test_messages_from_transcript_returns_public_user_assistant_turns_only() -> None:
    messages = messages_from_transcript(
        [
            {"role": "system", "content": "hidden system note"},
            {"role": "user", "content": "  hi  "},
            {
                "role": "assistant",
                "content": "  hello  ",
                "response_style": "supportive",
            },
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": "   "},
        ]
    )

    assert [(message.role, message.content) for message in messages] == [
        (MessageRole.USER, "hi"),
        (MessageRole.ASSISTANT, "hello"),
    ]
    assert messages[0].response_style is None
    assert messages[1].response_style == "supportive"


def test_messages_from_sdk_session_items_returns_public_chat_history_only() -> None:
    messages = messages_from_sdk_session_items(
        [
            {"role": "system", "content": "hidden system note"},
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "name": "show_memory_status"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {"role": "assistant", "content": ""},
        ]
    )

    assert [(message.role, message.content) for message in messages] == [
        (MessageRole.USER, "hi"),
        (MessageRole.ASSISTANT, "hello"),
    ]


def test_strip_recent_history_preserves_memory_context_for_sdk_sessions() -> None:
    prompt = (
        "System block.\n\n"
        "Recent conversation:\n"
        "user: old message\n"
        "assistant: old reply\n"
        "\nRelevant context from past sessions:\n"
        "- presentations make the user anxious\n"
        "\nCurrent user message:\n"
        "I feel anxious again"
    )

    stripped = strip_recent_history_from_prompt(prompt)

    assert "user: old message" not in stripped
    assert "assistant: old reply" not in stripped
    assert "conversation history is provided by the SDK session" in stripped
    assert "Relevant context from past sessions" in stripped
    assert "presentations make the user anxious" in stripped
    assert "Current user message" in stripped


def test_state_without_prompt_history_clears_copy_only() -> None:
    state = {
        "message": "current",
        "transcript": [{"role": "user", "content": "old"}],
        "history": [{"role": "assistant", "content": "older"}],
        "working_memory": [{"evidence_quote": "presentations make me anxious"}],
    }

    prompt_state = state_without_prompt_history(state)  # type: ignore[arg-type]

    assert prompt_state["transcript"] == []
    assert prompt_state["history"] == []
    assert prompt_state["working_memory"] == state["working_memory"]
    assert state["transcript"] == [{"role": "user", "content": "old"}]
    assert state["history"] == [{"role": "assistant", "content": "older"}]


def test_include_prompt_history_only_when_sdk_session_is_absent() -> None:
    assert include_prompt_history(None) is True
    assert include_prompt_history(object()) is False


def test_session_conversation_projects_public_turns_once() -> None:
    conversation = session_conversation_from_transcript(
        [
            {"role": "system", "content": "hidden system prompt"},
            {"role": "user", "content": "  I argued with Maya.  "},
            {
                "role": "assistant",
                "content": "  That sounds painful.  ",
                "response_style": "supportive",
            },
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": ""},
        ]
    )

    assert conversation.transcript_entries() == [
        {"role": "user", "content": "I argued with Maya."},
        {
            "role": "assistant",
            "content": "That sounds painful.",
            "response_style": "supportive",
        },
    ]
    assert conversation.user_texts() == ["I argued with Maya."]
    assert conversation.user_turn_count == 1
