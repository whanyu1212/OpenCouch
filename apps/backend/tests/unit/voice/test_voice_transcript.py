from __future__ import annotations

from agent.voice.transcript import voice_turn_to_transcript_entries


def test_voice_turn_to_transcript_entries() -> None:
    entries = voice_turn_to_transcript_entries(
        user_text="I feel overwhelmed.",
        assistant_text="That sounds like a lot to carry.",
        response_style="supportive",
    )

    assert entries == [
        {"role": "user", "content": "I feel overwhelmed."},
        {
            "role": "assistant",
            "content": "That sounds like a lot to carry.",
            "response_style": "supportive",
        },
    ]


def test_voice_turn_skips_blank_parts() -> None:
    entries = voice_turn_to_transcript_entries(
        user_text="",
        assistant_text="I am here with you.",
        response_style=None,
    )

    assert entries == [
        {
            "role": "assistant",
            "content": "I am here with you.",
            "response_style": "voice",
        }
    ]
