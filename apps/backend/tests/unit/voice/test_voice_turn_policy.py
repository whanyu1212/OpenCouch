from __future__ import annotations

from agent.voice.turn_policy import build_voice_turn_policy


def test_voice_turn_policy_marks_explicit_lookup() -> None:
    policy = build_voice_turn_policy(
        user_text="Can you look up the current 988 guidance?",
        memory_mode="persistent",
        has_active_guided_exercise=False,
        pending_memory_action=False,
    )

    assert policy.route == "grounded_lookup"
    assert policy.response_style == "grounded_lookup"
    assert policy.required_tool_name == "answer_grounded_lookup"
    assert policy.required_tool_arguments == {
        "query": "Can you look up the current 988 guidance?"
    }


def test_voice_turn_policy_requires_crisis_resource_tool_for_crisis_resource_request() -> (
    None
):
    policy = build_voice_turn_policy(
        user_text="I might hurt myself, can you find a crisis hotline in Singapore?",
        memory_mode="persistent",
        has_active_guided_exercise=False,
        pending_memory_action=False,
    )

    assert policy.route == "crisis"
    assert policy.response_style == "crisis_response"
    assert policy.required_tool_name == "lookup_crisis_resources"


def test_voice_turn_policy_keeps_support_turn_therapeutic() -> None:
    policy = build_voice_turn_policy(
        user_text="I feel overwhelmed tonight.",
        memory_mode="persistent",
        has_active_guided_exercise=False,
        pending_memory_action=False,
    )

    assert policy.route == "therapeutic"
    assert policy.response_style == "voice"
    assert policy.required_tool_name is None
