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


def test_voice_turn_policy_reflects_pending_memory_action() -> None:
    policy = build_voice_turn_policy(
        user_text="Yes, delete that one.",
        memory_mode="persistent",
        has_active_guided_exercise=False,
        pending_memory_action=True,
    )

    assert policy.route == "memory_control"
    assert policy.response_style == "memory_control"
    assert policy.required_tool_name is None
    assert "pending memory deletion" in policy.instructions
    assert "confirm_memory_deletion" in policy.instructions
    assert "cancel_memory_deletion" in policy.instructions


def test_voice_turn_policy_continues_active_exercise_without_starting_new_one() -> None:
    policy = build_voice_turn_policy(
        user_text="I noticed my shoulders relaxing.",
        memory_mode="persistent",
        has_active_guided_exercise=True,
        pending_memory_action=False,
    )

    assert policy.route == "therapeutic"
    assert "Continue the current guided exercise" in policy.instructions
    assert "Do not start a new guided exercise" in policy.instructions


def test_voice_turn_policy_skips_memory_control_route_in_incognito() -> None:
    """An incognito turn must not be routed into memory_control even when
    state carries a pending deletion from a prior persistent session.

    confirm_memory_deletion / cancel_memory_deletion are persistent-only
    tools. Routing here would push the model into an unfulfillable loop
    where the dispatcher rejects every resolution attempt.
    """

    policy = build_voice_turn_policy(
        user_text="Yes, delete that one.",
        memory_mode="incognito",
        has_active_guided_exercise=False,
        pending_memory_action=True,
    )

    assert policy.route == "therapeutic"
    assert policy.response_style == "voice"
    assert "Memory mode is incognito" in policy.instructions
    # Should not mention the persistent-only resolution tools.
    assert "confirm_memory_deletion" not in policy.instructions
    assert "cancel_memory_deletion" not in policy.instructions
