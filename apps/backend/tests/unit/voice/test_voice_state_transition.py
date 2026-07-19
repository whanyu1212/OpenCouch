from __future__ import annotations

import pytest

from agent.models import Channel
from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.events import VOICE_TURN_STATE_BUILT
from agent.observability.recorder import InMemoryTraceRecorder
from agent.state import AgentState
from agent.voice.state_transition import VoiceTurnStateInputs, build_voice_turn_state


def _initial_state() -> AgentState:
    return {
        "thread_id": "voice-thread",
        "channel": Channel.VOICE,
        "diagnostics": {},
        "grounded_lookup": {},
        "session_progress": {"turn_count": 0},
        "transcript": [],
    }


def test_build_voice_turn_state_appends_transcript_and_increments_turn_count() -> None:
    prior_state: AgentState = {
        "transcript": [{"role": "user", "content": "earlier"}],
        "session_progress": {"turn_count": 2},
        "diagnostics": {"existing": True},
        "grounded_lookup": {"existing": "kept"},
    }
    result = build_voice_turn_state(
        VoiceTurnStateInputs(
            thread_id="voice-thread",
            user_id="user-1",
            user_text="What's the latest guidance?",
            assistant_text="I found a verified answer.",
            route=None,
            response_style=None,
            tool_calls=[
                {
                    "tool_name": "answer_grounded_lookup",
                    "output": {"grounded_lookup": {"query": "latest guidance"}},
                }
            ],
            prior_state=prior_state,
            initial_state=_initial_state(),
            prior_turn_count=2,
        )
    )

    assert result.metadata.route == "grounded_lookup"
    assert result.metadata.response_style == "grounded_lookup"
    assert result.state["route"] == result.metadata.route
    assert result.state["response_style"] == result.metadata.response_style
    assert result.state["session_progress"]["turn_count"] == 3
    assert result.state["grounded_lookup"] == {
        "existing": "kept",
        "query": "latest guidance",
    }
    assert len(result.state["transcript"]) == 3
    assert result.state["transcript"][-1]["content"] == "I found a verified answer."


def test_build_voice_turn_state_emits_privacy_safe_trace_event() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-voice", config=TraceConfig(enabled=True))

    with use_trace_context(context, recorder):
        result = build_voice_turn_state(
            VoiceTurnStateInputs(
                thread_id="voice-thread",
                user_id="user-1",
                user_text="What's the latest guidance?",
                assistant_text="I found a verified answer.",
                route=None,
                response_style=None,
                tool_calls=[
                    {
                        "tool_name": "answer_grounded_lookup",
                        "output": {"grounded_lookup": {"query": "latest guidance"}},
                    }
                ],
                prior_state=None,
                initial_state=_initial_state(),
                prior_turn_count=0,
            )
        )

    assert result.metadata.route == "grounded_lookup"
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.name == VOICE_TURN_STATE_BUILT
    assert event.attributes == {
        "voice_runtime": "openai_realtime",
        "route": "grounded_lookup",
        "response_style": "grounded_lookup",
        "tool_call_count": 1,
        "resource_lookup_status": "not_attempted",
        "crisis_level": None,
    }
    assert "user_text" not in event.attributes
    assert "assistant_text" not in event.attributes
    assert "transcript" not in event.attributes


def test_build_voice_turn_state_ignores_client_reported_progress_delta() -> None:
    prior_state: AgentState = {
        "exercise_state": {
            "exercise_type": "grounding_box_breathing",
            "exercise_step": 1,
            "exercise_step_id": "hold_full",
            "exercise_version": 1,
            "exercise_therapeutic_approach": "dbt_skills",
        },
        "transcript": [],
    }
    result = build_voice_turn_state(
        VoiceTurnStateInputs(
            thread_id="voice-thread",
            user_id="user-1",
            user_text="I breathed in.",
            assistant_text="Good. Now hold gently.",
            route=None,
            response_style=None,
            tool_calls=[
                {
                    "tool_name": "record_guided_exercise_progress",
                    "output": {
                        "exercise_state_delta": {
                            "exercise_state": {
                                "exercise_type": None,
                                "exercise_step": None,
                                "exercise_step_id": None,
                                "exercise_version": None,
                                "exercise_therapeutic_approach": None,
                            }
                        }
                    },
                }
            ],
            prior_state=prior_state,
            initial_state=_initial_state(),
            prior_turn_count=0,
        )
    )

    assert result.metadata.route == "guided_exercise"
    assert result.metadata.response_style == "guided_exercise"
    assert result.state["exercise_state"] == {
        "exercise_type": "grounding_box_breathing",
        "exercise_step": 1,
        "exercise_step_id": "hold_full",
        "exercise_version": 1,
        "exercise_therapeutic_approach": "dbt_skills",
    }


def test_build_voice_turn_state_resets_stale_crisis_lookup_fields() -> None:
    prior_state: AgentState = {
        "resource_lookup_status": "found",
        "found_resources": [{"name": "Old hotline"}],
        "inferred_location": "Old City",
        "transcript": [],
    }
    result = build_voice_turn_state(
        VoiceTurnStateInputs(
            thread_id="voice-thread",
            user_id="user-1",
            user_text="I had a hard day.",
            assistant_text="Want to talk it through?",
            route=None,
            response_style="supportive",
            tool_calls=[],
            prior_state=prior_state,
            initial_state=_initial_state(),
            prior_turn_count=0,
        )
    )

    assert result.state["resource_lookup_status"] == "not_attempted"
    assert result.state["found_resources"] == []
    assert result.state["inferred_location"] == ""


def test_build_voice_turn_state_raises_for_empty_turn() -> None:
    with pytest.raises(
        ValueError, match="record_voice_turn requires user_text or assistant_text"
    ):
        build_voice_turn_state(
            VoiceTurnStateInputs(
                thread_id="voice-thread",
                user_id=None,
                user_text="",
                assistant_text="",
                route=None,
                response_style=None,
                tool_calls=[],
                prior_state=None,
                initial_state=_initial_state(),
                prior_turn_count=0,
            )
        )


def test_build_voice_turn_state_populates_crisis_audit_from_lookup_tool() -> None:
    result = build_voice_turn_state(
        VoiceTurnStateInputs(
            thread_id="voice-thread",
            user_id="user-1",
            user_text="I might hurt myself tonight.",
            assistant_text="I'm here with you right now.",
            route=None,
            response_style=None,
            tool_calls=[
                {
                    "tool_name": "lookup_crisis_resources",
                    "output": {
                        "resource_lookup_status": "found",
                        "found_resources": [{"name": "Samaritans", "phone": "1767"}],
                        "inferred_location": "Singapore",
                    },
                }
            ],
            prior_state=None,
            initial_state=_initial_state(),
            prior_turn_count=0,
        )
    )

    crisis = result.state["crisis"]
    assert result.metadata.route == "crisis"
    assert result.state["resource_lookup_status"] == "found"
    assert result.state["found_resources"] == [{"name": "Samaritans", "phone": "1767"}]
    assert result.state["inferred_location"] == "Singapore"
    assert crisis.level == 2
    assert crisis.reason == "voice_crisis_tool_call"
    assert result.state["diagnostics"]["openai_crisis_tool_calls"] == [
        "lookup_crisis_resources"
    ]
