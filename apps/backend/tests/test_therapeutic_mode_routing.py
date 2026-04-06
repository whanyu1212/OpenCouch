import pytest

from agent.graph import build_initial_state
from agent.models import AgentInput, Message, MessageRole
from agent.subgraphs.therapeutic import (
    run_therapeutic_subgraph,
    select_therapeutic_mode,
)


@pytest.mark.asyncio
async def test_orientation_mode_routes_for_first_time_orientation_message() -> None:
    """First-time orientation language should route to orientation mode."""

    state = build_initial_state(AgentInput(message="How does this work? I'm new here."))

    assert select_therapeutic_mode(state) == "orientation"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "orientation"
    assert "I can help you talk through difficult moments" in state["response_text"]


@pytest.mark.asyncio
async def test_guided_exercise_mode_routes_for_grounding_request() -> None:
    """Grounding requests should route to guided exercise mode."""

    state = build_initial_state(
        AgentInput(message="Can you give me a grounding exercise to help me calm down?")
    )

    assert select_therapeutic_mode(state) == "guided_exercise"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "guided_exercise"
    assert "5 things you can see" in state["response_text"]


@pytest.mark.asyncio
async def test_reflection_mode_routes_for_pattern_request() -> None:
    """Pattern-seeking requests should route to reflection mode."""

    state = build_initial_state(
        AgentInput(
            message="Can you help me understand why I keep getting stuck in this pattern?"
        )
    )

    assert select_therapeutic_mode(state) == "reflection"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "reflection"
    assert "A pattern I notice" in state["response_text"]


@pytest.mark.asyncio
async def test_out_of_scope_mode_routes_for_diagnosis_request() -> None:
    """Diagnosis requests should route to the out-of-scope boundary mode."""

    state = build_initial_state(
        AgentInput(
            message="Can you diagnose me and tell me what medication I should take?"
        )
    )

    assert select_therapeutic_mode(state) == "out_of_scope"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "out_of_scope"
    assert "can't diagnose" in state["response_text"]


@pytest.mark.asyncio
async def test_realignment_mode_routes_when_user_says_reply_missed() -> None:
    """Missed-response language should route to realignment mode."""

    state = build_initial_state(
        AgentInput(message="That's not what I meant. You misunderstood me.")
    )

    assert select_therapeutic_mode(state) == "realignment"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "realignment"
    assert "missed the point" in state["response_text"]


def test_session_intent_biases_ambiguous_follow_up_toward_guided_exercise() -> None:
    """Stored CBT intent should bias ambiguous follow-ups toward guided exercise."""

    state = build_initial_state(
        AgentInput(
            message="Okay, let's do that.",
            history=[
                Message(
                    role=MessageRole.USER,
                    content="I want to do CBT for this negative thought.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="We can work through that together.",
                ),
            ],
        )
    )

    state["session_intent"] = "guided_cbt_work"
    state["session_intent_source"] = "explicit"

    assert select_therapeutic_mode(state) == "guided_exercise"


def test_explicit_intent_shift_is_allowed_mid_session() -> None:
    """Explicit mid-session intent shifts should update the routing bias."""

    state = build_initial_state(
        AgentInput(
            message="Actually I just want to vent today. I don't want advice.",
            history=[
                Message(
                    role=MessageRole.USER,
                    content="I want to do CBT for this negative thought.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="We can work through that together.",
                ),
            ],
        )
    )

    assert state["session_intent"] == "just_need_to_vent"
    assert state["session_intent_source"] == "explicit"
    assert select_therapeutic_mode(state) == "support"


@pytest.mark.asyncio
async def test_support_mode_selects_interpersonal_overlay_for_relationship_distress() -> (
    None
):
    """Relationship distress should add the interpersonal therapy overlay."""

    state = build_initial_state(
        AgentInput(
            message="I keep fighting with my partner and I feel lonely in this relationship."
        )
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "support"
    assert "motivational_interviewing" in state["active_modalities"]
    assert "interpersonal_therapy" in state["active_modalities"]


@pytest.mark.asyncio
async def test_reflection_mode_selects_interpersonal_overlay_for_patterned_relationship_issue() -> (
    None
):
    """Relational pattern reflection should add the interpersonal therapy overlay."""

    state = build_initial_state(
        AgentInput(
            message="Can you help me understand why I keep ending up in the same conflict with my family?"
        )
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "reflection"
    assert "interpersonal_therapy" in state["active_modalities"]


@pytest.mark.asyncio
async def test_guided_exercise_mode_selects_dbt_skills_for_grounding_request() -> None:
    """Grounding requests should add DBT-skills and PFA overlays."""

    state = build_initial_state(
        AgentInput(
            message="Can you give me a grounding exercise? I feel overwhelmed and need to calm down."
        )
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "guided_exercise"
    assert "dbt_skills" in state["active_modalities"]
    assert "pfa" in state["active_modalities"]


@pytest.mark.asyncio
async def test_support_mode_selects_act_for_avoidance_and_rumination() -> None:
    """Avoidance and rumination should add the ACT overlay."""

    state = build_initial_state(
        AgentInput(
            message="I keep spiraling and avoiding everything because the anxiety won't stop."
        )
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "support"
    assert "act" in state["active_modalities"]
