import pytest

from agent.graph import build_initial_state
from agent.models import AgentInput, Message, MessageRole, ModeType
from agent.modality_selector import select_modalities_for_mode
from agent.subgraphs.therapeutic import (
    run_therapeutic_subgraph,
    select_therapeutic_mode,
)


@pytest.mark.asyncio
async def test_orientation_mode_routes_for_first_time_orientation_message() -> None:
    """First-time orientation language should route to orientation mode."""

    state = build_initial_state(AgentInput(message="How does this work? I'm new here."))

    mode, source = await select_therapeutic_mode(state)
    assert mode == "orientation"
    assert source == "keyword"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "orientation"
    assert state["mode_type"] == ModeType.OPERATIONAL
    assert "Explain what OpenCouch can help with" in state["response_guidance"]
    assert "I can help you talk through difficult moments" in state["response_text"]


@pytest.mark.asyncio
async def test_orientation_mode_routes_for_capability_question() -> None:
    """Capability questions should route to orientation without the LLM."""

    state = build_initial_state(AgentInput(message="Hi, what can you do for me?"))

    mode, source = await select_therapeutic_mode(state)
    assert mode == "orientation"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_guided_exercise_mode_routes_for_grounding_request() -> None:
    """Grounding requests should route to guided exercise mode."""

    state = build_initial_state(
        AgentInput(message="Can you give me a grounding exercise to help me calm down?")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "guided_exercise"
    assert source == "keyword"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "guided_exercise"
    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "grounding or regulation" in state["response_guidance"]
    assert "5 things you can see" in state["response_text"]


@pytest.mark.asyncio
async def test_psychoeducation_mode_routes_for_anxiety_explanation_request() -> None:
    """Psychoeducation requests should route to psychoeducation mode."""

    state = build_initial_state(
        AgentInput(
            message="Can you explain why my body reacts like this when I get anxious?"
        )
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "psychoeducation"
    assert source in {"keyword", "session_intent"}

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "psychoeducation"
    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "normalizing explanation" in state["response_guidance"]
    assert (
        "body" in state["response_text"].lower()
        or "system" in state["response_text"].lower()
    )


@pytest.mark.asyncio
async def test_reflection_mode_routes_for_pattern_request() -> None:
    """Pattern-seeking requests should route to pattern-review mode."""

    state = build_initial_state(
        AgentInput(
            message="Can you help me understand why I keep getting stuck in this pattern?"
        )
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "pattern_reflection"
    assert source == "keyword"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "pattern_reflection"
    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "Pattern reflection should" in state["response_guidance"]
    assert "A pattern I notice" in state["response_text"]


@pytest.mark.asyncio
async def test_out_of_scope_mode_routes_for_diagnosis_request() -> None:
    """Diagnosis requests should route to the out-of-scope boundary mode."""

    state = build_initial_state(
        AgentInput(
            message="Can you diagnose me and tell me what medication I should take?"
        )
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "out_of_scope"
    assert source == "keyword"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "out_of_scope"
    assert state["mode_type"] == ModeType.OPERATIONAL
    assert "Decline clearly" in state["response_guidance"]
    assert "can't diagnose" in state["response_text"]


@pytest.mark.asyncio
async def test_realignment_mode_routes_when_user_says_reply_missed() -> None:
    """Missed-response language should route to realignment mode."""

    state = build_initial_state(
        AgentInput(message="That's not what I meant. You misunderstood me.")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "realignment"
    assert source == "keyword"

    state = await run_therapeutic_subgraph(state)
    assert state["mode"] == "realignment"
    assert state["mode_type"] == ModeType.OPERATIONAL
    assert "Acknowledge the miss directly" in state["response_guidance"]
    assert "missed the point" in state["response_text"]


@pytest.mark.asyncio
async def test_session_intent_biases_ambiguous_follow_up_toward_guided_exercise() -> (
    None
):
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

    mode, source = await select_therapeutic_mode(state)
    assert mode == "guided_exercise"
    assert source == "session_intent"


@pytest.mark.asyncio
async def test_explicit_intent_shift_is_allowed_mid_session() -> None:
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

    mode, source = await select_therapeutic_mode(state)
    assert mode == "supportive_conversation"
    assert source == "session_intent"


@pytest.mark.asyncio
async def test_supportive_conversation_sets_hold_space_guidance_for_venting_intent() -> (
    None
):
    """Venting intent should compile into hold-space response guidance."""

    state = build_initial_state(
        AgentInput(message="I just want to vent. I do not want advice right now.")
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "supportive_conversation"
    assert "Hold space" in state["response_guidance"]


@pytest.mark.asyncio
async def test_anxiety_disclosure_routes_to_support_without_llm() -> None:
    """Ordinary anxiety disclosures should route deterministically to support."""

    state = build_initial_state(
        AgentInput(
            message=(
                "I'm feeling really anxious lately. It's like my body and mind are "
                "running non-stop and I feel drained."
            )
        )
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "supportive_conversation"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_coping_question_routes_to_guided_exercise_without_llm() -> None:
    """Action-oriented anxiety coping asks should route deterministically."""

    state = build_initial_state(
        AgentInput(
            message="What can I do to navigate the anxiety if it keeps bothering me?",
            history=[
                Message(
                    role=MessageRole.USER,
                    content=(
                        "I'm feeling really anxious lately. It's like my body and mind "
                        "are running non-stop and I feel drained."
                    ),
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="That sounds exhausting. Which part feels louder right now?",
                ),
            ],
        )
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "guided_exercise"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_practical_pet_medication_turn_gets_supportive_boundary_guidance() -> (
    None
):
    """Stressful pet-medication asks should stay supportive but set a boundary."""

    state = build_initial_state(
        AgentInput(
            message="My cat is unwell and I'm having a lot of difficulty feeding it medications.",
            history=[
                Message(
                    role=MessageRole.USER,
                    content="Technically I can tackle them one by one but it feels dreadful doing so.",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="That dread can make it hard to start the first thing.",
                ),
            ],
        )
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "supportive_conversation"
    assert "avoid procedural advice" in state["response_guidance"]


@pytest.mark.asyncio
async def test_psychoeducation_intent_biases_ambiguous_follow_up_toward_psychoeducation() -> (
    None
):
    """Stored psychoeducation intent should bias an ambiguous follow-up."""

    state = build_initial_state(
        AgentInput(
            message="Can you say a little more about that?",
            history=[
                Message(
                    role=MessageRole.USER,
                    content="Can you explain what anxiety is doing in my body?",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Anxiety can involve both body activation and racing thoughts.",
                ),
            ],
        )
    )

    assert state["session_intent"] == "psychoeducation"

    mode, source = await select_therapeutic_mode(state)
    assert mode == "psychoeducation"
    assert source == "session_intent"


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

    assert state["mode"] == "supportive_conversation"
    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "grief_support" not in state["active_modalities"]
    assert "relational strain or role transition" in state["response_guidance"]


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

    assert state["mode"] == "pattern_reflection"
    assert "communication patterns" in state["response_guidance"]


@pytest.mark.asyncio
async def test_guided_exercise_mode_selects_pfa_for_grounding_request() -> None:
    """Grounding requests should add the stabilization overlay."""

    state = build_initial_state(
        AgentInput(
            message="Can you give me a grounding exercise? I feel overwhelmed and need to calm down."
        )
    )

    state = await run_therapeutic_subgraph(state)

    assert state["mode"] == "guided_exercise"
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

    assert state["mode"] == "supportive_conversation"
    assert "act" in state["active_modalities"]


def test_state_concerns_shape_relational_guidance_without_relational_keywords() -> None:
    """Concern labels should be able to trigger relational guidance."""

    state = build_initial_state(AgentInput(message="I do not know what to do next."))
    state["active_concerns"] = ["relationship strain"]
    state["current_goal"] = "understand a recurring pattern"
    state["semantic_signals"] = {}
    state["response_guidance"] = ""

    from agent.response_shaping import build_response_guidance

    guidance = build_response_guidance(state, mode="pattern_reflection")

    assert "communication patterns" in guidance


def test_state_goal_can_select_cbt_overlay_without_explicit_cbt_keywords() -> None:
    """Current goal should be able to trigger CBT overlay selection."""

    state = build_initial_state(AgentInput(message="Okay, let's work on it."))
    state["current_goal"] = "work through a structured exercise"

    modalities = select_modalities_for_mode(state, "guided_exercise")

    assert "cbt" in modalities


def test_stage_and_goal_bias_grounding_toward_pfa() -> None:
    """Opening-stage grounding goals should prioritize the stabilization overlay."""

    state = build_initial_state(AgentInput(message="Can we keep going?"))
    state["session_stage"] = "opening"
    state["current_goal"] = "feel calmer right now"
    state["active_concerns"] = ["anxiety or rumination"]

    modalities = select_modalities_for_mode(state, "guided_exercise")

    assert modalities[0] == "pfa"


def test_support_policy_keeps_modalities_bounded() -> None:
    """Support modality selection should keep the overlay set intentionally small."""

    state = build_initial_state(
        AgentInput(
            message="I feel lonely, anxious, and stuck after this breakup and I want a CBT-style plan."
        )
    )
    state["active_concerns"] = [
        "relationship strain",
        "anxiety or rumination",
        "grief or loss",
    ]
    state["current_goal"] = "work through a structured exercise"

    modalities = select_modalities_for_mode(state, "supportive_conversation")

    assert len(modalities) <= 3


# --- Option A: New keyword expansion tests ---


@pytest.mark.asyncio
async def test_exercise_routes_for_panic() -> None:
    """Panic language should route to guided exercise."""

    state = build_initial_state(
        AgentInput(message="I'm panicking and I can't breathe.")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "guided_exercise"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_exercise_routes_for_overwhelmed() -> None:
    """Overwhelmed language should route to guided exercise."""

    state = build_initial_state(AgentInput(message="I feel so overwhelmed right now."))

    mode, source = await select_therapeutic_mode(state)
    assert mode == "guided_exercise"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_exercise_routes_for_walk_through_request() -> None:
    """Walk-through requests should route to guided exercise."""

    state = build_initial_state(
        AgentInput(message="Can you walk me through something to help?")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "guided_exercise"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_repair_routes_for_not_helpful() -> None:
    """'That's not helpful' should route to realignment."""

    state = build_initial_state(AgentInput(message="That's not helpful at all."))

    mode, source = await select_therapeutic_mode(state)
    assert mode == "realignment"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_repair_routes_for_not_getting_it() -> None:
    """'You're not getting it' should route to realignment."""

    state = build_initial_state(AgentInput(message="You're not getting it."))

    mode, source = await select_therapeutic_mode(state)
    assert mode == "realignment"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_psychoeducation_routes_for_is_this_normal() -> None:
    """Explicit anxiety/body-process questions should route to psychoeducation."""

    state = build_initial_state(
        AgentInput(message="Is it normal for anxiety to make my body shake like this?")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "psychoeducation"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_relational_distress_question_does_not_false_positive_to_psychoeducation() -> (
    None
):
    """Generic relational distress questions should stay out of psychoeducation."""

    state = build_initial_state(
        AgentInput(message="Why am I so upset after talking to my sister?")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "supportive_conversation"
    assert source == "session_intent"


@pytest.mark.asyncio
async def test_normalizing_relational_question_does_not_false_positive_to_psychoeducation() -> (
    None
):
    """'Is it normal' should not hijack relational-pattern questions."""

    state = build_initial_state(
        AgentInput(message="Is it normal to keep replaying arguments with my partner?")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "supportive_conversation"
    assert source == "default"


@pytest.mark.asyncio
async def test_reflection_routes_for_theme_question() -> None:
    """'Is there a theme here?' should route to pattern reflection."""

    state = build_initial_state(AgentInput(message="Is there a theme here?"))

    mode, source = await select_therapeutic_mode(state)
    assert mode == "pattern_reflection"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_reflection_routes_for_connection_question() -> None:
    """'Do you see a connection?' should route to pattern reflection."""

    state = build_initial_state(
        AgentInput(message="Do you see a connection between these?")
    )

    mode, source = await select_therapeutic_mode(state)
    assert mode == "pattern_reflection"
    assert source == "keyword"


@pytest.mark.asyncio
async def test_reflection_routes_for_keeps_happening() -> None:
    """'What keeps happening?' should route to pattern reflection."""

    state = build_initial_state(AgentInput(message="What keeps happening to me?"))

    mode, source = await select_therapeutic_mode(state)
    assert mode == "pattern_reflection"
    assert source == "keyword"
