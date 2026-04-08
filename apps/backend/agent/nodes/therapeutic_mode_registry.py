"""Registry-backed therapeutic mode generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from agent.models import ModeType, ResponseKind
from agent.prompts import (
    build_guided_exercise_response_prompt,
    build_guided_exercise_system_prompt,
    build_orientation_response_prompt,
    build_orientation_system_prompt,
    build_out_of_scope_response_prompt,
    build_out_of_scope_system_prompt,
    build_psychoeducation_response_prompt,
    build_psychoeducation_system_prompt,
    build_realignment_response_prompt,
    build_realignment_system_prompt,
    build_reflection_response_prompt,
    build_reflection_system_prompt,
    build_therapeutic_response_prompt,
    build_therapeutic_system_prompt,
)
from agent.prompts.catalog import Modality
from agent.response_shaping import (
    infer_guided_exercise_focus,
    infer_psychoeducation_topic,
    infer_support_strategy,
    needs_supportive_boundary,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def _fallback_supportive_response(state: AgentState) -> str:
    """Return the deterministic fallback for normal therapeutic replies."""

    if state.get("session_stage") == "closing":
        return (
            "It sounds like the most important thing from this conversation is that what you’re carrying has felt heavy, "
            "and you’ve started putting a little more shape around what you need. If it helps, the next step is to stay "
            "with one small thing that felt most grounding or clarifying today. We can pick this up again whenever you want."
        )
    if needs_supportive_boundary(state):
        return (
            "That sounds really stressful, especially on top of everything else you’re already carrying. "
            "I can help you slow it down, sort out what feels most overwhelming, or think through what support you need, "
            "but I can’t reliably guide the practical animal-care steps themselves. If it helps, we can focus on what feels hardest about trying again."
        )
    strategy = infer_support_strategy(state)
    if strategy == "hold_space":
        return (
            "That sounds like a lot to be carrying, and it makes sense that you’d want space to say it without having to turn it into a plan right away. "
            "I’m with you in it. You do not need to do anything with it in this moment beyond letting it be said."
        )
    if strategy == "strengths_based":
        return (
            "It sounds like something went differently this time, and that matters. Whatever you did there seems to say something real about your capacity, "
            "even if it did not feel perfect. It may be worth letting yourself notice that without immediately talking yourself out of it."
        )
    return "I’m here with you. Tell me a bit more about what feels hardest right now."


def _fallback_guided_exercise_response(state: AgentState) -> str:
    """Return a deterministic guided exercise reply."""

    if state.get("session_stage") == "closing":
        return (
            "Before we end, keep this very simple: hold onto one sentence or one step from today that felt most grounding or useful, "
            "and come back to it the next time this feeling shows up. You do not need to do a full exercise right now. The goal is just "
            "to leave with one small thing you can actually reuse."
        )

    focus = infer_guided_exercise_focus(state)
    if focus == "grounding":
        return (
            "Let's try one short grounding reset. Look around and name 5 things you can see, 4 things you can feel, "
            "3 things you can hear, 2 things you can smell, and 1 thing you can taste or imagine tasting. "
            "Take it slowly and just move through the list without trying to do it perfectly."
        )
    if focus == "behavioral_activation":
        return (
            "Let's make this very small and doable. Pick one action that takes about 5 to 10 minutes and is easier than your mind is telling you it has to be, "
            "like stepping outside, putting one dish away, sending one text, or taking a short shower. The goal is not to fix everything; it is to help your "
            "system feel a little more in motion again."
        )
    if focus == "acceptance":
        return (
            "Try this for a moment: name the thought or feeling in one short sentence, then add, 'I'm noticing that this is here right now.' "
            "That small shift can help you step back from the struggle a little. From there, choose one tiny action that still matters to you today, even if the feeling stays."
        )

    return (
        "Let's keep it simple and structured. Write down the situation, the main thought that showed up, "
        "the emotion you felt most strongly, and one alternative way to look at the same situation. "
        "You do not need to force a positive answer, just something a little more balanced."
    )


def _fallback_psychoeducation_response(state: AgentState) -> str:
    """Return a deterministic psychoeducation reply."""

    topic = infer_psychoeducation_topic(state)

    if topic == "anxiety_response":
        return (
            "What you’re describing can fit a common anxiety response: the body starts acting like something important or threatening is happening, "
            "even when the danger is more emotional than physical. That can show up as tension, racing thoughts, shallow breathing, or a strong urge "
            "to escape the situation. It does not mean anything is wrong with you; it often means your system is trying hard to protect you."
        )

    if topic == "stress_response":
        return (
            "When stress builds up for long enough, the mind and body can start staying in a more activated state even when the trigger is not constant. "
            "That can make concentration, rest, patience, and motivation all feel harder to access. In other words, your system may be acting like it has "
            "not had enough room to reset."
        )

    if topic == "grief_process":
        return (
            "Grief does not usually move in a straight line. People often have moments of numbness, sharp emotion, disorientation, or brief relief, and those "
            "states can alternate without warning. That shifting quality can make grief feel confusing, but it is a common part of how loss is processed."
        )

    return (
        "Sometimes it helps to know that emotional reactions are not just thoughts; they often involve body sensations, attention narrowing, and habits of "
        "protection that have built up over time. That can make a reaction feel automatic or bigger than you want it to be, even when part of you understands "
        "what is happening."
    )


def _fallback_reflection_response(state: AgentState) -> str:
    """Return a deterministic reflection reply."""

    if state.get("session_stage") == "closing":
        return (
            "What seems most important from this session is that this pattern has been pulling on you in a recurring way, "
            "and you’ve started naming it more clearly instead of just sitting inside it. A useful next step may be to notice "
            "the first moment this pattern shows up again and pause there. We can return to it when you’re ready."
        )

    if "grief_support" in state.get("active_modalities", []):
        return (
            "What stands out is how much this loss is still shaping the way things feel day to day. "
            "There seems to be both pain and a sense of being stuck with it, which can make grief feel very heavy. "
            "Does that feel close to what you mean?"
        )

    return (
        "A pattern I notice is that this seems to keep pulling you into the same emotional loop, even when part of you wants it to change. "
        "It sounds like the feeling itself and the meaning you attach to it may both be weighing on you. "
        "Does that fit, or is there a different pattern that feels more true to you?"
    )


def _fallback_orientation_response(state: AgentState) -> str:
    """Return a deterministic orientation reply."""

    del state
    return (
        "I can help you talk through difficult moments, reflect on patterns, and try simple "
        "self-help exercises when useful. I’m not a therapist or emergency service, but I "
        "can be a steady place to start. What feels most useful to focus on today?"
    )


def _fallback_out_of_scope_response(state: AgentState) -> str:
    """Return a deterministic out-of-scope reply."""

    del state
    return (
        "I can't diagnose that or give medication or legal guidance. "
        "If you want, I can help you describe what you're experiencing, think through questions to ask a licensed professional, "
        "or stay with the emotional side of what this brings up."
    )


def _fallback_realignment_response(state: AgentState) -> str:
    """Return a deterministic realignment reply."""

    del state
    return (
        "You're right, that missed the point. Let me slow down and try again more carefully. "
        "What feels most important or most off about what I just said?"
    )


@dataclass(frozen=True)
class TherapeuticModeConfig:
    """Prompt and fallback configuration for one therapeutic mode."""

    mode_type: ModeType
    default_modalities: tuple[Modality, ...]
    temperature: float
    prompt_builder: callable
    system_builder: callable
    fallback_builder: callable


THERAPEUTIC_MODE_CONFIGS: dict[str, TherapeuticModeConfig] = {
    "supportive_conversation": TherapeuticModeConfig(
        mode_type=ModeType.THERAPEUTIC,
        default_modalities=(),
        temperature=0.4,
        prompt_builder=build_therapeutic_response_prompt,
        system_builder=lambda modalities: build_therapeutic_system_prompt(
            modalities=cast(tuple[Modality, ...], modalities)
        ),
        fallback_builder=_fallback_supportive_response,
    ),
    "guided_exercise": TherapeuticModeConfig(
        mode_type=ModeType.THERAPEUTIC,
        default_modalities=("cbt",),
        temperature=0.3,
        prompt_builder=build_guided_exercise_response_prompt,
        system_builder=lambda modalities: build_guided_exercise_system_prompt(
            modalities=cast(tuple[Modality, ...], modalities)
        ),
        fallback_builder=_fallback_guided_exercise_response,
    ),
    "psychoeducation": TherapeuticModeConfig(
        mode_type=ModeType.THERAPEUTIC,
        default_modalities=("cbt",),
        temperature=0.3,
        prompt_builder=build_psychoeducation_response_prompt,
        system_builder=lambda modalities: build_psychoeducation_system_prompt(
            modalities=cast(tuple[Modality, ...], modalities)
        ),
        fallback_builder=_fallback_psychoeducation_response,
    ),
    "pattern_reflection": TherapeuticModeConfig(
        mode_type=ModeType.THERAPEUTIC,
        default_modalities=(),
        temperature=0.4,
        prompt_builder=build_reflection_response_prompt,
        system_builder=lambda modalities: build_reflection_system_prompt(
            modalities=cast(tuple[Modality, ...], modalities)
        ),
        fallback_builder=_fallback_reflection_response,
    ),
    "orientation": TherapeuticModeConfig(
        mode_type=ModeType.OPERATIONAL,
        default_modalities=(),
        temperature=0.3,
        prompt_builder=build_orientation_response_prompt,
        system_builder=lambda modalities: build_orientation_system_prompt(),
        fallback_builder=_fallback_orientation_response,
    ),
    "out_of_scope": TherapeuticModeConfig(
        mode_type=ModeType.OPERATIONAL,
        default_modalities=(),
        temperature=0.2,
        prompt_builder=build_out_of_scope_response_prompt,
        system_builder=lambda modalities: build_out_of_scope_system_prompt(),
        fallback_builder=_fallback_out_of_scope_response,
    ),
    "realignment": TherapeuticModeConfig(
        mode_type=ModeType.OPERATIONAL,
        default_modalities=(),
        temperature=0.3,
        prompt_builder=build_realignment_response_prompt,
        system_builder=lambda modalities: build_realignment_system_prompt(),
        fallback_builder=_fallback_realignment_response,
    ),
}


async def run_registered_therapeutic_mode_response(
    state: AgentState,
    *,
    mode: str,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Generate a reply using the registry-backed configuration for one mode."""

    config = THERAPEUTIC_MODE_CONFIGS[mode]
    state["response_type"] = ResponseKind.THERAPEUTIC
    state["mode"] = mode
    state["mode_type"] = config.mode_type
    modalities = cast(
        tuple[Modality, ...],
        tuple(state.get("active_modalities", list(config.default_modalities))),
    )

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=config.prompt_builder(state),
                system_instruction=config.system_builder(modalities),
                temperature=config.temperature,
            )
            return state
        except Exception:
            pass

    state["response_text"] = config.fallback_builder(state)
    return state
