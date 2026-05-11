"""LiveKit agent classes for the OpenCouch voice runtime."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, cast

from livekit.agents import (
    Agent,
    ChatContext,
    ChatMessage,
    RunContext,
    StopResponse,
    function_tool,
)
from pydantic import Field

from agent.gates.safety.service import CrisisRiskService
from agent.models import Channel, CrisisAssessment
from agent.state import AgentState
from agent.voice.activity import emit_voice_activity
from agent.voice.config import build_voice_system_prompt
from agent.voice.memory_context import VoiceMemoryContextService
from agent.voice.session_data import SessionData
from agent.voice.tasks import VoiceExerciseTask, supported_exercise_ids
from agent.voice.tools import (
    answer_grounded_factual_lookup,
    cancel_memory_deletion,
    confirm_memory_deletion,
    prepare_indexed_memory_deletion,
    prepare_memory_deletion,
    provide_crisis_resources,
    select_memory_deletion_candidate,
    set_proactive_memory_recall,
    show_memory_status,
    show_saved_memory,
)
from agent.voice.turn_policy import VoiceTurnPolicyService

logger = logging.getLogger(__name__)

ExerciseToolType = Literal[
    "behavioral_activation_tiny_action",
    "defusion_leaves_on_stream",
    "defusion_values_compass",
    "emotion_regulation_gratitude",
    "emotion_regulation_improve",
    "grounding_5_4_3_2_1",
    "grounding_box_breathing",
    "grounding_muscle_relaxation",
    "grounding_stop_technique",
    "self_compassion_break",
    "thought_work_behavioral_experiment",
    "thought_work_continuum",
    "thought_work_simple_record",
]

_THERAPEUTIC_PHASE_GUIDANCE = """\
Voice posture:
- Stay close to the user's latest experience before moving anywhere.
- Let turn guidance decide whether the next move is holding, orienting, identifying, examining, shifting, or grounding.
- Keep structure conversational. Do not become a worksheet, menu, or lecture.
- Do not start a structured exercise unless the current turn policy explicitly grants consent.
- If the user interrupts, yields, changes direction, or says an exercise is not helping, follow the interruption rather than forcing completion.
"""

_CRISIS_INSTRUCTIONS = """\
You are OpenCouch in crisis support mode. The user may be in danger.

YOUR ONLY PRIORITIES:
1. Stay calm, warm, and present. Do not panic or lecture.
2. Acknowledge what the user said. Do not minimize or dismiss.
3. If in the US or Canada, tell them they can call or text 988 (Suicide & Crisis Lifeline).
4. For other countries, use provide_crisis_resources when the user gives their location or asks for local resources.
5. Stay with the user. Do not rush to end the conversation.
6. If the user de-escalates and wants to continue talking, use the de_escalate tool.

Do NOT:
- Give medical or legal advice.
- Diagnose or label the user.
- Promise confidentiality you cannot guarantee.
- Attempt therapy techniques. Just be present.
- Invent hotline numbers or infer the user's location.
"""


def _chat_item_role(item: object) -> str:
    role = getattr(item, "role", "")
    return str(getattr(role, "value", role))


def _is_non_empty_dialogue_message(item: object) -> bool:
    if getattr(item, "type", None) != "message":
        return False
    if _chat_item_role(item) not in {"user", "assistant"}:
        return False
    return bool((getattr(item, "text_content", None) or "").strip())


def copy_dialogue_chat_ctx(chat_ctx: ChatContext | None) -> ChatContext | None:
    """Carry only user/assistant dialogue across agent and task boundaries."""

    if chat_ctx is None:
        return None
    return ChatContext(
        [item for item in chat_ctx.items if _is_non_empty_dialogue_message(item)]
    )


def _crisis_handoff_chat_ctx(
    turn_ctx: ChatContext | None,
    *,
    user_text: str,
) -> ChatContext:
    chat_ctx = copy_dialogue_chat_ctx(turn_ctx) or ChatContext()
    chat_ctx.add_message(role="user", content=user_text)
    return chat_ctx


def _history_from_chat_ctx(
    turn_ctx: ChatContext | None,
    *,
    user_text: str,
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    if turn_ctx is not None:
        for item in turn_ctx.items:
            if getattr(item, "type", None) != "message":
                continue
            role = _chat_item_role(item)
            if role not in {"user", "assistant"}:
                continue
            content = (getattr(item, "text_content", None) or "").strip()
            if content:
                history.append({"role": role, "content": content})
    history.append({"role": "user", "content": user_text})
    return history[-8:]


def _state_for_current_turn(
    userdata: SessionData,
    turn_ctx: ChatContext | None,
    *,
    user_text: str,
) -> AgentState:
    return cast(
        AgentState,
        {
            "message": user_text,
            "channel": Channel.VOICE,
            "user_id": userdata.user_id,
            "session_id": userdata.thread_id,
            "transcript": _history_from_chat_ctx(turn_ctx, user_text=user_text),
            "history": _history_from_chat_ctx(turn_ctx, user_text=user_text),
            "crisis": CrisisAssessment(),
        },
    )


def _build_policy_guidance_message(decision_reason: str, turn_guidance: str) -> str:
    return (
        "Voice turn policy for the next reply:\n"
        f"{turn_guidance.strip()}\n\n"
        "Policy note: use this as private guidance, not as text to recite.\n"
        f"Reason: {decision_reason.strip()}"
    )


def _has_current_turn_exercise_consent(
    userdata: SessionData,
    *,
    exercise_type: str,
) -> bool:
    if userdata.exercise_consent_turn_index != userdata.turn_index:
        return False
    if userdata.recommended_exercise_type is None:
        return True
    return exercise_type == userdata.recommended_exercise_type


def _compose_therapeutic_agent_instructions(*, base_instructions: str) -> str:
    return (
        f"{base_instructions}\n\n"
        "Memory control:\n"
        "- If the user asks what you remember or what is saved, call show_saved_memory.\n"
        "- If the user asks for memory status, call show_memory_status.\n"
        "- If the user asks you not to bring up past sessions or old memories, call set_proactive_memory_recall with enabled=false.\n"
        "- If the user says you may bring up past sessions when relevant, call set_proactive_memory_recall with enabled=true.\n"
        "- If the user asks you to forget or delete saved memory, first call a preparation tool. Do not delete anything until the user clearly confirms, then call confirm_memory_deletion.\n"
        "- If the user declines a pending deletion, call cancel_memory_deletion.\n\n"
        "Grounded factual lookup:\n"
        "- If the user explicitly asks you to look up, verify, check current information, or find factual resources, call answer_grounded_factual_lookup.\n"
        "- Do not call it for ordinary therapeutic support, coping tips, reflections, or exercise requests.\n"
        "- If a factual answer depends on location and the user did not provide one, ask what location they mean instead of guessing.\n\n"
        "Structured exercises:\n"
        "- Call start_grounding_exercise only when the private turn policy says exercise consent is granted for the current turn.\n"
        "- Pass an exact supported exercise_type id.\n"
        "- If consent is not granted, ask permission or stay in ordinary conversation.\n\n"
        f"{_THERAPEUTIC_PHASE_GUIDANCE}"
    )


class TherapeuticAgent(Agent):
    """Main conversational agent for OpenCouch voice sessions."""

    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: ChatContext | None = None,
        greet_on_enter: bool = False,
        greet_delay_seconds: float = 0.0,
        turn_policy_service: VoiceTurnPolicyService | None = None,
        memory_context_service: VoiceMemoryContextService | None = None,
        crisis_risk_service: CrisisRiskService | None = None,
    ) -> None:
        self._base_instructions = instructions
        self._greet_on_enter = greet_on_enter
        self._greet_delay_seconds = greet_delay_seconds
        self._turn_policy_service = turn_policy_service or VoiceTurnPolicyService()
        self._memory_context_service = (
            memory_context_service or VoiceMemoryContextService()
        )
        self._crisis_risk_service = crisis_risk_service or CrisisRiskService()
        super().__init__(
            instructions=_compose_therapeutic_agent_instructions(
                base_instructions=instructions,
            ),
            chat_ctx=chat_ctx,
            tools=[
                show_saved_memory,
                show_memory_status,
                set_proactive_memory_recall,
                prepare_memory_deletion,
                prepare_indexed_memory_deletion,
                select_memory_deletion_candidate,
                confirm_memory_deletion,
                cancel_memory_deletion,
                answer_grounded_factual_lookup,
            ],
        )

    async def on_enter(self) -> None:
        if not self._greet_on_enter:
            return

        if self._greet_delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self._greet_delay_seconds)

        await self.session.generate_reply(
            instructions="Greet the user briefly and warmly. Sound like a calm "
            "person joining them, not an intake form. One sentence is enough."
        )

    @function_tool()
    async def start_grounding_exercise(
        self,
        context: RunContext[SessionData],
        exercise_type: Annotated[
            ExerciseToolType,
            Field(
                description=(
                    "Exact supported exercise_type id. Use grounding_box_breathing "
                    "for box breathing, grounding_5_4_3_2_1 for sensory "
                    "grounding, grounding_stop_technique for STOP, and "
                    "grounding_muscle_relaxation for muscle relaxation."
                ),
            ),
        ],
    ) -> str:
        """Start a guided voice-safe exercise after current-turn consent."""

        normalized_exercise_type = exercise_type.strip()
        supported_ids = supported_exercise_ids(context.userdata.last_input_modality)
        if normalized_exercise_type not in supported_ids:
            supported = ", ".join(supported_ids)
            logger.warning(
                "TherapeuticAgent: unsupported exercise_type=%s modality=%s",
                normalized_exercise_type,
                context.userdata.last_input_modality,
            )
            return (
                "Do not start a structured exercise yet. The exercise_type "
                f"{normalized_exercise_type!r} is not supported for this "
                "modality. Choose one of these supported exercise_type ids and "
                f"call the tool again only if the user has agreed: {supported}."
            )

        if not _has_current_turn_exercise_consent(
            context.userdata,
            exercise_type=normalized_exercise_type,
        ):
            logger.info("TherapeuticAgent: blocked exercise without turn consent")
            return (
                "Do not start a structured exercise yet. The current turn policy "
                "has not granted exercise consent. Stay in ordinary conversation, "
                "or ask permission before offering a structured exercise."
            )

        await emit_voice_activity(
            context,
            activity="exercise",
            status="started",
            label="Exercise active",
            detail="A guided exercise is in progress.",
        )
        result = await VoiceExerciseTask(
            exercise_type=normalized_exercise_type,
            chat_ctx=copy_dialogue_chat_ctx(self.chat_ctx),
            input_modality=context.userdata.last_input_modality,
        )
        await emit_voice_activity(
            context,
            activity="exercise",
            status="completed" if result.outcome == "completed" else "cancelled",
            label=(
                "Exercise completed"
                if result.outcome == "completed"
                else "Exercise stopped"
            ),
            detail=(
                "The guided exercise finished."
                if result.outcome == "completed"
                else "The guided exercise was stopped early."
            ),
        )
        recent = [
            item
            for item in context.userdata.recent_exercise_types
            if item != result.exercise_type
        ]
        recent.append(result.exercise_type)
        context.userdata.recent_exercise_types = recent[-3:]
        raise StopResponse()

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """Run LLM-backed crisis, turn-policy, and memory-context services."""

        text = (new_message.text_content or "").strip()
        if not text:
            return

        userdata: SessionData = self.session.userdata
        userdata.turn_index += 1
        userdata.exercise_consent_turn_index = None
        userdata.exercise_consent_reason = ""
        userdata.recommended_exercise_type = None

        state = _state_for_current_turn(userdata, turn_ctx, user_text=text)
        crisis_result = await self._crisis_risk_service.assess_turn(
            state,
            llm_client=userdata.llm_client,
        )
        assessment = crisis_result.assessment
        userdata.crisis_level = assessment.level
        userdata.max_crisis_level = max(userdata.max_crisis_level, assessment.level)

        if assessment.level >= 2:
            logger.warning(
                "voice crisis gate: level=%s user=%s reason=%s",
                assessment.level,
                userdata.user_id,
                assessment.reason[:120],
            )
            self.session.update_agent(
                CrisisAgent(chat_ctx=_crisis_handoff_chat_ctx(turn_ctx, user_text=text))
            )
            return

        if assessment.level == 1:
            turn_ctx.add_message(
                role="system",
                content=(
                    "Voice crisis clarification for the next reply: the user's "
                    "message is concerning but ambiguous. Gently ask one direct "
                    "safety clarification question without escalating into a full "
                    "crisis response. Do not provide hotline numbers unless the "
                    "user confirms self-harm intent or asks for resources."
                ),
                extra={
                    "source": "voice_crisis_clarification",
                    "reason": assessment.reason,
                },
            )
            return

        supported_ids = supported_exercise_ids(userdata.last_input_modality)
        decision = await self._turn_policy_service.plan_turn(
            user_text=text,
            chat_ctx=turn_ctx,
            previous_state=userdata.therapeutic_state,
            supported_exercise_ids=supported_ids,
            recent_exercise_types=userdata.recent_exercise_types,
            llm_client=userdata.llm_client,
        )
        userdata.therapeutic_state = decision.to_process_state()
        logger.info(
            "voice turn policy: intent=%s stage=%s exercise_consent=%s exercise_type=%s confidence=%s",
            decision.session_intent,
            decision.process_stage,
            decision.exercise_consent,
            decision.exercise_type,
            decision.confidence,
        )
        if decision.exercise_consent == "granted":
            userdata.exercise_consent_turn_index = userdata.turn_index
            userdata.exercise_consent_reason = decision.reason
            userdata.recommended_exercise_type = decision.exercise_type

        turn_ctx.add_message(
            role="system",
            content=_build_policy_guidance_message(
                decision_reason=decision.reason,
                turn_guidance=decision.turn_guidance,
            ),
            extra={
                "source": "voice_turn_policy",
                "confidence": decision.confidence,
                "exercise_consent": decision.exercise_consent,
                "exercise_type": decision.exercise_type,
            },
        )

        injected_keys = await self._memory_context_service.inject_turn_relevant_memory(
            self,
            turn_ctx,
            user_id=userdata.user_id,
            mode=userdata.memory_mode,
            query=text,
            store=userdata.memory_store,
            already_injected_keys=userdata.injected_semantic_memory_keys,
            proactive_recall_enabled=userdata.proactive_recall_enabled,
        )
        userdata.injected_semantic_memory_keys.update(injected_keys)


class CrisisAgent(Agent):
    """Crisis support agent activated on level 2+ crisis signals."""

    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=_CRISIS_INSTRUCTIONS,
            chat_ctx=chat_ctx,
            tools=[provide_crisis_resources],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="The user may be in crisis. Acknowledge the immediate "
            "risk plainly and warmly. Give 988 for US/Canada if relevant. Keep "
            "your voice steady and stay with them; do not sound like a policy notice."
        )

    @function_tool()
    async def de_escalate(self, context: RunContext[SessionData]) -> str:
        """Return to the therapeutic agent after clear de-escalation."""

        logger.info("CrisisAgent: de-escalating back to TherapeuticAgent")
        context.userdata.crisis_level = 0

        instructions = (
            context.userdata.therapeutic_instructions or build_voice_system_prompt()
        )
        return (
            TherapeuticAgent(
                instructions=instructions,
                chat_ctx=copy_dialogue_chat_ctx(self.chat_ctx),
            ),
            "The user has de-escalated. Transitioning back to supportive conversation.",
        )


def build_therapeutic_agent(
    *,
    instructions: str,
    chat_ctx: ChatContext | None = None,
    greet_on_enter: bool = False,
    greet_delay_seconds: float = 0.0,
) -> TherapeuticAgent:
    """Construct the default therapeutic voice agent."""

    return TherapeuticAgent(
        instructions=instructions,
        chat_ctx=chat_ctx,
        greet_on_enter=greet_on_enter,
        greet_delay_seconds=greet_delay_seconds,
    )
