"""LLM-primary turn policy for LiveKit voice conversations."""

from __future__ import annotations

from typing import Literal

from livekit.agents import ChatContext
from pydantic import BaseModel, Field

from agent.voice.session_data import (
    GuidancePermission,
    ProcessStage,
    SessionIntent,
    TherapeuticApproach,
    TherapeuticFormulation,
    TherapeuticProcessState,
)
from llm.base import BaseLLMClient


ExerciseConsent = Literal["none", "granted"]


class VoiceTurnPolicyDecision(BaseModel):
    """Structured decision for one non-crisis voice turn."""

    session_intent: SessionIntent
    guidance_permission: GuidancePermission
    process_stage: ProcessStage
    therapeutic_approach: TherapeuticApproach
    active_target: str = Field(default="", max_length=180)
    primary_emotion: str = Field(default="", max_length=80)
    hot_thought: str = Field(default="", max_length=140)
    pattern: str = Field(default="", max_length=160)
    user_goal: str = Field(default="", max_length=180)
    exercise_consent: ExerciseConsent = Field(
        description=(
            "Use granted only when the current user turn explicitly asks for a "
            "structured exercise, including direct question forms like 'can we "
            "do box breathing now?', or clearly agrees after the assistant "
            "offered one."
        )
    )
    exercise_type: str | None = Field(
        default=None,
        description=(
            "Exact supported exercise id when exercise_consent is granted and a "
            "specific exercise is appropriate."
        ),
    )
    turn_guidance: str = Field(
        min_length=1,
        max_length=900,
        description="Compact system guidance for the next spoken reply.",
    )
    reason: str = Field(min_length=1, max_length=360)
    confidence: Literal["low", "medium", "high"]

    def to_process_state(self) -> TherapeuticProcessState:
        """Convert the structured policy decision to session state."""

        return TherapeuticProcessState(
            session_intent=self.session_intent,
            guidance_permission=self.guidance_permission,
            process_stage=self.process_stage,
            therapeutic_approach=self.therapeutic_approach,
            active_target=self.active_target,
            formulation=TherapeuticFormulation(
                situation=self.active_target,
                primary_emotion=self.primary_emotion,
                hot_thought=self.hot_thought,
                pattern=self.pattern,
                user_goal=self.user_goal,
            ),
        )


def _chat_role(item: object) -> str:
    role = getattr(item, "role", "")
    return str(getattr(role, "value", role))


def format_voice_history(chat_ctx: ChatContext | None, *, limit: int = 8) -> str:
    """Return a compact user/assistant history block for policy prompts."""

    if chat_ctx is None:
        return "(none)"

    turns: list[str] = []
    for item in getattr(chat_ctx, "items", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        role = _chat_role(item)
        if role not in {"user", "assistant"}:
            continue
        text = (getattr(item, "text_content", None) or "").strip()
        if text:
            turns.append(f"{role}: {text}")
    return "\n".join(turns[-limit:]) or "(none)"


def build_voice_turn_policy_system_prompt() -> str:
    """Build the system instruction for voice turn policy decisions."""

    return (
        "You are a strict voice conversation policy planner for a mental-health "
        "support assistant. Hard rule: when the current user directly asks to "
        "start, try, do, or be guided through a structured exercise, set "
        "exercise_consent=granted. Do not turn that direct request into another "
        "readiness-confirmation question. Return only the structured decision. "
        "Do not answer the user."
    )


def build_voice_turn_policy_prompt(
    *,
    user_text: str,
    chat_ctx: ChatContext | None,
    previous_state: TherapeuticProcessState,
    supported_exercise_ids: tuple[str, ...],
    recent_exercise_types: list[str],
) -> str:
    """Build the prompt for one non-crisis voice policy decision."""

    supported = ", ".join(supported_exercise_ids) or "(none)"
    recent = ", ".join(recent_exercise_types) or "(none)"
    return (
        "Plan the next spoken therapeutic turn.\n\n"
        "Boundaries:\n"
        "- Prefer ordinary conversation, emotional attunement, and one focused "
        "next step over structured exercises.\n"
        "- Set exercise_consent=granted only when the current user turn "
        "explicitly asks for a structured exercise or clearly agrees after the "
        "assistant offered one. Do not infer consent from distress alone.\n"
        "- A direct exercise request MUST be treated as consent even when "
        "phrased as a question and even if the assistant did not name that exact "
        "exercise in the previous turn. "
        "Examples: 'Can we do box breathing now?', 'Could you guide me through "
        "5-4-3-2-1?', 'I want to try grounding', or 'Let's do the breathing "
        "exercise.' Set exercise_consent=granted for these and do not require a "
        "second readiness confirmation.\n"
        "- Treat follow-up choices like 'yes, let's do that', 'let's do box "
        "breathing', 'the breathing one', or 'can we do 5-4-3-2-1' as explicit "
        "exercise consent when recent conversation shows the assistant offered "
        "an exercise or options.\n"
        "- Map common names to exact supported ids when possible: box breathing "
        "to grounding_box_breathing; 5-4-3-2-1 or five senses grounding to "
        "grounding_5_4_3_2_1.\n"
        "- If exercise_consent is granted, choose an exact supported exercise id. "
        "If the user did not name a method, pick a voice-safe option and avoid "
        "recent repeats when practical.\n"
        "- If the user sounds overwhelmed but has not consented to an exercise, "
        "turn_guidance should ask permission or offer a conversational micro-step.\n"
        "- Keep turn_guidance compact and usable as a system note for the next "
        "voice reply. It should guide posture and one next move, not produce a "
        "script.\n\n"
        "Labels:\n"
        "- session_intent: vent, understand, reflect, work, regulate, close.\n"
        "- guidance_permission: unknown, not_yet, granted.\n"
        "- process_stage: hold, orient, identify, examine, shift, ground.\n"
        "- therapeutic_approach: motivational_interviewing, cbt, act, "
        "dbt_skills, grief_support, interpersonal_therapy, pfa, none.\n\n"
        f"Supported exercise ids: {supported}\n"
        f"Recent exercise ids: {recent}\n"
        "Previous policy state:\n"
        f"- session_intent: {previous_state.session_intent}\n"
        f"- guidance_permission: {previous_state.guidance_permission}\n"
        f"- process_stage: {previous_state.process_stage}\n"
        f"- therapeutic_approach: {previous_state.therapeutic_approach}\n"
        f"- active_target: {previous_state.active_target or '(none)'}\n\n"
        "Recent conversation:\n"
        f"{format_voice_history(chat_ctx)}\n\n"
        f'Current user message: "{user_text}"'
    )


class VoiceTurnPolicyService:
    """LLM-backed policy service for one safe voice turn."""

    async def plan_turn(
        self,
        *,
        user_text: str,
        chat_ctx: ChatContext | None,
        previous_state: TherapeuticProcessState,
        supported_exercise_ids: tuple[str, ...],
        recent_exercise_types: list[str],
        llm_client: BaseLLMClient | None,
    ) -> VoiceTurnPolicyDecision:
        """Return the structured voice policy decision for a safe user turn."""

        if llm_client is None:
            raise RuntimeError("Voice turn policy requires an LLM client.")

        decision = await llm_client.generate_structured(
            prompt=build_voice_turn_policy_prompt(
                user_text=user_text,
                chat_ctx=chat_ctx,
                previous_state=previous_state,
                supported_exercise_ids=supported_exercise_ids,
                recent_exercise_types=recent_exercise_types,
            ),
            response_schema=VoiceTurnPolicyDecision,
            system_instruction=build_voice_turn_policy_system_prompt(),
        )
        if decision.exercise_consent == "granted":
            exercise_type = (decision.exercise_type or "").strip()
            if exercise_type not in supported_exercise_ids:
                raise ValueError(
                    "Voice turn policy granted exercise consent without a "
                    "supported exercise_type."
                )
            return decision.model_copy(update={"exercise_type": exercise_type})

        return decision.model_copy(update={"exercise_type": None})
