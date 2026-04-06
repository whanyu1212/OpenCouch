import pytest
from pydantic import BaseModel

from agent.graph import build_initial_state
from agent.models import AgentInput, CrisisAssessment, Message, MessageRole
from agent.nodes.session_stage import update_session_stage
from services.llm.base import BaseLLMClient


class FakeStageResponse(BaseModel):
    stage: str
    reason: str


class FakeStageLLMClient(BaseLLMClient):
    def __init__(self, response: FakeStageResponse) -> None:
        self.response = response
        self.structured_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        raise NotImplementedError

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema,
        system_instruction: str | None = None,
        temperature: float = 0,
    ):
        self.structured_calls += 1
        return response_schema(**self.response.model_dump())


@pytest.mark.asyncio
async def test_session_stage_defaults_to_opening_for_new_session() -> None:
    state = build_initial_state(AgentInput(message="Hi, I'm new here."))

    state = await update_session_stage(state)

    assert state["session_stage"] == "opening"
    assert state["session_stage_source"] == "deterministic"


@pytest.mark.asyncio
async def test_session_stage_moves_to_closing_on_explicit_wrap_up_language() -> None:
    state = build_initial_state(
        AgentInput(
            message="Before we wrap up, can you summarize this for me?",
            history=[
                Message(role=MessageRole.USER, content="I've been overwhelmed all week."),
                Message(role=MessageRole.ASSISTANT, content="That sounds exhausting."),
            ],
        )
    )

    state = await update_session_stage(state)

    assert state["session_stage"] == "closing"
    assert "wrap-up language" in state["session_stage_reason"]


@pytest.mark.asyncio
async def test_session_stage_uses_llm_refinement_when_available() -> None:
    state = build_initial_state(
        AgentInput(
            message="That helped. What should I try this week?",
            history=[
                Message(role=MessageRole.USER, content="I want to do CBT for this negative thought."),
                Message(role=MessageRole.ASSISTANT, content="Let's work through a thought record together."),
            ],
        )
    )
    llm_client = FakeStageLLMClient(
        FakeStageResponse(
            stage="stabilizing",
            reason="The user is moving from structured work into integration and next steps.",
        )
    )

    state = await update_session_stage(state, llm_client=llm_client)

    assert state["session_stage"] == "stabilizing"
    assert state["session_stage_source"] == "llm"
    assert llm_client.structured_calls == 1


@pytest.mark.asyncio
async def test_session_stage_stays_conservative_for_safety_check() -> None:
    state = build_initial_state(AgentInput(message="I feel hopeless and trapped."))
    state["crisis"] = CrisisAssessment(
        level=1,
        confidence="medium",
        reason="Detected high-distress language without explicit self-harm signal.",
        needs_crisis_response=False,
        needs_clarification=True,
    )

    state = await update_session_stage(state)

    assert state["session_stage"] == "opening"
    assert "Safety-sensitive" in state["session_stage_reason"]
