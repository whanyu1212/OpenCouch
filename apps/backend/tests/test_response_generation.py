import pytest

from agent.graph import build_initial_state
from agent.models import AgentInput, CrisisAssessment, ResponseKind
from agent.nodes.crisis_response import run_crisis_response
from agent.nodes.therapeutic import run_therapeutic_response
from services.llm.base import BaseLLMClient


class FakeTextLLMClient(BaseLLMClient):
    def __init__(
        self,
        *,
        text_response: str = "Generated response",
        raise_on_text: bool = False,
    ) -> None:
        self.text_response = text_response
        self.raise_on_text = raise_on_text
        self.text_calls = 0
        self.last_prompt: str | None = None
        self.last_system_instruction: str | None = None

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        self.text_calls += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        if self.raise_on_text:
            raise RuntimeError("Simulated provider failure")
        return self.text_response

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema,
        system_instruction: str | None = None,
        temperature: float = 0,
    ):
        raise NotImplementedError("Structured generation is not used in text node tests.")


@pytest.mark.asyncio
async def test_therapeutic_node_uses_llm_for_normal_support() -> None:
    state = build_initial_state(AgentInput(message="I had a rough day and feel drained."))
    llm_client = FakeTextLLMClient(
        text_response="That sounds exhausting. It makes sense that you're feeling worn down after a day like that."
    )

    state = await run_therapeutic_response(state, llm_client=llm_client)

    assert state["response_type"] == ResponseKind.THERAPEUTIC
    assert state["response_text"].startswith("That sounds exhausting.")
    assert llm_client.text_calls == 1
    assert llm_client.last_prompt is not None
    assert "I had a rough day and feel drained." in llm_client.last_prompt


@pytest.mark.asyncio
async def test_therapeutic_node_bypasses_llm_for_safety_check() -> None:
    state = build_initial_state(AgentInput(message="I feel hopeless and trapped."))
    state["crisis"] = CrisisAssessment(
        level=1,
        confidence="medium",
        reason="Detected high-distress language without explicit self-harm signal.",
        needs_crisis_response=False,
        needs_clarification=True,
    )
    llm_client = FakeTextLLMClient()

    state = await run_therapeutic_response(state, llm_client=llm_client)

    assert state["response_type"] == ResponseKind.THERAPEUTIC
    assert "check on your safety" in state["response_text"]
    assert llm_client.text_calls == 0


@pytest.mark.asyncio
async def test_crisis_node_uses_llm_for_crisis_reply() -> None:
    state = build_initial_state(
        AgentInput(message="I've been thinking about ending it all.")
    )
    state["crisis"] = CrisisAssessment(
        level=2,
        confidence="high",
        reason="Detected clear self-harm or suicidal ideation language.",
        needs_crisis_response=True,
        needs_clarification=False,
    )
    llm_client = FakeTextLLMClient(
        text_response=(
            "I'm really glad you told me. Please reach out to a crisis hotline or "
            "someone you trust who can be with you right now."
        )
    )

    state = await run_crisis_response(state, llm_client=llm_client)

    assert state["response_type"] == ResponseKind.CRISIS
    assert "reach out to a crisis hotline" in state["response_text"]
    assert llm_client.text_calls == 1
    assert llm_client.last_prompt is not None
    assert "Detected clear self-harm or suicidal ideation language." in llm_client.last_prompt


@pytest.mark.asyncio
async def test_crisis_node_falls_back_when_llm_generation_fails() -> None:
    state = build_initial_state(
        AgentInput(message="I have pills and I am going to take them tonight.")
    )
    state["crisis"] = CrisisAssessment(
        level=3,
        confidence="high",
        reason="Detected imminent self-harm language with plan, means, or timing.",
        needs_crisis_response=True,
        needs_clarification=False,
    )
    llm_client = FakeTextLLMClient(raise_on_text=True)

    state = await run_crisis_response(state, llm_client=llm_client)

    assert state["response_type"] == ResponseKind.CRISIS
    assert "emergency services" in state["response_text"]
    assert llm_client.text_calls == 1
