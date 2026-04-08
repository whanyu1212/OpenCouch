import pytest

from agent.graph import build_initial_state
from agent.models import AgentInput, CrisisAssessment, ModeType, ResponseKind
from agent.nodes.crisis_response import run_crisis_response
from agent.nodes.guided_exercise import run_guided_exercise_response
from agent.nodes.psychoeducation import run_psychoeducation_response
from agent.nodes.therapeutic import run_therapeutic_response
from agent.prompts.builders import build_therapeutic_response_prompt
from services.llm.base import BaseLLMClient


class FakeTextLLMClient(BaseLLMClient):
    """Fake provider client for response-generation tests."""

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
        """Return the configured text response or raise a fake failure.

        Args:
            prompt: User/task prompt sent to the fake provider.
            system_instruction: Optional system prompt sent to the fake provider.
            temperature: Sampling temperature for generation.

        Returns:
            The configured fake text response.

        Raises:
            RuntimeError: Raised when the fake client is configured to fail.
        """

        self.text_calls += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        if self.raise_on_text:
            raise RuntimeError("Simulated provider failure")
        return self.text_response

    async def generate_text_stream(
        self, *, prompt, system_instruction=None, temperature=0
    ):
        yield await self.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema,
        system_instruction: str | None = None,
        temperature: float = 0,
    ):
        """Raise because structured generation is unused in these tests.

        Args:
            prompt: User/task prompt sent to the fake provider.
            response_schema: Structured schema requested by the caller.
            system_instruction: Optional system prompt sent to the fake provider.
            temperature: Sampling temperature for generation.

        Returns:
            This function does not return a value.

        Raises:
            NotImplementedError: Always raised for this fake client path.
        """

        raise NotImplementedError(
            "Structured generation is not used in text node tests."
        )


@pytest.mark.asyncio
async def test_therapeutic_node_uses_llm_for_normal_support() -> None:
    """Therapeutic node should use the provider for ordinary support replies."""

    state = build_initial_state(
        AgentInput(message="I had a rough day and feel drained.")
    )
    llm_client = FakeTextLLMClient(
        text_response="That sounds exhausting. It makes sense that you're feeling worn down after a day like that."
    )

    state = await run_therapeutic_response(state, llm_client=llm_client)

    assert state["response_type"] == ResponseKind.THERAPEUTIC
    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert state["response_text"].startswith("That sounds exhausting.")
    assert llm_client.text_calls == 1
    assert llm_client.last_prompt is not None
    assert "I had a rough day and feel drained." in llm_client.last_prompt


@pytest.mark.asyncio
async def test_therapeutic_node_bypasses_llm_for_safety_check() -> None:
    """Safety-check replies should bypass the provider and stay bounded."""

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
    assert state["mode_type"] == ModeType.OPERATIONAL
    assert "check on your safety" in state["response_text"]
    assert llm_client.text_calls == 0


@pytest.mark.asyncio
async def test_therapeutic_node_holds_space_for_venting_intent() -> None:
    """Venting intent should suppress fallback problem-solving behavior."""

    state = build_initial_state(
        AgentInput(message="I just want to vent. I don't want advice right now.")
    )

    state = await run_therapeutic_response(state)

    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "space to say it" in state["response_text"].lower()
    assert "tell me a bit more" not in state["response_text"].lower()


def test_therapeutic_prompt_includes_response_guidance_block() -> None:
    """Therapeutic prompts should include the compiled response guidance."""

    state = build_initial_state(
        AgentInput(message="I just want to vent. I do not want advice right now.")
    )
    state["response_guidance"] = (
        "User is venting or explicitly does not want advice. Hold space."
    )

    prompt = build_therapeutic_response_prompt(state)

    assert "Turn-specific guidance:" in prompt
    assert "Hold space" in prompt


@pytest.mark.asyncio
async def test_therapeutic_node_uses_strengths_based_fallback_for_progress_updates() -> (
    None
):
    """Progress updates should trigger a strengths-based fallback shape."""

    state = build_initial_state(
        AgentInput(message="I actually handled it better this time and stayed calmer.")
    )

    state = await run_therapeutic_response(state)

    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "went differently this time" in state["response_text"].lower()
    assert "capacity" in state["response_text"].lower()


@pytest.mark.asyncio
async def test_therapeutic_node_uses_supportive_boundary_fallback_for_pet_task() -> (
    None
):
    """Practical pet-medication tasks should not get procedural fallback advice."""

    state = build_initial_state(
        AgentInput(
            message="My cat is unwell and I'm having trouble giving it medication.",
        )
    )

    state = await run_therapeutic_response(state)

    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "practical animal-care steps" in state["response_text"].lower()
    assert "bathroom" not in state["response_text"].lower()
    assert "towel" not in state["response_text"].lower()


@pytest.mark.asyncio
async def test_guided_exercise_node_uses_behavioral_activation_fallback_for_stuckness() -> (
    None
):
    """Stuck and avoidant language should trigger behavioral-activation fallback."""

    state = build_initial_state(
        AgentInput(message="I feel stuck, drained, and keep avoiding everything.")
    )
    state["active_modalities"] = ["cbt"]

    state = await run_guided_exercise_response(state)

    assert "5 to 10 minutes" in state["response_text"]
    assert "not to fix everything" in state["response_text"].lower()


@pytest.mark.asyncio
async def test_psychoeducation_node_uses_fallback_explanation_for_anxiety() -> None:
    """Psychoeducation fallback should explain anxiety without diagnosing."""

    state = build_initial_state(
        AgentInput(
            message="Can you explain why my body reacts like this when I get anxious?"
        )
    )

    state = await run_psychoeducation_response(state)

    assert state["mode"] == "psychoeducation"
    assert state["mode_type"] == ModeType.THERAPEUTIC
    assert "protect you" in state["response_text"].lower()


@pytest.mark.asyncio
async def test_crisis_node_uses_llm_for_crisis_reply() -> None:
    """Crisis node should use the provider when crisis generation is available."""

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
    assert state["mode"] == "crisis_response"
    assert state["mode_type"] == ModeType.CRISIS
    assert "reach out to a crisis hotline" in state["response_text"]
    assert llm_client.text_calls == 1
    assert llm_client.last_prompt is not None
    assert (
        "Detected clear self-harm or suicidal ideation language."
        in llm_client.last_prompt
    )


@pytest.mark.asyncio
async def test_crisis_node_falls_back_when_llm_generation_fails() -> None:
    """Crisis node should fall back cleanly when provider generation fails."""

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
    assert state["mode"] == "crisis_response"
    assert state["mode_type"] == ModeType.CRISIS
    assert "emergency services" in state["response_text"]
    assert llm_client.text_calls == 1
