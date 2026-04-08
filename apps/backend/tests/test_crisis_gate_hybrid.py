from pydantic import BaseModel
import pytest

from agent.graph import build_initial_state
from agent.models import AgentInput
from agent.nodes.crisis_gate import run_crisis_gate
from agent.nodes.therapeutic import run_therapeutic_response
from services.llm.base import BaseLLMClient


class FakeStructuredResponse(BaseModel):
    level: int
    confidence: str
    reason: str
    needs_crisis_response: bool
    needs_clarification: bool


class FakeLLMClient(BaseLLMClient):
    """Fake provider client for hybrid crisis-gate tests."""

    def __init__(
        self,
        response: FakeStructuredResponse,
        *,
        raise_on_structured: bool = False,
    ) -> None:
        self.response = response
        self.raise_on_structured = raise_on_structured
        self.structured_calls = 0
        self.last_prompt: str | None = None
        self.last_system_instruction: str | None = None

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> str:
        """Raise because text generation is unused in crisis-gate tests.

        Args:
            prompt: User/task prompt sent to the fake provider.
            system_instruction: Optional system prompt sent to the fake provider.
            temperature: Sampling temperature for generation.

        Returns:
            This function does not return a value.

        Raises:
            NotImplementedError: Always raised for this fake client path.
        """

        raise NotImplementedError("Text generation is not used in crisis gate tests.")

    async def generate_text_stream(
        self, *, prompt, system_instruction=None, temperature=0
    ):
        yield ""

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> BaseModel:
        """Return the configured structured crisis response or raise a fake failure.

        Args:
            prompt: User/task prompt sent to the fake provider.
            response_schema: Structured schema requested by the caller.
            system_instruction: Optional system prompt sent to the fake provider.
            temperature: Sampling temperature for generation.

        Returns:
            A schema instance populated from the configured fake response.

        Raises:
            RuntimeError: Raised when the fake client is configured to fail.
        """

        self.structured_calls += 1
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        if self.raise_on_structured:
            raise RuntimeError("Simulated structured generation failure")
        return response_schema(**self.response.model_dump())


@pytest.mark.asyncio
async def test_hybrid_gate_uses_llm_for_non_override_case() -> None:
    """Hybrid gate should call the provider when no override applies."""

    state = build_initial_state(AgentInput(message="I just wish I could disappear."))
    llm_client = FakeLLMClient(
        FakeStructuredResponse(
            level=2,
            confidence="medium",
            reason="Passive ideation judged conservatively.",
            needs_crisis_response=True,
            needs_clarification=True,
        )
    )

    state = await run_crisis_gate(state, llm_client=llm_client)

    assert llm_client.structured_calls == 1
    assert state["crisis"].level == 2
    assert state["route"] == "crisis"


@pytest.mark.asyncio
async def test_hybrid_gate_bypasses_llm_for_imminent_override() -> None:
    """Imminent-risk overrides should bypass provider classification."""

    state = build_initial_state(
        AgentInput(message="I have a plan to kill myself tonight.")
    )
    llm_client = FakeLLMClient(
        FakeStructuredResponse(
            level=0,
            confidence="low",
            reason="Should never be used.",
            needs_crisis_response=False,
            needs_clarification=False,
        )
    )

    state = await run_crisis_gate(state, llm_client=llm_client)

    assert llm_client.structured_calls == 0
    assert state["crisis"].level == 3
    assert state["route"] == "crisis"


@pytest.mark.asyncio
async def test_hybrid_gate_bypasses_llm_for_idiomatic_safe_override() -> None:
    """Idiomatic-safe overrides should bypass provider classification."""

    state = build_initial_state(AgentInput(message="Work is killing me this week."))
    llm_client = FakeLLMClient(
        FakeStructuredResponse(
            level=2,
            confidence="high",
            reason="Should never be used.",
            needs_crisis_response=True,
            needs_clarification=False,
        )
    )

    state = await run_crisis_gate(state, llm_client=llm_client)

    assert llm_client.structured_calls == 0
    assert state["crisis"].level == 0
    assert state["route"] == "therapeutic"


@pytest.mark.asyncio
async def test_hybrid_gate_normalizes_invalid_llm_fields() -> None:
    """Malformed provider output should be normalized before use."""

    state = build_initial_state(
        AgentInput(message="This is concerning but not explicit.")
    )
    llm_client = FakeLLMClient(
        FakeStructuredResponse(
            level=9,
            confidence="unknown",
            reason="Malformed upstream output.",
            needs_crisis_response=False,
            needs_clarification=True,
        )
    )

    state = await run_crisis_gate(state, llm_client=llm_client)

    assert state["crisis"].level == 3
    assert state["crisis"].confidence == "medium"
    assert state["crisis"].needs_crisis_response is True


@pytest.mark.asyncio
async def test_hybrid_gate_passes_history_into_llm_prompt() -> None:
    """Hybrid classifier prompts should include recent history."""

    state = build_initial_state(
        AgentInput(
            message="I keep thinking about it.",
            history=[{"role": "user", "content": "Sometimes I want to kill myself."}],
        )
    )
    llm_client = FakeLLMClient(
        FakeStructuredResponse(
            level=2,
            confidence="high",
            reason="History indicates escalating ideation.",
            needs_crisis_response=True,
            needs_clarification=False,
        )
    )

    await run_crisis_gate(state, llm_client=llm_client)

    assert llm_client.last_prompt is not None
    assert "Sometimes I want to kill myself." in llm_client.last_prompt
    assert "I keep thinking about it." in llm_client.last_prompt


@pytest.mark.asyncio
async def test_safety_check_from_llm_routes_to_semi_dynamic_template() -> None:
    """LLM-triggered safety checks should use the bounded safety template."""

    state = build_initial_state(
        AgentInput(message="I feel completely hopeless and trapped.")
    )
    llm_client = FakeLLMClient(
        FakeStructuredResponse(
            level=1,
            confidence="medium",
            reason="Detected high-distress language without explicit self-harm signal.",
            needs_crisis_response=False,
            needs_clarification=True,
        )
    )

    state = await run_crisis_gate(state, llm_client=llm_client)
    state = await run_therapeutic_response(state)

    assert state["route"] == "therapeutic"
    assert state["crisis"].needs_clarification is True
    assert "check on your safety" in state["response_text"]


@pytest.mark.asyncio
async def test_hybrid_gate_falls_back_to_deterministic_when_llm_fails() -> None:
    """Hybrid gate should fall back to deterministic logic on provider failure."""

    state = build_initial_state(
        AgentInput(message="I feel completely hopeless and trapped.")
    )
    llm_client = FakeLLMClient(
        FakeStructuredResponse(
            level=0,
            confidence="low",
            reason="Should never be used.",
            needs_crisis_response=False,
            needs_clarification=False,
        ),
        raise_on_structured=True,
    )

    state = await run_crisis_gate(state, llm_client=llm_client)

    assert llm_client.structured_calls == 1
    assert state["crisis"].level == 1
    assert state["crisis"].needs_clarification is True
    assert state["route"] == "therapeutic"
