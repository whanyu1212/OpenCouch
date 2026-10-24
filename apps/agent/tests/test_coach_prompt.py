"""The companion runs on the interim coach prompt (HY-351)."""

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from opencouch_agent.agent import CompanionDeps, build_companion_agent


async def test_the_model_is_given_the_coach_prompt() -> None:
    seen: list[str | None] = []

    def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions)
        return ModelResponse(parts=[TextPart("ok")])

    agent = build_companion_agent(FunctionModel(reply))
    await agent.run("hi", deps=CompanionDeps())

    assert seen[0] is not None
    assert seen[0].startswith("You are the coach in OpenCouch")
    assert "## Reassurance-seeking" in seen[0]
