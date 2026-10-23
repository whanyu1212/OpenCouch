from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from opencouch_agent.agent import CompanionDeps
from opencouch_agent.safety.crisis import build_crisis_agent


async def test_the_crisis_responder_offers_the_model_no_tools() -> None:
    seen: list[AgentInfo] = []

    def reply(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart("Are you safe right now?")])

    agent = build_crisis_agent(FunctionModel(reply))
    await agent.run("i can't do this anymore", deps=CompanionDeps())

    assert seen[0].function_tools == []
    assert seen[0].output_tools == []
