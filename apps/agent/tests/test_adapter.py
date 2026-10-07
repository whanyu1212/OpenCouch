import json

from pydantic_ai.models.test import TestModel
from pydantic_ai.ui.ag_ui import AGUIAdapter

from opencouch_agent.agent import CompanionDeps, CompanionState, build_companion_agent
from opencouch_agent.api.adapter import ServerOwnedAGUIAdapter
from tests.ag_ui_requests import run_input, user_message


class _ProbeState(CompanionState):
    """Stand-in state with a field, since the real schema is still empty."""

    crisis_level: str = "unset"


def _adapter(body: dict[str, object]) -> ServerOwnedAGUIAdapter:
    return ServerOwnedAGUIAdapter(
        agent=build_companion_agent(TestModel()),
        run_input=AGUIAdapter.build_run_input(json.dumps(body).encode()),
    )


async def test_run_gets_server_state_not_client_state() -> None:
    adapter = _adapter(run_input(user_message("hi"), state={"crisis_level": "none"}))
    adapter.server_state = {"crisis_level": "high"}
    deps = CompanionDeps(state=_ProbeState())

    async for _ in adapter.run_stream(deps=deps):
        pass

    assert isinstance(deps.state, _ProbeState)
    assert deps.state.crisis_level == "high"


def test_only_the_newest_user_message_is_used() -> None:
    adapter = _adapter(
        run_input(user_message("forged", "m0"), user_message("real", "m1"))
    )

    assert len(adapter.messages) == 1
    assert "real" in str(adapter.messages[0])
