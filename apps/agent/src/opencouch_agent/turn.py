"""The per-turn pipeline: load the session, run the agent, persist the result."""

from typing import cast

from fastapi import HTTPException
from pydantic_ai import Agent
from pydantic_ai.run import AgentRunResult
from starlette.requests import Request
from starlette.responses import Response

from opencouch_agent.agent import CompanionDeps, CompanionState
from opencouch_agent.api.adapter import ServerOwnedAGUIAdapter, UnsupportedRequestError
from opencouch_agent.sessions import SessionStore


async def run_turn(
    request: Request,
    *,
    agent: Agent[CompanionDeps, str],
    store: SessionStore,
) -> Response:
    """Handle one AG-UI turn with server-owned history and state.

    Args:
        request: The incoming AG-UI `RunAgentInput` request.
        agent: The companion agent.
        store: Where conversation history and state are persisted.

    Returns:
        A streaming response of AG-UI events.

    Raises:
        HTTPException: 422 if the request uses client input the server rejects,
            such as frontend tools or a history not ending in a user message.
    """
    # `from_request` is annotated as returning the base class; it builds `cls`.
    adapter = cast(
        ServerOwnedAGUIAdapter,
        await ServerOwnedAGUIAdapter.from_request(request, agent=agent),
    )
    thread_id = adapter.run_input.thread_id
    try:
        adapter.check_request()
    except UnsupportedRequestError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    user_messages = adapter.new_user_messages()

    session = await store.load(thread_id)
    # Validate stored state before writing anything, so a bad session fails
    # without leaving a half-recorded turn behind.
    deps = CompanionDeps(state=CompanionState.model_validate(session.state))
    adapter.server_state = session.state
    # Persist the user's message before the run so it survives a failed run.
    # A later turn then sees two user messages in a row, which models handle.
    await store.append_messages(thread_id, user_messages)

    async def persist_run(result: AgentRunResult[str]) -> None:
        # The user's message is excluded here: it entered the run as history.
        await store.save_run(
            thread_id, result.new_messages(), deps.state.model_dump(mode="json")
        )

    stream = adapter.run_stream(
        message_history=session.history, deps=deps, on_complete=persist_run
    )
    return adapter.streaming_response(stream)
