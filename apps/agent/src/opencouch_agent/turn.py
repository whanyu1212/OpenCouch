"""The per-turn pipeline: load the session, run the agent, persist the result."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import timedelta
from typing import cast

import anyio
from ag_ui.core import BaseEvent, RunErrorEvent
from fastapi import HTTPException
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.run import AgentRunResult
from starlette.requests import Request
from starlette.responses import Response

from opencouch_agent.agent import CompanionDeps, CompanionState
from opencouch_agent.api.adapter import ServerOwnedAGUIAdapter, UnsupportedRequestError
from opencouch_agent.sessions import SessionStore

logger = logging.getLogger(__name__)

# The lease outlives the turn's time limit by this much, so a running turn's
# lease can never expire and let a second turn in.
LEASE_MARGIN = timedelta(seconds=30)

TURN_TIMEOUT_MESSAGE = "The reply took too long. Please try again."


async def run_turn(
    request: Request,
    *,
    agent: Agent[CompanionDeps, str],
    store: SessionStore,
    turn_timeout: timedelta,
) -> Response:
    """Handle one AG-UI turn with server-owned history and state.

    Only one turn runs per thread at a time. The thread is claimed before
    anything is written and released when the stream ends, however it ends.

    Args:
        request: The incoming AG-UI `RunAgentInput` request.
        agent: The companion agent.
        store: Where conversation history and state are persisted.
        turn_timeout: Hard limit on the turn, including streaming the reply.

    Returns:
        A streaming response of AG-UI events.

    Raises:
        HTTPException: 422 if the body isn't a valid AG-UI `RunAgentInput`, or
            uses client input the server rejects, such as frontend tools or a
            history not ending in a user message. 409 if another turn is
            already running on the thread.
    """
    adapter = await _parse_request(request, agent)
    thread_id = adapter.run_input.thread_id
    lease_token = uuid.uuid4().hex

    if not await store.claim_turn(thread_id, lease_token, turn_timeout + LEASE_MARGIN):
        raise HTTPException(
            status_code=409,
            detail="A reply is already in progress for this conversation.",
        )
    try:
        events = await _start_run(
            adapter, store=store, thread_id=thread_id, lease_token=lease_token
        )
    except BaseException:
        await _release_turn(store, thread_id, lease_token)
        raise

    return adapter.streaming_response(
        _guard_turn(
            events,
            store=store,
            thread_id=thread_id,
            lease_token=lease_token,
            turn_timeout=turn_timeout,
        )
    )


async def _parse_request(
    request: Request, agent: Agent[CompanionDeps, str]
) -> ServerOwnedAGUIAdapter:
    """Parse and check the request, turning client errors into 422s."""
    try:
        # `from_request` is annotated as returning the base class; it builds `cls`.
        adapter = cast(
            ServerOwnedAGUIAdapter,
            await ServerOwnedAGUIAdapter.from_request(request, agent=agent),
        )
    except ValidationError as error:
        # Leave the input out of the error: it may contain what the user wrote.
        detail = error.errors(include_input=False, include_url=False)
        raise HTTPException(status_code=422, detail=detail) from error
    try:
        adapter.check_request()
    except UnsupportedRequestError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return adapter


async def _start_run(
    adapter: ServerOwnedAGUIAdapter,
    *,
    store: SessionStore,
    thread_id: str,
    lease_token: str,
) -> AsyncIterator[BaseEvent]:
    """Load the session, record the user's message and start the agent run."""
    session = await store.load(thread_id)
    # Validate stored state before writing anything, so a bad session fails
    # without leaving a half-recorded turn behind.
    deps = CompanionDeps(state=CompanionState.model_validate(session.state))
    adapter.server_state = session.state
    # Persist the user's message before the run so it survives a failed run.
    # A later turn then sees two user messages in a row, which models handle.
    await store.append_messages(thread_id, adapter.new_user_messages())

    async def persist_run(result: AgentRunResult[str]) -> None:
        # The user's message is excluded here: it entered the run as history.
        saved = await store.save_run(
            thread_id,
            lease_token,
            result.new_messages(),
            deps.state.model_dump(mode="json"),
        )
        if not saved:
            # Another turn took over after this one's lease expired (e.g. a
            # very slow client). Its writes win; this reply is dropped.
            logger.warning(
                "Turn lost its lease; reply not saved (thread %s)", thread_id
            )

    return adapter.run_stream(
        message_history=session.history, deps=deps, on_complete=persist_run
    )


async def _guard_turn(
    events: AsyncIterator[BaseEvent],
    *,
    store: SessionStore,
    thread_id: str,
    lease_token: str,
    turn_timeout: timedelta,
) -> AsyncGenerator[BaseEvent]:
    """Enforce the turn's time limit and release its claim when the stream ends.

    The `finally` runs whether the run completes, fails, times out or the
    client disconnects.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + turn_timeout.total_seconds()
    try:
        while True:
            # Sending to a slow client happens at the `yield` below, outside the
            # timeout, so check the deadline again before each wait.
            if loop.time() >= deadline:
                yield RunErrorEvent(message=TURN_TIMEOUT_MESSAGE, code="turn_timeout")
                return
            try:
                # The timeout wraps only the wait for the next event, never a
                # `yield`, so it can't fire while the client is being sent data.
                async with asyncio.timeout_at(deadline):
                    event = await anext(events)
            except StopAsyncIteration:
                return
            except TimeoutError:
                yield RunErrorEvent(message=TURN_TIMEOUT_MESSAGE, code="turn_timeout")
                return
            yield event
    finally:
        # Shielded: a client disconnect cancels this generator, and cleanup must
        # still finish.
        with anyio.CancelScope(shield=True):
            await _close_events(events)
            await _release_turn(store, thread_id, lease_token)


async def _close_events(events: AsyncIterator[BaseEvent]) -> None:
    """Close the agent's event stream so an abandoned run stops."""
    aclose = getattr(events, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        logger.exception("Closing the agent event stream failed")


async def _release_turn(store: SessionStore, thread_id: str, lease_token: str) -> None:
    # A failed release only delays the thread until the lease expires, so log
    # it rather than masking whatever ended the turn.
    try:
        await store.release_turn(thread_id, lease_token)
    except Exception:
        logger.exception("Releasing the turn lease failed for thread %s", thread_id)
