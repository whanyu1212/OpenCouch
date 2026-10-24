"""The per-turn pipeline: load, screen, route, run the agent, persist."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import cast

import anyio
from ag_ui.core import BaseEvent, RunErrorEvent
from fastapi import HTTPException
from pydantic import ValidationError
from pydantic_ai import Agent, CancellationToken
from pydantic_ai.messages import ModelMessage
from starlette.requests import Request
from starlette.responses import Response

from opencouch_agent.agent import CompanionDeps, CompanionState
from opencouch_agent.api.adapter import ServerOwnedAGUIAdapter, UnsupportedRequestError
from opencouch_agent.run_recorder import RunRecorder
from opencouch_agent.safety import JevClient, conversation_turns, run_gate
from opencouch_agent.safety.crisis import (
    CrisisResponder,
    crisis_events,
    crisis_reply,
    crisis_services,
)
from opencouch_agent.sessions import Session, SessionStore

logger = logging.getLogger(__name__)

# The lease outlives the turn's time limit by this much, so a running turn's
# lease can never expire and let a second turn in.
LEASE_MARGIN = timedelta(seconds=30)
# How long a cancelled run gets to stop and hand over its partial reply. Well
# inside LEASE_MARGIN, so the save still holds the lease.
CANCEL_GRACE_SECONDS = 5.0

TURN_TIMEOUT_MESSAGE = "The reply took too long. Please try again."
TURN_IN_PROGRESS_MESSAGE = "A reply is already in progress for this conversation."
# The client sends only its newest message, so this leaves room for one at
# MAX_MESSAGE_CHARS plus the rest of the AG-UI request.
MAX_REQUEST_BYTES = 64_000
# Longer messages add Jev cost and delay, and push the rest of the
# conversation out of what Jev reads.
MAX_MESSAGE_CHARS = 4000


@dataclass(frozen=True)
class TurnAgents:
    """The agents a turn can be routed to.

    Attributes:
        companion: Replies while the conversation is below CRISIS.
        crisis: Replies at CRISIS, or when screening fails. Has no tools.
    """

    companion: Agent[CompanionDeps, str]
    crisis: CrisisResponder


@dataclass(frozen=True)
class _UserTurn:
    thread_id: str
    lease_token: str
    messages: list[ModelMessage]
    text: str


@dataclass(frozen=True)
class _ActiveRun:
    events: AsyncIterator[BaseEvent]
    cancellation: CancellationToken
    recorder: RunRecorder


async def run_turn(
    request: Request,
    *,
    agents: TurnAgents,
    jev: JevClient,
    store: SessionStore,
    turn_timeout: timedelta,
) -> Response:
    """Handle one AG-UI turn with server-owned history, state and risk.

    Only one turn runs per thread at a time. The thread is claimed before
    anything is written and released when the stream ends, however it ends.
    Every turn is screened before any agent runs.

    Args:
        request: The incoming AG-UI `RunAgentInput` request.
        agents: The companion and the crisis responder.
        jev: The Jev client for the safety gate.
        store: Where conversation history, state and risk are persisted.
        turn_timeout: Hard limit on the agent's reply, including streaming
            it. Screening runs before it, under its own deadline.

    Returns:
        A streaming response of AG-UI events.

    Raises:
        HTTPException: 422 if the body isn't a valid AG-UI `RunAgentInput`, or
            uses client input the server rejects, such as frontend tools, a
            history not ending in a user message, or a message with no text or
            over `MAX_MESSAGE_CHARS`. 413 if the body is over the app's cap
            (`MAX_REQUEST_BYTES`, enforced by `BodySizeLimit`). 409 if another
            turn is already running on the thread.
    """
    adapter = await _parse_request(request, agents.companion)
    user_messages = adapter.new_user_messages()
    turn = _UserTurn(
        thread_id=adapter.run_input.thread_id,
        lease_token=uuid.uuid4().hex,
        messages=user_messages,
        text=_user_text(user_messages),
    )

    lease = turn_timeout + LEASE_MARGIN
    if not await store.claim_turn(turn.thread_id, turn.lease_token, lease):
        raise HTTPException(status_code=409, detail=TURN_IN_PROGRESS_MESSAGE)
    try:
        routed, run = await _start_run(
            adapter, turn, agents=agents, jev=jev, store=store
        )
    except BaseException:
        await _release_turn(store, turn.thread_id, turn.lease_token)
        raise

    return routed.streaming_response(
        _guard_turn(run, store=store, turn=turn, turn_timeout=turn_timeout)
    )


async def _parse_request(
    request: Request, agent: Agent[CompanionDeps, str]
) -> ServerOwnedAGUIAdapter:
    """Parse and check the request, turning client errors into 4xx responses."""
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


def _user_text(messages: list[ModelMessage]) -> str:
    """The text of the user's new message, which the gate screens."""
    text = "\n".join(turn.text for turn in conversation_turns(messages))
    if not text.strip():
        # Safety: empty text would screen as calm, so a message with nothing
        # to screen (an image alone, say) is refused rather than let through.
        raise HTTPException(
            status_code=422, detail="Expected the message to contain text."
        )
    if len(text) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Expected a message of at most {MAX_MESSAGE_CHARS} characters.",
        )
    return text


async def _start_run(
    adapter: ServerOwnedAGUIAdapter,
    turn: _UserTurn,
    *,
    agents: TurnAgents,
    jev: JevClient,
    store: SessionStore,
) -> tuple[ServerOwnedAGUIAdapter, _ActiveRun]:
    """Load the session, run the gate, record the turn and start the chosen agent."""
    session = await store.load(turn.thread_id)
    # Validate stored state before writing anything, so a bad session fails
    # without leaving a half-recorded turn behind.
    deps = CompanionDeps(state=CompanionState.model_validate(session.state))
    decision = await run_gate(
        jev, session.risk, session.history, turn.text, at=datetime.now(UTC)
    )
    # Safety: persist the user's message and the new risk before any agent
    # runs, so a failed run can't lose an escalation. A later turn then sees
    # two user messages in a row, which models handle.
    recorded = await store.record_user_turn(
        turn.thread_id, turn.lease_token, turn.messages, decision.risk
    )
    if not recorded:
        # Another turn took over after this one's lease expired.
        raise HTTPException(status_code=409, detail=TURN_IN_PROGRESS_MESSAGE)
    if decision.risk.level is not session.risk.level:
        logger.info(
            "Risk level %s -> %s (thread %s)",
            session.risk.level,
            decision.risk.level,
            turn.thread_id,
        )

    # The resource card belongs to crisis turns, so every other turn clears it.
    deps.state.crisis_resources = []
    recorder = RunRecorder(store, turn.thread_id, turn.lease_token, deps)
    if decision.path == "crisis":
        return _start_crisis_reply(
            adapter, turn, agents.crisis, session, recorder, imminent=decision.imminent
        )
    return _start_companion(adapter, agents.companion, session.history, recorder)


def _start_companion(
    adapter: ServerOwnedAGUIAdapter,
    agent: Agent[CompanionDeps, str],
    history: list[ModelMessage],
    recorder: RunRecorder,
) -> tuple[ServerOwnedAGUIAdapter, _ActiveRun]:
    """Start the companion, streaming its reply."""
    state = recorder.deps.state.model_dump(mode="json")
    routed = replace(adapter, agent=agent, server_state=state)
    cancellation = CancellationToken()
    events = routed.run_stream(
        message_history=history,
        deps=recorder.deps,
        cancellation_token=cancellation,
        on_complete=recorder.save_completed,
        on_cancel=recorder.save_cancelled,
    )
    return routed, _ActiveRun(events, cancellation, recorder)


def _start_crisis_reply(
    adapter: ServerOwnedAGUIAdapter,
    turn: _UserTurn,
    responder: CrisisResponder,
    session: Session,
    recorder: RunRecorder,
    *,
    imminent: bool,
) -> tuple[ServerOwnedAGUIAdapter, _ActiveRun]:
    """Start the crisis responder, whose reply is checked and sent whole."""
    services = crisis_services(responder, imminent=imminent)
    recorder.deps.state.crisis_resources = [service.key for service in services]
    state = recorder.deps.state.model_dump(mode="json")
    routed = replace(adapter, agent=responder.agent, server_state=state)
    cancellation = CancellationToken()
    write_reply = partial(
        crisis_reply,
        responder,
        [*session.history, *turn.messages],
        services,
        deps=recorder.deps,
        imminent=imminent,
        cancellation=cancellation,
    )
    events = crisis_events(
        write_reply,
        thread_id=turn.thread_id,
        run_id=adapter.run_input.run_id,
        state=state,
        save_reply=recorder.save_reply,
    )
    return routed, _ActiveRun(events, cancellation, recorder)


async def _guard_turn(
    run: _ActiveRun,
    *,
    store: SessionStore,
    turn: _UserTurn,
    turn_timeout: timedelta,
) -> AsyncGenerator[BaseEvent]:
    """Enforce the turn's time limit, save the run and release the turn's claim.

    The `finally` runs whether the run completes, fails, times out or the
    client disconnects, and saves the run before releasing the claim, since
    saving needs it.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + turn_timeout.total_seconds()
    # The run's events are read in one task of their own, so the deadline can
    # end the wait without cancelling the run from outside, which would lose
    # its partial reply. The client gets its error on time, and the run is
    # stopped through its token afterwards.
    queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
    pump = asyncio.create_task(_pump(run.events, queue))
    try:
        while True:
            # Sending to a slow client happens at the `yield` below, outside the
            # timeout, so check the deadline again before each wait.
            if loop.time() >= deadline:
                yield RunErrorEvent(message=TURN_TIMEOUT_MESSAGE, code="turn_timeout")
                return
            try:
                async with asyncio.timeout_at(deadline):
                    event = await queue.get()
            except TimeoutError:
                yield RunErrorEvent(message=TURN_TIMEOUT_MESSAGE, code="turn_timeout")
                return
            if event is None:
                return
            run.recorder.observe(event)
            yield event
    finally:
        # Shielded: a client disconnect cancels this generator, and cleanup must
        # still finish.
        with anyio.CancelScope(shield=True):
            await _stop_run(run, pump)
            await _save_shown(run.recorder, turn.thread_id)
            await _release_turn(store, turn.thread_id, turn.lease_token)


async def _pump(
    events: AsyncIterator[BaseEvent], queue: asyncio.Queue[BaseEvent | None]
) -> None:
    """Read every event of the run into `queue`, then `None`."""
    try:
        async for event in events:
            queue.put_nowait(event)
    finally:
        queue.put_nowait(None)


async def _stop_run(run: _ActiveRun, pump: asyncio.Task[None]) -> None:
    """Stop an unfinished run so it saves its partial reply, then close it.

    Cancelling through the token, rather than cancelling the task reading the
    run, is what makes Pydantic AI hand over the partial reply. The events
    still to come are read but not sent: the client has gone or already has
    its error. A run that already ended is left as it is.
    """
    run.cancellation.cancel()
    try:
        # A run that ignores the token is cancelled from outside once the
        # grace period is over; what was shown is then saved instead.
        async with asyncio.timeout(CANCEL_GRACE_SECONDS):
            await pump
    except TimeoutError:
        logger.warning("Cancelled run didn't stop in time; saving what was shown")
    except Exception:
        logger.exception("Reading the agent run failed")
    await _close_events(run.events)


async def _save_shown(recorder: RunRecorder, thread_id: str) -> None:
    # Logged, not raised: the claim must still be released, and the original
    # failure is what ended the turn.
    try:
        await recorder.save_shown()
    except Exception:
        logger.exception("Saving the interrupted reply failed for thread %s", thread_id)


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
