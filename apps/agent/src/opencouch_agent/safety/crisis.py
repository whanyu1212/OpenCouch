"""The crisis responder: what replies whenever the gate says CRISIS (HY-230).

Its agent has no tools, so no coach tool can ever run on a crisis turn. Its
reply isn't streamed: the whole reply is written, checked, then sent, because
a number split across streamed chunks can't be checked and can't be taken
back once shown. Anything that goes wrong sends a fixed, reviewed fallback.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from importlib import resources
from typing import Final, Self

from ag_ui.core import (
    BaseEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from pydantic_ai import Agent, CancellationToken, UsageLimits
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model

from opencouch_agent.agent import CompanionDeps
from opencouch_agent.safety.crisis_guidance import (
    GENERAL_INSTRUCTIONS,
    IMMINENT_INSTRUCTIONS,
    RETURN_GUIDANCE,
    TYPE_INSTRUCTIONS,
    crisis_fallback,
)
from opencouch_agent.safety.crisis_reply_check import allowed_numbers, reply_problem
from opencouch_agent.safety.crisis_resources import (
    CrisisCatalog,
    CrisisResource,
    CrisisType,
    Region,
    ordered_crisis_types,
    select_resources,
)
from opencouch_agent.safety.follow_up import CHECK_IN_GUIDANCE, RETURN_CHECK_IN

logger = logging.getLogger(__name__)


def _read(name: str) -> str:
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


CRISIS_INSTRUCTIONS: Final = _read("crisis_prompt.md")
SERVICES_HEADING: Final = "## The services you may point to"

# A cold first reply took 7-8 seconds against the real model; this leaves room
# for that, and still sends the fallback well inside the turn's own limit.
REPLY_TIMEOUT: Final = timedelta(seconds=15)
# No tools, so one model call is all a reply needs.
REPLY_LIMITS: Final = UsageLimits(request_limit=1)
# The only region served so far.
_REGION: Final[Region] = "SG"
_CHANNEL_KINDS: Final = {"whatsapp": "WhatsApp", "call": "phone", "sms": "SMS"}


@dataclass(frozen=True)
class CrisisResponder:
    """The crisis agent and the vetted resources it may point to.

    Attributes:
        agent: Writes the reply. Has no tools.
        catalog: The vetted crisis resources.
        allowed_numbers: Every catalog number, the only ones a reply may name.
    """

    agent: Agent[CompanionDeps, str]
    catalog: CrisisCatalog
    allowed_numbers: frozenset[str]

    @classmethod
    def from_catalog(
        cls, agent: Agent[CompanionDeps, str], catalog: CrisisCatalog
    ) -> Self:
        """A responder that may name any number in `catalog`."""
        return cls(agent, catalog, allowed_numbers(catalog))


def build_crisis_agent(model: Model | str) -> Agent[CompanionDeps, str]:
    """Build the crisis agent, with no tools.

    Args:
        model: A Pydantic AI model instance or model string.

    Returns:
        The crisis agent. It shares the companion's deps type so it can read
        the same conversation history.
    """
    return Agent(
        model,
        deps_type=CompanionDeps,
        instructions=CRISIS_INSTRUCTIONS,
        name="crisis_responder",
    )


def crisis_services(
    responder: CrisisResponder, *, types: frozenset[CrisisType], imminent: bool
) -> list[CrisisResource]:
    """The vetted services for this crisis turn, in the order to show them."""
    return select_resources(responder.catalog, _REGION, types, imminent=imminent)


async def crisis_reply(
    responder: CrisisResponder,
    history: Sequence[ModelMessage],
    services: Sequence[CrisisResource],
    *,
    deps: CompanionDeps,
    types: frozenset[CrisisType],
    imminent: bool,
    cancellation: CancellationToken,
    returning: bool = False,
) -> str:
    """Write a crisis reply that is safe to send, or the fallback.

    Args:
        responder: The crisis agent and catalog.
        history: The conversation, ending with the user's new message.
        services: The services the reply may point to; see `crisis_services`.
        deps: The run's dependencies.
        types: Types observed in the thread's server-owned risk state.
        imminent: Whether the danger may be immediate.
        cancellation: Stops the model call when the turn ends early.
        returning: Start with a safety check-in after silence, including on failure.

    Returns:
        The model's reply if it passes the check, otherwise the fallback.

    Raises:
        RunCancelled: If the turn was cancelled; nothing was shown, so
            nothing is sent.
    """
    # Urgent help takes precedence over a routine check-in after silence.
    check_in_first = returning and not imminent
    prefix = f"{RETURN_CHECK_IN}\n\n" if check_in_first else ""
    fallback = prefix + crisis_fallback(types, include_question=not check_in_first)
    instructions = turn_instructions(services, types=types, imminent=imminent)
    if deps.state.crisis_check_in is not None:
        instructions += f"\n\n{CHECK_IN_GUIDANCE}"
    if check_in_first:
        instructions += f"\n{RETURN_GUIDANCE}"
    try:
        async with asyncio.timeout(REPLY_TIMEOUT.total_seconds()):
            result = await responder.agent.run(
                message_history=list(history),
                deps=deps,
                instructions=instructions,
                usage_limits=REPLY_LIMITS,
                cancellation_token=cancellation,
            )
    except RunCancelled:
        raise
    except TimeoutError:
        logger.warning("Crisis reply timed out; sending the fallback")
        return fallback
    # Fail closed: any failure to write a reply still sends safe, vetted text.
    except Exception:
        logger.exception("Crisis reply failed; sending the fallback")
        return fallback
    problem = reply_problem(result.output, responder.allowed_numbers)
    if problem is not None:
        logger.warning(
            "Crisis reply failed its check (%s); sending the fallback", problem
        )
        return fallback
    return prefix + result.output


def turn_instructions(
    services: Sequence[CrisisResource], *, types: frozenset[CrisisType], imminent: bool
) -> str:
    """This turn's instructions: the services to point to, and any urgency."""
    guidance = dict.fromkeys(
        TYPE_INSTRUCTIONS.get(kind, GENERAL_INSTRUCTIONS)
        for kind in ordered_crisis_types(types)
    )
    lines = [*guidance, "", SERVICES_HEADING, ""]
    for service in services:
        hours = f" ({service.hours})" if service.hours else ""
        ways = "; ".join(
            f"{channel.display} by {_CHANNEL_KINDS[channel.kind]} ({channel.label})"
            for channel in service.channels
        )
        lines.append(f"- {service.name}{hours}. {ways}")
    text = "\n".join(lines) + "\n"
    return f"{text}\n{IMMINENT_INSTRUCTIONS}" if imminent else text


async def crisis_events(
    write_reply: Callable[[], Awaitable[str]],
    *,
    thread_id: str,
    run_id: str,
    state: Mapping[str, object],
    save_reply: Callable[[str], Awaitable[None]],
) -> AsyncGenerator[BaseEvent]:
    """The AG-UI events of a crisis turn, with its reply sent in one piece.

    Args:
        write_reply: Writes the checked reply; see `crisis_reply`.
        thread_id: The turn's thread.
        run_id: The turn's run.
        state: The shared state, sent before the reply so the resource card
            shows while the reply is being written.
        save_reply: Saves the reply exactly as it was sent.
    """
    yield RunStartedEvent(thread_id=thread_id, run_id=run_id)
    yield StateSnapshotEvent(snapshot=dict(state))
    try:
        text = await write_reply()
    except RunCancelled:
        # The turn ended before the reply was written, so nothing was shown
        # and there is nothing to send or save.
        return
    message_id = str(uuid.uuid4())
    yield TextMessageStartEvent(message_id=message_id, role="assistant")
    yield TextMessageContentEvent(message_id=message_id, delta=text)
    yield TextMessageEndEvent(message_id=message_id)
    # Saved before the run is reported finished, so a crisis turn that ends is
    # always on record with what the user was told (HY-204).
    await save_reply(text)
    yield RunFinishedEvent(thread_id=thread_id, run_id=run_id)
