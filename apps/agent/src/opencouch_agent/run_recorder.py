"""Saving what a turn's agent run produced, however the run ends (HY-204).

A run that succeeds, or that the turn cancels itself (on a timeout or a
client disconnect), saves Pydantic AI's own record of it, partial reply
included. Any other failure loses that record, so the fallback saves the text
the user was shown instead, marked as interrupted. Either way, a crisis turn
that fails still leaves a record of what the user was told.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from ag_ui.core import BaseEvent, TextMessageContentEvent
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.run import AgentRunResult

from opencouch_agent.agent import CompanionDeps
from opencouch_agent.sessions import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class RunRecorder:
    """Saves one run's reply and state exactly once, on whichever path it ends.

    Attributes:
        store: Where the run is saved.
        thread_id: The run's thread.
        lease_token: The turn's claim on the thread; saves are fenced by it.
        deps: The run's dependencies, whose state is saved with the reply.
    """

    store: SessionStore
    thread_id: str
    lease_token: str
    deps: CompanionDeps
    _shown: list[str] = field(default_factory=list)
    _saved: bool = False

    def observe(self, event: BaseEvent) -> None:
        """Note reply text as it is sent to the user."""
        if isinstance(event, TextMessageContentEvent):
            self._shown.append(event.delta)

    async def save_completed(self, result: AgentRunResult[str]) -> None:
        """Save a run that finished."""
        # `new_messages()` leaves out the user's message, which was saved
        # before the run.
        await self._save(result.new_messages())

    async def save_cancelled(self, cancelled: RunCancelled) -> None:
        """Save a run the turn cancelled, with its partial reply."""
        # A run cancelled before the model said anything leaves an empty
        # response, which models can reject as history.
        messages = [
            message
            for message in cancelled.new_messages()
            if not isinstance(message, ModelResponse) or message.parts
        ]
        await self._save(messages)

    async def save_shown(self) -> None:
        """Save the text the user was shown, unless the run was already saved.

        For runs that ended any other way, such as a model or tool error.
        Completed tool calls from the run are not kept: only Pydantic AI's
        record has them, and an error loses it.
        """
        if self._saved:
            return
        text = "".join(self._shown)
        # Marked as interrupted, as Pydantic AI marks a cancelled reply, so the
        # next turn's model knows the reply was cut off.
        shown = [ModelResponse(parts=[TextPart(text)], state="interrupted")]
        await self._save(shown if text else [])

    async def _save(self, messages: Sequence[ModelMessage]) -> None:
        if self._saved:
            return
        saved = await self.store.save_run(
            self.thread_id,
            self.lease_token,
            messages,
            self.deps.state.model_dump(mode="json"),
        )
        # Set only once `save_run` returns: if it raises, the fallback retries.
        self._saved = True
        if not saved:
            # Another turn took over after this one's lease expired (e.g. a
            # very slow client). Its writes win; this run is dropped.
            logger.warning(
                "Turn lost its lease; reply not saved (thread %s)", self.thread_id
            )
