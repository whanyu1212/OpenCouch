"""AG-UI adapter that keeps conversation history and state on the server."""

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import DeferredToolResults
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.ui.ag_ui import AGUIAdapter

from opencouch_agent.agent import CompanionDeps


class UnsupportedRequestError(ValueError):
    """The AG-UI request uses input the server doesn't accept."""


@dataclass
class ServerOwnedAGUIAdapter(AGUIAdapter[CompanionDeps, str]):
    """AG-UI adapter where the server, not the client, owns the conversation.

    The stock adapter trusts several client inputs: it appends every
    client-sent message to the history it is given, replaces the run's state
    with client-sent state, gives the model any tools the client declares, and
    resumes deferred tool calls the client names. Here the client contributes
    only its newest user message. History and state come from the session
    store, and client tools and resumes are rejected until the server supports
    them deliberately.

    Attributes:
        server_state: Shared state loaded from the session store. Set it before
            the run starts.
    """

    server_state: dict[str, Any] = field(default_factory=dict)

    def check_request(self) -> None:
        """Reject requests that rely on client input the server doesn't trust.

        Raises:
            UnsupportedRequestError: If the last message isn't from the user, or
                the request declares frontend tools or resumes tool calls.
        """
        client_messages = self.run_input.messages
        if not client_messages or client_messages[-1].role != "user":
            raise UnsupportedRequestError(
                "Expected the last message to be from the user."
            )
        # Tool descriptions reach the model unsanitized, so client-declared
        # tools are a prompt-injection path.
        if self.run_input.tools:
            raise UnsupportedRequestError("Frontend tools are not supported yet.")
        if getattr(self.run_input, "resume", None):
            raise UnsupportedRequestError("Resuming tool calls is not supported yet.")

    @property
    def messages(self) -> list[ModelMessage]:
        """The newest user message from the request, ignoring the rest.

        Call `check_request` first; this assumes the last message is the user's.
        """
        return self.load_messages(self.run_input.messages[-1:])

    @property
    def state(self) -> dict[str, Any]:
        """The server-owned state.

        Client-sent state is untrusted user input and is ignored.
        """
        return self.server_state

    @property
    def toolset(self) -> AbstractToolset[CompanionDeps] | None:
        """No frontend tools; see `check_request`."""
        return None

    @property
    def deferred_tool_results(self) -> DeferredToolResults | None:
        """No client-resumed tool calls; see `check_request`."""
        return None

    def new_user_messages(self) -> list[ModelMessage]:
        """The request's new user message, sanitized and ready to persist."""
        return self.sanitize_messages(self.messages)
