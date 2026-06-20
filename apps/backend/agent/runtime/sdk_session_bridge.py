"""Bridge between app-owned state and the OpenAI Agents SDK text session.

Extracted from :class:`~agent.runtime.runtime.PersistentAgentRuntime` to shrink
the runtime module and isolate the two-way synchronization between the app's
durable ``AgentState`` transcript and the SDK-owned, model-visible episodic
session history.

The bridge owns three concerns:

- the lazily-constructed serving :class:`OpenAITextRuntime` singleton,
- recovering an empty SDK session from app transcript state (read-in), and
- recording a finalized turn back into the SDK session (write-back).

The SDK session owns model-visible conversation history. ``AgentState.transcript``
is app-visible state for UI/API/audit use, and only crosses back into the SDK
session when that session is empty (e.g. migration or local session-db loss).
"""

from __future__ import annotations

from typing import Any

from agent.models import Message
from agent.runtime.openai_text_runtime import OpenAITextRuntime
from agent.runtime.session.history import messages_from_transcript
from agent.runtime.thread_state_reader import merge_history_response_styles
from agent.state import AgentState


class SdkSessionBridge:
    """Synchronizes app transcript state with the OpenAI SDK text session.

    Holds the serving :class:`OpenAITextRuntime` singleton and the session
    seed/record helpers. A single instance is owned by the runtime for its
    lifetime; the class is not thread-safe.
    """

    def __init__(self, *, text_session_store: Any | None) -> None:
        """Initialize the bridge.

        Args:
            text_session_store: The SDK-backed text session store, or ``None``
                when SDK session persistence is disabled.

        Returns:
            None: Stores the session store and prepares the lazy runtime slot.
        """

        self._text_session_store = text_session_store
        self._openai_text_runtime: OpenAITextRuntime | None = None

    def get_text_runtime(self) -> OpenAITextRuntime:
        """Return the serving OpenAI Agents SDK text runtime.

        The runtime is constructed lazily on first use and reused thereafter, so
        repeated calls return the same instance.

        Returns:
            OpenAITextRuntime: The shared serving text runtime.
        """

        if self._openai_text_runtime is None:
            self._openai_text_runtime = OpenAITextRuntime()
        return self._openai_text_runtime

    async def session_for_thread(
        self,
        thread_id: str,
        *,
        current_user_message: str,
        prior_state: AgentState | None,
    ) -> Any | None:
        """Return the SDK session for OpenAI serving turns when enabled.

        Args:
            thread_id: The thread identifier.
            current_user_message: The user message starting the turn.
            prior_state: The thread's prior persisted state, used to bootstrap an
                empty SDK session.

        Returns:
            The SDK turn session, or ``None`` when SDK persistence is disabled.
        """

        if self._text_session_store is None:
            return None
        await self.recover_empty_session_from_state(thread_id, prior_state)
        return self._text_session_store.turn_session_for_thread(
            thread_id,
            current_user_message=current_user_message,
        )

    async def recover_empty_session_from_state(
        self,
        thread_id: str,
        prior_state: AgentState | None,
    ) -> bool:
        """Recover an empty SDK session from app-visible transcript state.

        The SDK session owns model-visible episodic conversation history.
        ``AgentState.transcript`` is app-visible state for UI/API/audit use, and
        only crosses back into the SDK session when that session is empty.

        Args:
            thread_id: The thread identifier.
            prior_state: The thread's prior persisted state.

        Returns:
            ``True`` when the SDK session was seeded from transcript state.
        """

        if self._text_session_store is None or prior_state is None:
            return False
        messages = messages_from_transcript(prior_state.get("transcript", []))
        if not messages:
            return False
        return await self._text_session_store.seed_thread_from_messages(
            thread_id,
            messages,
        )

    async def ensure_turn_recorded(
        self,
        thread_id: str,
        *,
        user_message: str,
        final_state: AgentState,
    ) -> None:
        """Ensure SDK history contains the finalized OpenAI user/assistant turn.

        Args:
            thread_id: The thread identifier.
            user_message: The user message for the turn.
            final_state: The finalized turn state carrying the response text.

        Returns:
            None: Records the turn into the SDK session when enabled.
        """

        if self._text_session_store is None:
            return
        response_text = str(final_state.get("response_text") or "").strip()
        await self._text_session_store.ensure_turn_recorded(
            thread_id,
            user_message=user_message,
            assistant_message=response_text,
        )

    async def history_for_final_state(
        self,
        thread_id: str,
        final_state: AgentState,
    ) -> list[Message]:
        """Return public history without calling the public get_history method.

        Args:
            thread_id: The thread identifier.
            final_state: The finalized turn state.

        Returns:
            list[Message]: Materialized history, preferring SDK session history
            enriched with transcript response styles, else the transcript.
        """

        if self._text_session_store is not None:
            history = await self._text_session_store.get_history(thread_id)
            if history:
                return merge_history_response_styles(history, final_state)
        return messages_from_transcript(final_state.get("transcript", []))


__all__ = ["SdkSessionBridge"]
