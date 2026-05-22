"""Session-lifecycle helpers for the persistent runtime.

A *session* in this codebase is the conversation window between a user
opening a thread and the runtime declaring it ended (timeout,
explicit-end, or rotation). This package owns the runtime-side helpers
that span that window:

- ``commit``: write held memory candidates to the store at session end.
- ``finalization``: run the full end-of-session pipeline (commit +
  summarize).
- ``history``: transcript and SDK-session history boundary helpers.
- ``state``: pure-functional helpers for slicing, measuring, and
  zeroing session-relevant fields in :class:`agent.state.AgentState`.
- ``summarize``: run session summarization to produce the episodic arc.
- ``tracking``: in-process session bookkeeping (per-thread metadata
  that doesn't need durable storage).

The package is the public surface — callers should import from
``agent.runtime.session`` rather than reaching into sibling modules.
"""

from __future__ import annotations

from agent.runtime.session.commit import run_commit_session_memory
from agent.runtime.session.finalization import finalize_session_window
from agent.runtime.session.history import (
    SessionConversation,
    content_to_text,
    include_prompt_history,
    messages_from_sdk_session_items,
    messages_from_transcript,
    session_conversation_from_transcript,
    state_without_prompt_history,
    strip_recent_history_from_prompt,
)
from agent.runtime.session.state import (
    EXERCISE_STATE_FIELDS,
    active_transcript_length,
    crisis_level_from_state,
    session_continuity_clear_delta,
    slice_state_to_active_session,
    transcript_length,
    turn_count_from_state,
)
from agent.runtime.session.summarize import run_summarize_session
from agent.runtime.session.tracking import RuntimeSessionTracker

__all__ = [
    "EXERCISE_STATE_FIELDS",
    "RuntimeSessionTracker",
    "SessionConversation",
    "active_transcript_length",
    "content_to_text",
    "crisis_level_from_state",
    "finalize_session_window",
    "include_prompt_history",
    "messages_from_sdk_session_items",
    "messages_from_transcript",
    "run_commit_session_memory",
    "run_summarize_session",
    "session_conversation_from_transcript",
    "session_continuity_clear_delta",
    "slice_state_to_active_session",
    "state_without_prompt_history",
    "strip_recent_history_from_prompt",
    "transcript_length",
    "turn_count_from_state",
]
