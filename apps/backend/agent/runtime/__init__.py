"""Runtime support types and helpers for persistent agent sessions.

The runtime package exports convenience symbols lazily so submodules such as
``agent.runtime.agents`` and ``agent.runtime.tools`` can be imported without
eagerly importing the full persistent runtime.
"""

from __future__ import annotations

from typing import Any

_RUNTIME_EXPORTS = {
    "ALLOWED_MSGPACK_MODULES",
    "DEFAULT_CRISIS_LOG_DB_PATH",
    "DEFAULT_FEEDBACK_DB_PATH",
    "DEFAULT_MEMORY_DB_PATH",
    "DEFAULT_TEXT_SESSION_DB_PATH",
    "DEFAULT_THREAD_DB_PATH",
    "SESSION_TIMEOUT",
    "PersistentAgentRuntime",
    "_iso_now",
}

_ACTIVE_SESSION_EXPORTS = {
    "PersistedActiveSessionState",
    "PostgresActiveSessionStore",
    "SqliteActiveSessionStore",
}

_SESSION_STORE_EXPORTS = {
    "TextSessionBackend",
    "TextSessionStore",
    "TextSessionStoreConfig",
    "create_text_session_store",
}

_TEXT_EXPORTS = {
    "OpenAITextRuntime",
}

_TURN_EXPORTS = {
    "build_initial_state",
    "run_agent",
    "state_to_output",
}

_TYPES_EXPORTS = {
    "ActiveSessionExists",
    "ExpectedSessionLiveness",
    "PersistentTurnResult",
    "SessionInterrupted",
    "SessionLeaseExpired",
    "SessionStatus",
    "TextRuntimeChunkEvent",
    "TextRuntimeConfig",
    "TextRuntimeShadowResult",
    "TextRuntimeShadowStatus",
    "TextRuntimeStateEvent",
    "TextRuntimeStatusEvent",
    "TextRuntimeStreamEvent",
    "ThreadSummary",
}


def __getattr__(name: str) -> Any:
    """Resolve public runtime exports on first access."""

    if name in _RUNTIME_EXPORTS:
        from agent.runtime import runtime as _runtime

        return getattr(_runtime, name)
    if name in _ACTIVE_SESSION_EXPORTS:
        from agent.runtime import active_session as _active_session

        return getattr(_active_session, name)
    if name in _SESSION_STORE_EXPORTS:
        from agent.runtime import session_store as _session_store

        return getattr(_session_store, name)
    if name in _TEXT_EXPORTS:
        from agent.runtime import text as _text

        return getattr(_text, name)
    if name in _TURN_EXPORTS:
        from agent.runtime import turn as _turn

        return getattr(_turn, name)
    if name in _TYPES_EXPORTS:
        from agent.runtime import types as _types

        return getattr(_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALLOWED_MSGPACK_MODULES",
    "DEFAULT_CRISIS_LOG_DB_PATH",
    "DEFAULT_FEEDBACK_DB_PATH",
    "DEFAULT_MEMORY_DB_PATH",
    "DEFAULT_TEXT_SESSION_DB_PATH",
    "DEFAULT_THREAD_DB_PATH",
    "SESSION_TIMEOUT",
    "ActiveSessionExists",
    "ExpectedSessionLiveness",
    "OpenAITextRuntime",
    "PersistedActiveSessionState",
    "PersistentAgentRuntime",
    "PersistentTurnResult",
    "PostgresActiveSessionStore",
    "SessionInterrupted",
    "SessionLeaseExpired",
    "SessionStatus",
    "SqliteActiveSessionStore",
    "TextRuntimeChunkEvent",
    "TextRuntimeConfig",
    "TextRuntimeShadowResult",
    "TextRuntimeShadowStatus",
    "TextRuntimeStateEvent",
    "TextRuntimeStatusEvent",
    "TextRuntimeStreamEvent",
    "TextSessionBackend",
    "TextSessionStore",
    "TextSessionStoreConfig",
    "ThreadSummary",
    "_iso_now",
    "build_initial_state",
    "create_text_session_store",
    "run_agent",
    "state_to_output",
]
