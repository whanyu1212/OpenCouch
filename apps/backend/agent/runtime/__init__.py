"""Runtime support types and helpers for persistent agent sessions.

The runtime package exports convenience symbols lazily so submodules such as
runtime support modules can be imported without
eagerly importing the full persistent runtime.
"""

from __future__ import annotations

from typing import Any

_RUNTIME_EXPORTS = {
    "PersistentAgentRuntime",
    "_iso_now",
}

_CONFIGURATION_EXPORTS = {
    "DEFAULT_CRISIS_LOG_DB_PATH",
    "DEFAULT_FEEDBACK_DB_PATH",
    "DEFAULT_MEMORY_DB_PATH",
    "DEFAULT_TEXT_SESSION_DB_PATH",
    "DEFAULT_THREAD_DB_PATH",
    "RuntimeBehaviorConfig",
    "RuntimeDependencies",
    "RuntimePersistenceConfig",
    "RuntimeStoragePaths",
    "SESSION_TIMEOUT",
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

_OPENAI_TEXT_RUNTIME_EXPORTS = {
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
    if name in _CONFIGURATION_EXPORTS:
        from agent.runtime import configuration as _configuration

        return getattr(_configuration, name)
    if name in _ACTIVE_SESSION_EXPORTS:
        from agent.runtime.session import active_session as _active_session

        return getattr(_active_session, name)
    if name in _SESSION_STORE_EXPORTS:
        from agent.runtime import session_store as _session_store

        return getattr(_session_store, name)
    if name in _OPENAI_TEXT_RUNTIME_EXPORTS:
        from agent.runtime import openai_text_runtime as _openai_text_runtime

        return getattr(_openai_text_runtime, name)
    if name in _TURN_EXPORTS:
        from agent.runtime import turn as _turn

        return getattr(_turn, name)
    if name in _TYPES_EXPORTS:
        from agent.runtime import types as _types

        return getattr(_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_CRISIS_LOG_DB_PATH",
    "DEFAULT_FEEDBACK_DB_PATH",
    "DEFAULT_MEMORY_DB_PATH",
    "DEFAULT_TEXT_SESSION_DB_PATH",
    "DEFAULT_THREAD_DB_PATH",
    "RuntimeBehaviorConfig",
    "RuntimeDependencies",
    "RuntimePersistenceConfig",
    "RuntimeStoragePaths",
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
