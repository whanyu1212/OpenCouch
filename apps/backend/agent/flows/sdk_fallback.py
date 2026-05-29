"""Shared OpenAI SDK fallback helpers for flow executors."""

from __future__ import annotations

from openai import APIConnectionError, AuthenticationError, OpenAIError

from agent.runtime_context import WorkflowContext

_OPENAI_API_KEY_FALLBACK_REASON = "missing_openai_api_key"
_OPENAI_CONNECTION_FALLBACK_REASON = "openai_api_connection_error"


def can_fallback_to_control_response(
    exc: Exception,
    context: WorkflowContext,
) -> bool:
    """Return whether the control LLM can replace a failed SDK turn."""

    return (
        context.llm_client is not None and openai_sdk_fallback_reason(exc) is not None
    )


def openai_sdk_fallback_reason(exc: Exception) -> str | None:
    """Return the structured fallback reason for recoverable SDK failures."""

    if _is_missing_openai_api_key_error(exc):
        return _OPENAI_API_KEY_FALLBACK_REASON
    if isinstance(exc, APIConnectionError):
        return _OPENAI_CONNECTION_FALLBACK_REASON
    return None


def _is_missing_openai_api_key_error(exc: Exception) -> bool:
    if isinstance(exc, AuthenticationError):
        return True
    if not isinstance(exc, OpenAIError):
        return False
    message = str(exc)
    return (
        "OPENAI_API_KEY" in message and "api_key" in message
    ) or "Missing bearer or basic authentication in header" in message


__all__ = [
    "can_fallback_to_control_response",
    "openai_sdk_fallback_reason",
]
