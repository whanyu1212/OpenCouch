"""Decorator-style tracing helpers for agent observability."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, TypeVar, overload

from agent.observability.context import (
    get_current_span_id,
    get_current_trace_recorder,
    use_parent_span,
)
from agent.observability.redaction import sanitize_attributes

F = TypeVar("F", bound=Callable[..., Any])
AttrsFactory = Callable[[tuple[Any, ...], dict[str, Any]], Mapping[str, Any] | None]
ResultAttrsFactory = Callable[[Any], Mapping[str, Any] | None]
ErrorAttrsFactory = Callable[[BaseException], Mapping[str, Any] | None]


@overload
def trace_span(name: str) -> Callable[[F], F]: ...


@overload
def trace_span(
    name: str,
    *,
    attrs: Mapping[str, Any] | AttrsFactory | None = None,
    result_attrs: ResultAttrsFactory | None = None,
    error_attrs: Mapping[str, Any] | ErrorAttrsFactory | None = None,
    record_error_message: bool = True,
) -> Callable[[F], F]: ...


def trace_span(
    name: str,
    *,
    attrs: Mapping[str, Any] | AttrsFactory | None = None,
    result_attrs: ResultAttrsFactory | None = None,
    error_attrs: Mapping[str, Any] | ErrorAttrsFactory | None = None,
    record_error_message: bool = True,
) -> Callable[[F], F]:
    """Decorate a sync or async function with a trace span."""

    def decorator(func: F) -> F:
        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def async_generator_wrapper(*args: Any, **kwargs: Any) -> Any:
                recorder = get_current_trace_recorder()
                span = recorder.start_span(
                    name,
                    _resolve_attrs(attrs, args, kwargs),
                    parent_span_id=get_current_span_id(),
                )
                ended = False
                try:
                    with use_parent_span(span.span_id):
                        async for item in func(*args, **kwargs):
                            yield item
                    span.end(status="ok")
                    ended = True
                except Exception as exc:
                    _end_span_with_error(
                        span,
                        exc,
                        error_attrs=error_attrs,
                        record_error_message=record_error_message,
                    )
                    ended = True
                    raise
                finally:
                    if not ended:
                        span.end(status="closed")

            return async_generator_wrapper  # type: ignore[return-value]

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                recorder = get_current_trace_recorder()
                span = recorder.start_span(
                    name,
                    _resolve_attrs(attrs, args, kwargs),
                    parent_span_id=get_current_span_id(),
                )
                try:
                    with use_parent_span(span.span_id):
                        result = await func(*args, **kwargs)
                    span.end(
                        status="ok",
                        attributes=_resolve_result_attrs(result_attrs, result),
                    )
                    return result
                except Exception as exc:
                    _end_span_with_error(
                        span,
                        exc,
                        error_attrs=error_attrs,
                        record_error_message=record_error_message,
                    )
                    raise

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            recorder = get_current_trace_recorder()
            span = recorder.start_span(
                name,
                _resolve_attrs(attrs, args, kwargs),
                parent_span_id=get_current_span_id(),
            )
            try:
                with use_parent_span(span.span_id):
                    result = func(*args, **kwargs)
                span.end(
                    status="ok",
                    attributes=_resolve_result_attrs(result_attrs, result),
                )
                return result
            except Exception as exc:
                _end_span_with_error(
                    span,
                    exc,
                    error_attrs=error_attrs,
                    record_error_message=record_error_message,
                )
                raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def trace_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    """Record a semantic trace event under the active span when available."""

    try:
        get_current_trace_recorder().event(
            name,
            sanitize_attributes(attributes),
            span_id=get_current_span_id(),
        )
    except Exception:
        return


def _resolve_attrs(
    attrs: Mapping[str, Any] | AttrsFactory | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if attrs is None:
        return {}
    try:
        if callable(attrs):
            return sanitize_attributes(attrs(args, kwargs))
        return sanitize_attributes(attrs)
    except Exception:
        return {}


def _resolve_result_attrs(
    result_attrs: ResultAttrsFactory | None,
    result: Any,
) -> dict[str, Any]:
    if result_attrs is None:
        return {}
    try:
        return sanitize_attributes(result_attrs(result))
    except Exception:
        return {}


def _resolve_error_attrs(
    error_attrs: Mapping[str, Any] | ErrorAttrsFactory | None,
    error: BaseException,
) -> dict[str, Any]:
    if error_attrs is None:
        return {}
    try:
        if callable(error_attrs):
            return sanitize_attributes(error_attrs(error))
        return sanitize_attributes(error_attrs)
    except Exception:
        return {}


def _end_span_with_error(
    span: Any,
    error: BaseException,
    *,
    error_attrs: Mapping[str, Any] | ErrorAttrsFactory | None,
    record_error_message: bool,
) -> None:
    attributes = _resolve_error_attrs(error_attrs, error)
    if record_error_message:
        span.end(status="error", error=error, attributes=attributes)
        return

    span.end(
        status="error", attributes={"error_type": type(error).__name__, **attributes}
    )
