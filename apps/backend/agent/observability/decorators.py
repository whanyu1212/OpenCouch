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


@overload
def trace_span(name: str) -> Callable[[F], F]: ...


@overload
def trace_span(
    name: str,
    *,
    attrs: Mapping[str, Any] | AttrsFactory | None = None,
    result_attrs: ResultAttrsFactory | None = None,
) -> Callable[[F], F]: ...


def trace_span(
    name: str,
    *,
    attrs: Mapping[str, Any] | AttrsFactory | None = None,
    result_attrs: ResultAttrsFactory | None = None,
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
                try:
                    with use_parent_span(span.span_id):
                        async for item in func(*args, **kwargs):
                            yield item
                    span.end(status="ok")
                except Exception as exc:
                    span.end(status="error", error=exc)
                    raise

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
                    span.end(status="error", error=exc)
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
                span.end(status="error", error=exc)
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
