"""Explicit registry for OpenAI text-runtime route handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from agent.flows.crisis import build_crisis_route_handler
from agent.flows.grounded_lookup import build_grounded_lookup_route_handler
from agent.flows.guided_exercise import build_guided_exercise_route_handler
from agent.flows.therapeutic import build_therapeutic_route_handler
from agent.runtime.services import TextRuntimeServicesFactory
from agent.runtime.types import RouteHandler


class TextRouteRegistry:
    """Immutable mapping from text route kinds to flow-owned handlers."""

    def __init__(self, handlers: Mapping[str, RouteHandler]) -> None:
        self._handlers = MappingProxyType(dict(handlers))

    @property
    def handlers(self) -> Mapping[str, RouteHandler]:
        """Return the immutable explicit route mapping."""

        return self._handlers

    def handler_for(self, kind: str) -> RouteHandler:
        """Return the registered handler for one route kind.

        Raises:
            KeyError: If no handler is registered for ``kind``.
        """

        return self._handlers[kind]


TextRouteRegistryFactory = Callable[[TextRuntimeServicesFactory], TextRouteRegistry]


def build_default_text_route_registry(
    services_factory: TextRuntimeServicesFactory,
) -> TextRouteRegistry:
    """Compose the product's default flow-owned text route handlers."""

    crisis = build_crisis_route_handler(services_factory)
    therapeutic = build_therapeutic_route_handler(services_factory)
    return TextRouteRegistry(
        {
            "crisis_response": crisis,
            "crisis_clarification": crisis,
            "grounded_lookup": build_grounded_lookup_route_handler(services_factory),
            "guided_exercise": build_guided_exercise_route_handler(services_factory),
            "memory_control": therapeutic,
            "therapeutic": therapeutic,
        }
    )


__all__ = [
    "TextRouteRegistry",
    "TextRouteRegistryFactory",
    "build_default_text_route_registry",
]
