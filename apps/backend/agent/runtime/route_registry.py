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
    """Immutable route-to-handler mapping with an explicit default handler."""

    def __init__(
        self,
        handlers: Mapping[str, RouteHandler],
        *,
        default_handler: RouteHandler,
    ) -> None:
        self._handlers = MappingProxyType(dict(handlers))
        self._default_handler = default_handler

    @property
    def handlers(self) -> Mapping[str, RouteHandler]:
        """Return the immutable explicit route mapping."""

        return self._handlers

    @property
    def default_handler(self) -> RouteHandler:
        """Return the handler used for therapeutic and unknown route kinds."""

        return self._default_handler

    def handler_for(self, kind: str) -> RouteHandler:
        """Resolve one route kind, falling back to the therapeutic handler."""

        return self._handlers.get(kind, self._default_handler)


TextRouteRegistryFactory = Callable[[TextRuntimeServicesFactory], TextRouteRegistry]


def build_default_text_route_registry(
    services_factory: TextRuntimeServicesFactory,
) -> TextRouteRegistry:
    """Compose the product's default flow-owned text route handlers."""

    crisis = build_crisis_route_handler(services_factory)
    return TextRouteRegistry(
        {
            "crisis_response": crisis,
            "crisis_clarification": crisis,
            "grounded_lookup": build_grounded_lookup_route_handler(services_factory),
            "guided_exercise": build_guided_exercise_route_handler(services_factory),
        },
        default_handler=build_therapeutic_route_handler(services_factory),
    )


__all__ = [
    "TextRouteRegistry",
    "TextRouteRegistryFactory",
    "build_default_text_route_registry",
]
