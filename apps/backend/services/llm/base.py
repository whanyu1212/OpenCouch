"""Provider-neutral LLM client interface used by the agent runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class for provider-backed model clients.

    Domain layers should provide prompts, schemas, and use-case-specific
    instructions. Provider implementations should focus only on making model API
    calls and returning normalized results.
    """

    @abstractmethod
    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        """Generate a plain-text completion.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            use_search: Whether to enable provider-native web search/grounding.

        Returns:
            The generated text response.
        """

    @abstractmethod
    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate a plain-text completion as a stream of string chunks.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.

        Yields:
            String chunks of the generated text as they arrive from the provider.
        """
        # This yield makes the method an async generator even though it is abstract.
        # Concrete subclasses must override with their own streaming implementation.
        yield ""  # pragma: no cover

    @abstractmethod
    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        """Generate and parse a structured response.

        Args:
            prompt: The user or task prompt to send to the model.
            response_schema: The Pydantic schema expected in the response.
            system_instruction: Optional top-level instruction for model behavior.

        Returns:
            A parsed Pydantic object matching `response_schema`.
        """
