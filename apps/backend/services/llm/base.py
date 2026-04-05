from __future__ import annotations

from abc import ABC, abstractmethod
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
        temperature: float = 0,
    ) -> str:
        """Generate a plain-text completion.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            temperature: Sampling temperature for the request.

        Returns:
            The generated text response.
        """

    @abstractmethod
    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> StructuredResponseT:
        """Generate and parse a structured response.

        Args:
            prompt: The user or task prompt to send to the model.
            response_schema: The Pydantic schema expected in the response.
            system_instruction: Optional top-level instruction for model behavior.
            temperature: Sampling temperature for the request.

        Returns:
            A parsed Pydantic object matching `response_schema`.
        """
