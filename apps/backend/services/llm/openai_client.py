"""OpenAI provider adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI

from services.llm.base import BaseLLMClient, StructuredResponseT

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class OpenAILLMClient(BaseLLMClient):
    """OpenAI implementation of `BaseLLMClient`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        """Initialize an OpenAI-backed model client.

        Args:
            api_key: Optional explicit API key override.
            model: Model identifier to use for requests.

        Raises:
            ValueError: If no OpenAI API key can be resolved.
        """
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY.")

        self.model = model
        self.client = AsyncOpenAI(api_key=resolved_key)

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
        use_search: bool = False,
    ) -> str:
        """Generate a plain-text response with OpenAI.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            temperature: Sampling temperature for the request.
            use_search: Unused for the OpenAI provider; accepted for interface
                compatibility with `BaseLLMClient`.

        Returns:
            The generated text response.

        Raises:
            ValueError: If OpenAI returns an empty text payload.
        """
        input_items = []
        if system_instruction:
            input_items.append({"role": "system", "content": system_instruction})
        input_items.append({"role": "user", "content": prompt})

        response = await self.client.responses.create(
            model=self.model,
            input=input_items,
            temperature=temperature,
        )
        text = response.output_text
        if not text:
            raise ValueError("OpenAI text generation returned an empty response.")
        return text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> AsyncIterator[str]:
        """Stream a plain-text response from OpenAI.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            temperature: Sampling temperature for the request.

        Yields:
            String chunks of the generated text as they arrive.
        """

        input_items = []
        if system_instruction:
            input_items.append({"role": "system", "content": system_instruction})
        input_items.append({"role": "user", "content": prompt})

        async with self.client.responses.stream(
            model=self.model,
            input=input_items,
            temperature=temperature,
        ) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> StructuredResponseT:
        """Generate a structured response with OpenAI.

        Args:
            prompt: The user or task prompt to send to the model.
            response_schema: The Pydantic schema expected in the response.
            system_instruction: Optional top-level instruction for model behavior.
            temperature: Sampling temperature for the request.

        Returns:
            A parsed object matching `response_schema`.

        Raises:
            ValueError: If OpenAI does not return parsed structured output.
        """
        input_items = []
        if system_instruction:
            input_items.append({"role": "system", "content": system_instruction})
        input_items.append({"role": "user", "content": prompt})

        response = await self.client.responses.parse(
            model=self.model,
            input=input_items,
            text_format=response_schema,
            temperature=temperature,
        )

        parsed = response.output_parsed
        if not isinstance(parsed, response_schema):
            raise ValueError(
                "OpenAI structured generation did not return parsed output."
            )

        return cast(StructuredResponseT, parsed)
