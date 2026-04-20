"""Google Gen AI provider adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import cast

from google import genai
from google.genai import types

from services.llm.base import BaseLLMClient, StructuredResponseT

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


class GeminiLLMClient(BaseLLMClient):
    """Google Gen AI implementation of `BaseLLMClient`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
    ) -> None:
        """Initialize a Gemini-backed model client.

        Args:
            api_key: Optional explicit API key override.
            model: Model identifier to use for requests.

        Raises:
            ValueError: If no Gemini API key can be resolved.
        """
        resolved_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        self.model = model
        self.client = genai.Client(api_key=resolved_key)

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        """Generate a plain-text response with Gemini.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            use_search: When True, enables Google Search grounding so the model
                can cite live web results (e.g. regional crisis hotlines).

        Returns:
            The generated text response.

        Raises:
            ValueError: If Gemini returns an empty text payload.
        """
        tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
            ),
        )
        text = response.text
        if not text:
            raise ValueError("Gemini text generation returned an empty response.")
        return text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a plain-text response from Gemini.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.

        Yields:
            String chunks of the generated text as they arrive.
        """

        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        """Generate a structured response with Gemini.

        Args:
            prompt: The user or task prompt to send to the model.
            response_schema: The Pydantic schema expected in the response.
            system_instruction: Optional top-level instruction for model behavior.

        Returns:
            A parsed object matching `response_schema`.

        Raises:
            ValueError: If Gemini does not return parsed structured output.
        """

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        parsed = response.parsed
        if not isinstance(parsed, response_schema):
            raise ValueError("Gemini structured generation did not return parsed JSON.")
        return cast(StructuredResponseT, parsed)
