"""Google Gen AI provider adapter."""

from __future__ import annotations

import os
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
        temperature: float = 0,
    ) -> str:
        """Generate a plain-text response with Gemini.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            temperature: Sampling temperature for the request.

        Returns:
            The generated text response.

        Raises:
            ValueError: If Gemini returns an empty text payload.
        """
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            ),
        )
        text = response.text
        if not text:
            raise ValueError("Gemini text generation returned an empty response.")
        return text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        temperature: float = 0,
    ) -> StructuredResponseT:
        """Generate a structured response with Gemini.

        Args:
            prompt: The user or task prompt to send to the model.
            response_schema: The Pydantic schema expected in the response.
            system_instruction: Optional top-level instruction for model behavior.
            temperature: Sampling temperature for the request.

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
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        parsed = response.parsed
        if not isinstance(parsed, response_schema):
            raise ValueError("Gemini structured generation did not return parsed JSON.")
        return cast(StructuredResponseT, parsed)
