"""OpenAI provider adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, cast

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
        use_search: bool = False,
    ) -> str:
        """Generate a plain-text response with OpenAI.

        When ``use_search=True``, the OpenAI hosted ``web_search`` tool
        is attached to the Responses API call so the model can ground
        its reply against live web results. This is the call path the
        crisis resource lookup tool uses to find verified regional
        hotlines.

        Before v0.8, this client silently ignored ``use_search=True``
        and documented the parameter as "unused for interface parity."
        That was a real safety gap for the crisis resource lookup
        path — the tool's system prompt told the model to "use your
        web search capability" but the client never attached a search
        tool, leaving the model free to produce results from training
        data (sometimes accurate for well-known regions, sometimes
        hallucinated for less-common ones). The v0.8 fix wires up
        OpenAI's ``web_search`` tool so ``use_search=True`` does what
        it says on the tin.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.
            use_search: When True, attach OpenAI's hosted ``web_search``
                tool so the model can ground its reply against live
                web results. When False (the default), the call runs
                without any tool attached.

        Returns:
            The generated text response.

        Raises:
            ValueError: If OpenAI returns an empty text payload.
        """
        input_items: list[dict[str, str]] = []
        if system_instruction:
            input_items.append({"role": "system", "content": system_instruction})
        input_items.append({"role": "user", "content": prompt})

        # The Responses API accepts ``tools`` as an optional parameter.
        # When absent, the call behaves exactly as before. When present,
        # the model may route the call through the attached tool before
        # producing its final text — but ``response.output_text`` still
        # returns just the final user-visible text, so the caller
        # contract stays the same.
        #
        # We only attach the tool list when ``use_search=True``; passing
        # an empty list is a different code path in the SDK (it forces
        # tools-enabled request shape with no tools available) and
        # isn't what we want for the common no-search case.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }
        if use_search:
            kwargs["tools"] = [{"type": "web_search"}]

        response = await self.client.responses.create(**kwargs)
        text = response.output_text
        if not text:
            raise ValueError("OpenAI text generation returned an empty response.")
        return text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a plain-text response from OpenAI.

        Args:
            prompt: The user or task prompt to send to the model.
            system_instruction: Optional top-level instruction for model behavior.

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
    ) -> StructuredResponseT:
        """Generate a structured response with OpenAI.

        Args:
            prompt: The user or task prompt to send to the model.
            response_schema: The Pydantic schema expected in the response.
            system_instruction: Optional top-level instruction for model behavior.

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
        )

        parsed = response.output_parsed
        if not isinstance(parsed, response_schema):
            raise ValueError(
                "OpenAI structured generation did not return parsed output."
            )

        return cast(StructuredResponseT, parsed)
