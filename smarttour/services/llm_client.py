"""
Centralized OpenAI-compatible LLM wrapper with structured JSON output.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel

from smarttour.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Centralized client for LLM-backed features.

    Supports any OpenAI-compatible endpoint (OpenAI, OpenRouter, etc.)
    and provides both plain-text completion and structured JSON output
    that is validated into Pydantic models.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model: Override for the model name.
            base_url: Override for the API base URL.
            api_key: Override for the API key.
        """

        settings = get_settings()
        self.model = model or settings.openai_model
        self.base_url = base_url or settings.openai_base_url
        self.api_key = api_key or settings.openai_api_key

    @property
    def enabled(self) -> bool:
        """
        Return whether LLM access is configured.

        Returns:
            ``True`` when an API key is available.
        """

        return bool(self.api_key)

    def _get_client(self):
        """
        Create an OpenAI client configured for the current endpoint.

        Returns:
            An OpenAI client instance.

        Raises:
            RuntimeError: When no API key is configured.
        """

        if not self.enabled:
            raise RuntimeError("LLM client is not configured: missing API key.")
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Request a plain-text completion.

        Args:
            system_prompt: System-level prompt text.
            user_prompt: User-level prompt text.

        Returns:
            The model output text.

        Raises:
            RuntimeError: When the client is not configured or the API call fails.
        """

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("LLM returned empty response.")
        return content.strip()

    def parse(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        """
        Request a structured JSON response and validate it into a Pydantic model.

        The method instructs the LLM to respond with JSON matching the schema
        of ``response_model``, then parses and validates the output.

        Args:
            system_prompt: System-level prompt text.
            user_prompt: User-level prompt text.
            response_model: The Pydantic model class to validate against.

        Returns:
            A validated instance of ``response_model``.

        Raises:
            RuntimeError: When the client is not configured or parsing fails.
        """

        schema = response_model.model_json_schema()
        schema_text = json.dumps(schema, indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON that conforms to this schema:\n"
            f"```json\n{schema_text}\n```\n"
            f"Output ONLY the raw JSON object. No markdown fences, no explanation."
        )
        raw = self.complete(augmented_system, user_prompt)
        cleaned = self._extract_json(raw)
        try:
            return response_model.model_validate_json(cleaned)
        except Exception as first_error:
            logger.warning("JSON validation failed on first attempt: %s", first_error)
            try:
                data = json.loads(cleaned)
                return response_model.model_validate(data)
            except Exception as second_error:
                raise RuntimeError(
                    f"Failed to parse LLM response into {response_model.__name__}: "
                    f"{second_error}\nRaw output: {raw[:500]}"
                ) from second_error

    def _extract_json(self, text: str) -> str:
        """
        Extract a JSON object from LLM output that may contain markdown fences.

        Args:
            text: Raw LLM output text.

        Returns:
            Cleaned JSON string.
        """

        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            start = 1
            end = len(lines)
            for i in range(1, len(lines)):
                if lines[i].strip() == "```":
                    end = i
                    break
            stripped = "\n".join(lines[start:end]).strip()
        brace_start = stripped.find("{")
        brace_end = stripped.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return stripped[brace_start : brace_end + 1]
        return stripped
