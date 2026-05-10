"""OpenAI SDK client for requirement data augmentation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

PRIMARY_ENV_NAMES = (
    "OPENAI_API_BASEURL",
    "OPENAI_API_KEY",
    "OPENAI_API_MODEL",
)
BACKUP_ENV_NAMES = (
    "OPENAI_API_BASEURL_BACKUP",
    "OPENAI_API_KEY_BACKUP",
    "OPENAI_API_MODEL_BACKUP",
)
RESPONSE_FORMAT = {"type": "json_object"}


class LlmConfigurationError(RuntimeError):
    """
    Error raised when LLM endpoint configuration is incomplete.
    """


class LlmRequestError(RuntimeError):
    """
    Error raised when all configured LLM endpoints fail.
    """


@dataclass(frozen=True, slots=True)
class LlmEndpointConfig:
    """
    OpenAI-compatible endpoint configuration.
    """

    name: str
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True, slots=True)
class LlmSettings:
    """
    Primary and backup LLM endpoint settings.
    """

    primary: LlmEndpointConfig | None
    backup: LlmEndpointConfig | None
    warnings: tuple[str, ...] = ()

    def ordered_endpoints(self) -> tuple[LlmEndpointConfig, ...]:
        """
        Return configured endpoints in deterministic failover order.

        Returns:
            The configured endpoints.
        """
        endpoints: list[LlmEndpointConfig] = []
        if self.primary is not None:
            endpoints.append(self.primary)
        if self.backup is not None:
            endpoints.append(self.backup)
        return tuple(endpoints)


class OpenAiAugmentationClient:
    """
    Generate marked utterances through the official OpenAI Python SDK.
    """

    def __init__(
        self,
        settings: LlmSettings,
        client_factory: Callable[..., Any] = OpenAI,
        max_attempts: int = 3,
        timeout_seconds: float = 60.0,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        """
        Initialize the augmentation client.

        Args:
            settings: The primary and backup LLM endpoint settings.
            client_factory: Factory compatible with `openai.OpenAI`.
            max_attempts: Maximum attempts per endpoint.
            timeout_seconds: Per-request SDK timeout in seconds.
            retry_sleep_seconds: Delay between retryable failures.

        Raises:
            LlmConfigurationError: Raised when no endpoint is configured.
        """
        if not settings.ordered_endpoints():
            raise LlmConfigurationError("no complete LLM endpoint is configured")
        self.settings = settings
        self.client_factory = client_factory
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.retry_sleep_seconds = retry_sleep_seconds

    def generate_marked_utterances(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> list[str]:
        """
        Generate marked utterances from chat messages.

        Args:
            messages: Chat messages for the augmentation prompt.
            temperature: Sampling temperature for diverse paraphrases.
            max_tokens: Maximum output tokens.

        Returns:
            The generated marked utterances.
        """
        content = self.create_json_completion(messages, temperature, max_tokens)
        payload = parse_json_object(content)
        utterances = payload.get("utterances")
        if not isinstance(utterances, list) or not all(
            isinstance(value, str) for value in utterances
        ):
            raise LlmRequestError("LLM response did not contain utterance strings")
        return utterances

    def create_json_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Create one JSON chat completion with primary-first failover.

        Args:
            messages: Chat messages for the model.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            The message content string.

        Raises:
            LlmRequestError: Raised when all endpoints fail.
        """
        last_error: Exception | None = None
        for endpoint in self.settings.ordered_endpoints():
            try:
                return self._create_with_endpoint(
                    endpoint, messages, temperature, max_tokens
                )
            except (AuthenticationError, BadRequestError) as error:
                raise LlmRequestError(
                    f"{endpoint.name} LLM endpoint failed with a non-retryable error"
                ) from error
            except (APIConnectionError, APITimeoutError, RateLimitError) as error:
                last_error = error
            except APIStatusError as error:
                if not is_retryable_status(error):
                    raise LlmRequestError(
                        f"{endpoint.name} LLM endpoint returned status "
                        f"{error.status_code}"
                    ) from error
                last_error = error
        raise LlmRequestError("all configured LLM endpoints failed") from last_error

    def _create_with_endpoint(
        self,
        endpoint: LlmEndpointConfig,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Create one JSON completion against a single endpoint.

        Args:
            endpoint: The endpoint configuration.
            messages: Chat messages for the model.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            The response content.
        """
        client = self.client_factory(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            timeout=self.timeout_seconds,
        )
        for attempt_index in range(1, self.max_attempts + 1):
            try:
                completion = client.chat.completions.create(
                    model=endpoint.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=RESPONSE_FORMAT,
                )
                content = completion.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise LlmRequestError("LLM response content was empty")
                return content
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt_index >= self.max_attempts:
                    raise
                time.sleep(self.retry_sleep_seconds)
            except APIStatusError as error:
                if attempt_index >= self.max_attempts or not is_retryable_status(error):
                    raise
                time.sleep(self.retry_sleep_seconds)
        raise LlmRequestError(f"{endpoint.name} LLM endpoint exhausted attempts")


def load_llm_settings(
    env_file: Path = Path(".env"),
    environ: Mapping[str, str] | None = None,
) -> LlmSettings:
    """
    Load primary and backup LLM settings from environment values.

    Args:
        env_file: Optional dotenv file path.
        environ: Process environment override mapping.

    Returns:
        The loaded LLM settings.

    Raises:
        LlmConfigurationError: Raised when no complete endpoint is available.
    """
    values = read_env_file(env_file)
    environment = os.environ if environ is None else environ
    values.update(environment)
    primary, primary_missing = build_endpoint("primary", PRIMARY_ENV_NAMES, values)
    backup, backup_missing = build_endpoint("backup", BACKUP_ENV_NAMES, values)
    warnings = build_configuration_warnings(primary, backup, primary_missing)
    if primary is None and backup is None:
        missing_names = sorted(set(primary_missing + backup_missing))
        joined_names = ", ".join(missing_names)
        raise LlmConfigurationError(
            f"missing complete LLM configuration variables: {joined_names}"
        )
    return LlmSettings(primary=primary, backup=backup, warnings=tuple(warnings))


def read_env_file(env_file: Path) -> dict[str, str]:
    """
    Read simple dotenv key-value pairs.

    Args:
        env_file: The dotenv file path.

    Returns:
        Parsed environment values.
    """
    values: dict[str, str] = {}
    if not env_file.exists():
        return values
    with env_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            if "=" not in stripped_line:
                continue
            name, value = stripped_line.split("=", maxsplit=1)
            values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def build_endpoint(
    name: str, env_names: tuple[str, str, str], values: Mapping[str, str]
) -> tuple[LlmEndpointConfig | None, list[str]]:
    """
    Build one endpoint configuration from environment values.

    Args:
        name: The endpoint name.
        env_names: Base URL, API key, and model variable names.
        values: Environment values.

    Returns:
        The endpoint configuration and missing variable names.
    """
    base_url_name, api_key_name, model_name = env_names
    missing_names = [
        env_name for env_name in env_names if not values.get(env_name, "").strip()
    ]
    if missing_names:
        return None, missing_names
    return (
        LlmEndpointConfig(
            name=name,
            base_url=values[base_url_name].strip(),
            api_key=values[api_key_name].strip(),
            model=values[model_name].strip(),
        ),
        [],
    )


def build_configuration_warnings(
    primary: LlmEndpointConfig | None,
    backup: LlmEndpointConfig | None,
    primary_missing: list[str],
) -> list[str]:
    """
    Build sanitized configuration warnings.

    Args:
        primary: The primary endpoint, if complete.
        backup: The backup endpoint, if complete.
        primary_missing: Missing primary variable names.

    Returns:
        Warning messages without secret values.
    """
    if primary is None and backup is not None:
        joined_names = ", ".join(primary_missing)
        return [f"primary LLM configuration incomplete: {joined_names}"]
    return []


def parse_json_object(content: str) -> dict[str, Any]:
    """
    Parse a JSON object response.

    Args:
        content: The response content.

    Returns:
        The parsed JSON object.

    Raises:
        LlmRequestError: Raised when content is not a JSON object.
    """
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise LlmRequestError("LLM response was not valid JSON") from error
    if not isinstance(payload, dict):
        raise LlmRequestError("LLM response JSON was not an object")
    return payload


def is_retryable_status(error: APIStatusError) -> bool:
    """
    Return whether an API status error should be retried.

    Args:
        error: The OpenAI SDK status error.

    Returns:
        Whether the status is retryable.
    """
    return error.status_code in {408, 409, 429} or error.status_code >= 500
