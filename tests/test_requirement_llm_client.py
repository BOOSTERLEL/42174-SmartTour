"""Tests for the requirement data LLM augmentation client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from scripts.requirement_model.llm_client import (
    LlmConfigurationError,
    LlmEndpointConfig,
    LlmSettings,
    OpenAiAugmentationClient,
    load_llm_settings,
)
from scripts.requirement_model.prompts import build_augmentation_messages


class RecordingOpenAiClient:
    """
    Minimal OpenAI SDK stand-in that records construction and request payloads.
    """

    instances: list[RecordingOpenAiClient] = []

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the recording client.

        Args:
            kwargs: OpenAI SDK constructor keyword arguments.
        """
        self.kwargs = kwargs
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )
        RecordingOpenAiClient.instances.append(self)

    def create_completion(self, **kwargs: Any) -> Any:
        """
        Record one completion request and return JSON content.

        Args:
            kwargs: OpenAI SDK chat completion keyword arguments.

        Returns:
            A minimal completion-shaped object.
        """
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"utterances": ["Plan [DESTINATION|Tokyo]."]}'
                    )
                )
            ]
        )


class FailingPrimaryClient:
    """
    OpenAI SDK stand-in that fails for the primary endpoint and succeeds later.
    """

    created_names: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the failing client.

        Args:
            kwargs: OpenAI SDK constructor keyword arguments.
        """
        self.kwargs = kwargs
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )
        FailingPrimaryClient.created_names.append(str(kwargs["api_key"]))

    def create_completion(self, **kwargs: Any) -> Any:
        """
        Fail primary calls and return a valid response for backup calls.

        Args:
            kwargs: OpenAI SDK chat completion keyword arguments.

        Returns:
            A minimal completion-shaped object.

        Raises:
            APIConnectionError: Raised for the primary endpoint.
        """
        if self.kwargs["api_key"] == "primary-key":
            request = httpx.Request("POST", "https://example.test")
            raise APIConnectionError(request=request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"utterances": ["Plan [DESTINATION|Paris]."]}'
                    )
                )
            ]
        )


def test_load_llm_settings_loads_primary_and_backup(tmp_path: Any) -> None:
    """
    Verify that primary and backup endpoint variables are loaded by name.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_BASEURL=https://primary.test/v1",
                "OPENAI_API_KEY=primary-key",
                "OPENAI_API_MODEL=primary-model",
                "OPENAI_API_BASEURL_BACKUP=https://backup.test/v1",
                "OPENAI_API_KEY_BACKUP=backup-key",
                "OPENAI_API_MODEL_BACKUP=backup-model",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_llm_settings(env_file=env_file, environ={})

    assert settings.primary == LlmEndpointConfig(
        name="primary",
        base_url="https://primary.test/v1",
        api_key="primary-key",
        model="primary-model",
    )
    assert settings.backup == LlmEndpointConfig(
        name="backup",
        base_url="https://backup.test/v1",
        api_key="backup-key",
        model="backup-model",
    )
    assert settings.warnings == ()


def test_load_llm_settings_uses_backup_when_primary_missing(tmp_path: Any) -> None:
    """
    Verify that a complete backup endpoint can be used without a primary.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_BASEURL_BACKUP=https://backup.test/v1",
                "OPENAI_API_KEY_BACKUP=backup-key",
                "OPENAI_API_MODEL_BACKUP=backup-model",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_llm_settings(env_file=env_file, environ={})

    assert settings.primary is None
    assert settings.backup is not None
    assert "OPENAI_API_KEY" in settings.warnings[0]
    assert "backup-key" not in settings.warnings[0]


def test_load_llm_settings_reports_missing_names_without_values(
    tmp_path: Any,
) -> None:
    """
    Verify that configuration errors list variable names without secret values.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret-value", encoding="utf-8")

    with pytest.raises(LlmConfigurationError) as error_info:
        load_llm_settings(env_file=env_file, environ={})

    error_text = str(error_info.value)
    assert "OPENAI_API_BASEURL" in error_text
    assert "OPENAI_API_MODEL" in error_text
    assert "secret-value" not in error_text


def test_client_builds_official_openai_sdk_request_shape() -> None:
    """
    Verify that the client uses standard OpenAI SDK construction and fields.
    """
    RecordingOpenAiClient.instances = []
    settings = LlmSettings(
        primary=LlmEndpointConfig(
            name="primary",
            base_url="https://primary.test/v1",
            api_key="primary-key",
            model="primary-model",
        ),
        backup=None,
    )
    client = OpenAiAugmentationClient(
        settings=settings,
        client_factory=RecordingOpenAiClient,
        retry_sleep_seconds=0,
    )
    messages = build_augmentation_messages(["Plan [DESTINATION|Tokyo]."], 1)

    utterances = client.generate_marked_utterances(messages)

    assert utterances == ["Plan [DESTINATION|Tokyo]."]
    sdk_client = RecordingOpenAiClient.instances[0]
    assert sdk_client.kwargs == {
        "api_key": "primary-key",
        "base_url": "https://primary.test/v1",
        "timeout": 60.0,
    }
    request = sdk_client.requests[0]
    assert request["model"] == "primary-model"
    assert request["messages"] == messages
    assert request["temperature"] == 0.7
    assert request["response_format"] == {"type": "json_object"}
    assert "extra_body" not in request


def test_client_falls_back_to_backup_after_retryable_primary_failure() -> None:
    """
    Verify that retryable primary failures use the backup endpoint.
    """
    FailingPrimaryClient.created_names = []
    settings = LlmSettings(
        primary=LlmEndpointConfig(
            name="primary",
            base_url="https://primary.test/v1",
            api_key="primary-key",
            model="primary-model",
        ),
        backup=LlmEndpointConfig(
            name="backup",
            base_url="https://backup.test/v1",
            api_key="backup-key",
            model="backup-model",
        ),
    )
    client = OpenAiAugmentationClient(
        settings=settings,
        client_factory=FailingPrimaryClient,
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    utterances = client.generate_marked_utterances(
        build_augmentation_messages(["Plan [DESTINATION|Paris]."], 1)
    )

    assert utterances == ["Plan [DESTINATION|Paris]."]
    assert FailingPrimaryClient.created_names == ["primary-key", "backup-key"]
