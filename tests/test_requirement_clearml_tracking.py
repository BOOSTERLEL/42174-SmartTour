"""Tests for optional ClearML tracking helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.requirement_model.clearml_tracking import (
    CLEARML_ENV_NAMES,
    ClearMlConfigurationError,
    build_dataset_description,
    ensure_clearml_environment,
    initialize_clearml_task,
    load_clearml_environment,
)


def test_disabled_tracker_does_not_require_clearml() -> None:
    """
    Verify disabled tracking returns a no-op tracker without credentials.
    """
    tracker = initialize_clearml_task(is_enabled=False, task_name="unused")

    assert not tracker.is_enabled
    tracker.report_scalar("metric", "value", 1.0, 1)
    tracker.upload_artifact("artifact", {"ok": True})
    tracker.close()


def test_load_clearml_environment_reads_dotenv_without_overwriting(
    tmp_path: Path,
) -> None:
    """
    Verify dotenv loading only fills missing ClearML environment values.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                'CLEARML_API_ACCESS_KEY="from-file"',
                "CLEARML_API_SECRET_KEY=secret-from-file",
                "CLEARML_API_HOST=https://api.clear.ml",
                "CLEARML_WEB_HOST=https://app.clear.ml",
                "CLEARML_FILES_HOST=https://files.clear.ml",
            )
        ),
        encoding="utf-8",
    )
    environ = {"CLEARML_API_ACCESS_KEY": "existing"}

    loaded_names = load_clearml_environment(env_file, environ)

    assert "CLEARML_API_ACCESS_KEY" not in loaded_names
    assert environ["CLEARML_API_ACCESS_KEY"] == "existing"
    assert environ["CLEARML_API_SECRET_KEY"] == "secret-from-file"
    assert set(CLEARML_ENV_NAMES).issubset(environ)


def test_missing_clearml_environment_error_is_sanitized(tmp_path: Path) -> None:
    """
    Verify missing credential errors name variables without secret values.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("CLEARML_API_SECRET_KEY=super-secret\n", encoding="utf-8")

    with pytest.raises(ClearMlConfigurationError) as error_info:
        ensure_clearml_environment(env_file, {})

    message = str(error_info.value)
    assert "CLEARML_API_ACCESS_KEY" in message
    assert "CLEARML_API_HOST" in message
    assert "super-secret" not in message


def test_build_dataset_description_sorts_split_counts() -> None:
    """
    Verify dataset descriptions are deterministic.
    """
    description = build_dataset_description({"train": 2, "test": 1})

    assert description == "Requirement model JSONL splits: test=1, train=2"
