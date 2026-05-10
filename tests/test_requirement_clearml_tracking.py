"""Tests for optional ClearML tracking helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.requirement_model.clearml_cleanup import validate_cleanup_mode
from scripts.requirement_model.clearml_tracking import (
    CLEARML_ENV_NAMES,
    ClearMlConfigurationError,
    build_dataset_description,
    clear_task_script_diff,
    ensure_clearml_environment,
    force_clearml_requirements_env_freeze,
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
    tracker.report_single_value("summary", 1.0)
    tracker.report_table("table", "rows", [["name", "value"], ["ok", 1]])
    tracker.report_histogram("histogram", "values", [1, 2, 3])
    tracker.report_confusion_matrix("matrix", "labels", [[1, 0], [0, 1]], ["A", "B"])
    tracker.upload_artifact("artifact", {"ok": True})
    assert tracker.register_model_package(Path("."), "model") is None
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


def test_clear_task_script_diff_waits_then_clears() -> None:
    """
    Verify script diff clearing waits for repository detection.
    """

    class FakeTask:
        """
        Minimal ClearML task test double.
        """

        def __init__(self) -> None:
            """
            Initialize recorded calls.
            """
            self.did_wait = False
            self.diff: str | None = None

        def _wait_for_repo_detection(self, timeout: float) -> None:
            """
            Record repository detection wait.

            Args:
                timeout: The timeout passed by the helper.
            """
            self.did_wait = timeout == 30.0

        def set_script(self, diff: str) -> None:
            """
            Record the script diff value.

            Args:
                diff: The diff value passed by the helper.
            """
            self.diff = diff

    task = FakeTask()

    clear_task_script_diff(task)

    assert task.did_wait
    assert task.diff == ""


def test_force_clearml_requirements_env_freeze_uses_full_environment() -> None:
    """
    Verify ClearML package capture is forced before task initialization.
    """

    class FakeTask:
        """
        Minimal ClearML Task class test double.
        """

        did_force = False

        @classmethod
        def force_requirements_env_freeze(cls, force: bool = True) -> None:
            """
            Record the force flag.

            Args:
                force: Whether environment freezing was requested.
            """
            cls.did_force = force

    force_clearml_requirements_env_freeze(FakeTask)

    assert FakeTask.did_force


def test_validate_cleanup_mode_requires_exactly_one_mode() -> None:
    """
    Verify destructive cleanup commands require an explicit single mode.
    """
    validate_cleanup_mode(is_previous_smoke_records=True, is_clear_project=False)
    validate_cleanup_mode(is_previous_smoke_records=False, is_clear_project=True)

    with pytest.raises(ClearMlConfigurationError, match="exactly one mode"):
        validate_cleanup_mode(
            is_previous_smoke_records=False,
            is_clear_project=False,
        )

    with pytest.raises(ClearMlConfigurationError, match="exactly one mode"):
        validate_cleanup_mode(
            is_previous_smoke_records=True,
            is_clear_project=True,
        )
