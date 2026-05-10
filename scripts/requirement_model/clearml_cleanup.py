"""Delete ClearML records for requirement model reruns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.clearml_tracking import (
    DEFAULT_CLEARML_PROJECT,
    ClearMlConfigurationError,
    ensure_clearml_environment,
)

PREVIOUS_SMOKE_TASK_IDS: tuple[str, ...] = (
    "42807344bd364a9e81ef429d98698725",
    "59e9972771e64a42a30a04397155a818",
    "6503983e1ba946e184aa29eb113ac5de",
)
PREVIOUS_SMOKE_DATASET_NAME = "requirement_model_data"
PREVIOUS_SMOKE_DATASET_VERSION = "clearml-smoke-20260510-1512"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Delete ClearML requirement model records."
    )
    parser.add_argument("--previous-smoke-records", action="store_true")
    parser.add_argument("--clear-project", action="store_true")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    return parser.parse_args()


def delete_task(task_id: str) -> str:
    """
    Delete one ClearML task by identifier.

    Args:
        task_id: The ClearML task identifier.

    Returns:
        A status string for command output.
    """
    from clearml import Task

    try:
        task: Any = Task.get_task(task_id=task_id)
    except Exception as error:  # noqa: BLE001
        error_name = type(error).__name__
        return f"task {task_id}: already absent or inaccessible ({error_name})"
    did_delete = task.delete(
        delete_artifacts_and_models=True,
        skip_models_used_by_other_tasks=True,
        raise_on_error=False,
    )
    if did_delete:
        return f"task {task_id}: deleted"
    return f"task {task_id}: delete skipped or already absent"


def delete_dataset(
    project_name: str, dataset_name: str, dataset_version: str
) -> str:
    """
    Delete one ClearML Dataset version.

    Args:
        project_name: The ClearML dataset project name.
        dataset_name: The ClearML dataset name.
        dataset_version: The ClearML dataset version.

    Returns:
        A status string for command output.
    """
    from clearml import Dataset

    try:
        Dataset.delete(
            dataset_project=project_name,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            force=True,
            entire_dataset=False,
            delete_files=True,
        )
    except Exception as error:  # noqa: BLE001
        return (
            f"dataset {project_name}/{dataset_name}@{dataset_version}: "
            f"already absent or inaccessible ({type(error).__name__})"
        )
    return f"dataset {project_name}/{dataset_name}@{dataset_version}: deleted"


def delete_previous_smoke_records(project_name: str) -> list[str]:
    """
    Delete the known previous ClearML smoke records.

    Args:
        project_name: The ClearML project name.

    Returns:
        Status lines for deleted or already-absent objects.
    """
    ensure_clearml_environment()
    results = [delete_task(task_id) for task_id in PREVIOUS_SMOKE_TASK_IDS]
    results.append(
        delete_dataset(
            project_name=project_name,
            dataset_name=PREVIOUS_SMOKE_DATASET_NAME,
            dataset_version=PREVIOUS_SMOKE_DATASET_VERSION,
        )
    )
    return results


def delete_project_records(project_name: str) -> list[str]:
    """
    Delete a ClearML project and all of its contents.

    Args:
        project_name: The ClearML project name.

    Returns:
        Status lines for deleted or already-absent project contents.
    """
    ensure_clearml_environment()

    from clearml import Task
    from clearml.backend_api.session.client import APIClient

    project_id = Task.get_project_id(project_name, search_hidden=True)
    if project_id is None:
        return [f"project {project_name}: already absent"]
    client = APIClient()
    client.projects.delete(
        project=project_id,
        force=True,
        delete_contents=True,
    )
    return [f"project {project_name} ({project_id}): deleted with contents"]


def validate_cleanup_mode(
    is_previous_smoke_records: bool, is_clear_project: bool
) -> None:
    """
    Validate that exactly one cleanup mode was requested.

    Args:
        is_previous_smoke_records: Whether known smoke-record cleanup was requested.
        is_clear_project: Whether full project cleanup was requested.

    Raises:
        ClearMlConfigurationError: Raised when the cleanup mode is ambiguous.
    """
    selected_mode_count = int(is_previous_smoke_records) + int(is_clear_project)
    if selected_mode_count != 1:
        raise ClearMlConfigurationError(
            "cleanup requires exactly one mode: "
            "--previous-smoke-records or --clear-project"
        )


def main() -> None:
    """
    Delete known previous smoke records from ClearML.
    """
    args = parse_args()
    validate_cleanup_mode(args.previous_smoke_records, args.clear_project)
    if args.clear_project:
        results = delete_project_records(args.clearml_project)
    else:
        results = delete_previous_smoke_records(args.clearml_project)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
