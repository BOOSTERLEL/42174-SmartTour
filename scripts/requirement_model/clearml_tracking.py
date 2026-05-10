"""Optional ClearML tracking helpers for requirement model scripts."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CLEARML_PROJECT = "Smartour"
CLEARML_ENV_NAMES: tuple[str, ...] = (
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
    "CLEARML_API_HOST",
    "CLEARML_WEB_HOST",
    "CLEARML_FILES_HOST",
)


class ClearMlConfigurationError(RuntimeError):
    """
    Error raised when ClearML tracking is requested but unavailable.
    """


@dataclass(slots=True)
class ClearMlTracker:
    """
    Small no-op-safe wrapper around a ClearML task.
    """

    task: Any | None = None

    @property
    def is_enabled(self) -> bool:
        """
        Return whether a live ClearML task is attached.

        Returns:
            Whether this tracker can report to ClearML.
        """
        return self.task is not None

    def report_scalar(
        self, title: str, series: str, value: float, iteration: int
    ) -> None:
        """
        Report one scalar value to the ClearML task.

        Args:
            title: The scalar chart title.
            series: The series name inside the chart.
            value: The scalar value.
            iteration: The scalar iteration.
        """
        if self.task is None:
            return
        self.task.get_logger().report_scalar(
            title=title,
            series=series,
            value=value,
            iteration=iteration,
        )

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        """
        Upload one artifact to the ClearML task.

        Args:
            name: The ClearML artifact name.
            artifact_object: The local path or serializable object to upload.
        """
        if self.task is None:
            return
        self.task.upload_artifact(name=name, artifact_object=artifact_object)

    def report_single_value(self, name: str, value: float) -> None:
        """
        Report one summary value to the ClearML task.

        Args:
            name: The metric name.
            value: The metric value.
        """
        if self.task is None:
            return
        self.task.get_logger().report_single_value(name=name, value=value)

    def report_table(
        self,
        title: str,
        series: str,
        rows: Sequence[Sequence[Any]],
        iteration: int = 0,
    ) -> None:
        """
        Report a table to the ClearML task.

        Args:
            title: The table title.
            series: The table series name.
            rows: Table rows, usually including a header row.
            iteration: The reporting iteration.
        """
        if self.task is None:
            return
        self.task.get_logger().report_table(
            title=title,
            series=series,
            table_plot=list(rows),
            iteration=iteration,
        )

    def report_histogram(
        self,
        title: str,
        series: str,
        values: Sequence[int | float],
        iteration: int = 0,
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        """
        Report a histogram to the ClearML task.

        Args:
            title: The histogram title.
            series: The histogram series name.
            values: Numeric values to plot.
            iteration: The reporting iteration.
            xaxis: Optional x-axis label.
            yaxis: Optional y-axis label.
        """
        if self.task is None:
            return
        self.task.get_logger().report_histogram(
            title=title,
            series=series,
            values=list(values),
            iteration=iteration,
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def report_confusion_matrix(
        self,
        title: str,
        series: str,
        matrix: Sequence[Sequence[int | float]],
        labels: Sequence[str],
        iteration: int = 0,
    ) -> None:
        """
        Report a confusion matrix to the ClearML task.

        Args:
            title: The matrix title.
            series: The matrix series name.
            matrix: Matrix values.
            labels: Matrix axis labels.
            iteration: The reporting iteration.
        """
        if self.task is None:
            return
        import numpy as np

        self.task.get_logger().report_confusion_matrix(
            title=title,
            series=series,
            matrix=np.array(matrix),
            iteration=iteration,
            xlabels=list(labels),
            ylabels=list(labels),
        )

    def register_model_package(
        self,
        model_path: Path,
        name: str,
        config: Mapping[str, Any] | None = None,
        labels: Mapping[str, int] | None = None,
    ) -> str | None:
        """
        Register a local model directory as a ClearML output model package.

        Args:
            model_path: The local model directory.
            name: The ClearML output model name.
            config: Optional model configuration dictionary.
            labels: Optional label enumeration.

        Returns:
            The registered model URI, if ClearML returned one.
        """
        if self.task is None:
            return None
        from clearml import OutputModel

        output_model = OutputModel(
            task=self.task,
            config_dict=dict(config) if config is not None else None,
            label_enumeration=dict(labels) if labels is not None else None,
            name=name,
            framework="PyTorch",
        )
        return output_model.update_weights_package(
            weights_path=str(model_path),
            target_filename=f"{name}.zip",
        )

    def close(self) -> None:
        """
        Close the ClearML task when one exists.
        """
        if self.task is None:
            return
        flush = getattr(self.task, "flush", None)
        if callable(flush):
            flush(wait_for_uploads=True)
        self.task.close()


def load_clearml_environment(
    env_file: Path = Path(".env"),
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """
    Load ClearML credential variables from a dotenv file.

    Args:
        env_file: The dotenv file to read.
        environ: Optional environment mapping for tests.

    Returns:
        Names loaded into the target environment.
    """
    target_environ = os.environ if environ is None else environ
    values = read_env_file(env_file)
    loaded_names: list[str] = []
    for name in CLEARML_ENV_NAMES:
        value = values.get(name, "").strip()
        if not value or target_environ.get(name):
            continue
        target_environ[name] = value
        loaded_names.append(name)
    return tuple(loaded_names)


def ensure_clearml_environment(
    env_file: Path = Path(".env"),
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """
    Ensure all required ClearML credential variables are available.

    Args:
        env_file: The dotenv file to read.
        environ: Optional environment mapping for tests.

    Raises:
        ClearMlConfigurationError: Raised when required variables are missing.
    """
    target_environ = os.environ if environ is None else environ
    load_clearml_environment(env_file, target_environ)
    missing_names = [
        name for name in CLEARML_ENV_NAMES if not target_environ.get(name, "").strip()
    ]
    if missing_names:
        joined_names = ", ".join(missing_names)
        raise ClearMlConfigurationError(
            f"missing ClearML configuration variables: {joined_names}"
        )


def initialize_clearml_task(
    is_enabled: bool,
    task_name: str,
    project_name: str = DEFAULT_CLEARML_PROJECT,
    task_type: str | None = None,
    tags: Sequence[str] = (),
    configuration: Mapping[str, Any] | None = None,
    env_file: Path = Path(".env"),
    environ: MutableMapping[str, str] | None = None,
) -> ClearMlTracker:
    """
    Initialize a ClearML task or return a disabled tracker.

    Args:
        is_enabled: Whether ClearML tracking was requested.
        task_name: The ClearML task name.
        project_name: The ClearML project name.
        task_type: Optional ClearML task type.
        tags: Optional ClearML task tags.
        configuration: Optional configuration dictionary to connect.
        env_file: The dotenv file to read.
        environ: Optional environment mapping for tests.

    Returns:
        A ClearML tracker wrapper.

    Raises:
        ClearMlConfigurationError: Raised when ClearML cannot be initialized.
    """
    if not is_enabled:
        return ClearMlTracker()
    ensure_clearml_environment(env_file, environ)
    try:
        from clearml import Task
    except ImportError as error:
        raise ClearMlConfigurationError(
            "clearml package is required when --clearml is supplied"
        ) from error
    force_clearml_requirements_env_freeze(Task)
    task: Any = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        tags=list(tags),
        reuse_last_task_id=False,
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True,
        auto_resource_monitoring=True,
        auto_connect_streams=True,
    )
    clear_task_script_diff(task)
    if configuration is not None:
        task.connect(dict(configuration), name="configuration")
    return ClearMlTracker(task=task)


def force_clearml_requirements_env_freeze(task_class: Any) -> None:
    """
    Force ClearML to capture the full active Python environment.

    Args:
        task_class: The ClearML Task class.

    Raises:
        ClearMlConfigurationError: Raised when requirements capture cannot be set.
    """
    try:
        task_class.force_requirements_env_freeze(True)
    except Exception as error:
        raise ClearMlConfigurationError(
            "failed to configure ClearML Python package capture"
        ) from error


def clear_task_script_diff(task: Any) -> None:
    """
    Clear ClearML's captured git diff from a task.

    Args:
        task: A ClearML task-like object.

    Raises:
        ClearMlConfigurationError: Raised when the diff cannot be cleared.
    """
    try:
        wait_for_repo_detection = getattr(task, "_wait_for_repo_detection", None)
        if callable(wait_for_repo_detection):
            wait_for_repo_detection(timeout=30.0)
        task.set_script(diff="")
    except Exception as error:
        raise ClearMlConfigurationError(
            "failed to clear ClearML task git diff"
        ) from error


def publish_clearml_dataset(
    data_dir: Path,
    dataset_name: str,
    dataset_version: str,
    project_name: str,
    split_counts: Mapping[str, int],
    env_file: Path = Path(".env"),
) -> str:
    """
    Publish requirement model JSONL files as a ClearML Dataset.

    Args:
        data_dir: The directory containing JSONL split files.
        dataset_name: The ClearML dataset name.
        dataset_version: The ClearML dataset version.
        project_name: The ClearML dataset project.
        split_counts: Split counts recorded in the dataset description.
        env_file: The dotenv file to read.

    Returns:
        The ClearML dataset identifier when available.

    Raises:
        ClearMlConfigurationError: Raised when ClearML cannot be initialized.
    """
    ensure_clearml_environment(env_file)
    try:
        from clearml import Dataset
    except ImportError as error:
        raise ClearMlConfigurationError(
            "clearml package is required when --clearml is supplied"
        ) from error
    description = build_dataset_description(split_counts)
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=project_name,
        dataset_version=dataset_version,
        description=description,
    )
    dataset.add_files(path=str(data_dir), wildcard="*.jsonl", recursive=False)
    dataset.upload()
    dataset.finalize()
    return str(getattr(dataset, "id", ""))


def build_dataset_description(split_counts: Mapping[str, int]) -> str:
    """
    Build a short ClearML Dataset description from split counts.

    Args:
        split_counts: Record counts keyed by split name.

    Returns:
        A human-readable dataset description.
    """
    counts_text = ", ".join(
        f"{split_name}={split_count}"
        for split_name, split_count in sorted(split_counts.items())
    )
    return f"Requirement model JSONL splits: {counts_text}"


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
