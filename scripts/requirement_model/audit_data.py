"""Audit generated requirement model JSONL data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.clearml_tracking import (
    DEFAULT_CLEARML_PROJECT,
    ClearMlTracker,
    initialize_clearml_task,
    publish_clearml_dataset,
)
from scripts.requirement_model.generate_data import (
    GENERATED_SPLITS,
    validate_split_records,
)
from scripts.requirement_model.schema import load_jsonl

DEFAULT_DATA_DIR = Path("data/requirement_model")
DEFAULT_CLEARML_DATASET_NAME = "requirement_model_data"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Audit requirement model JSONL splits."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reviewed-test", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--clearml-project", default=DEFAULT_CLEARML_PROJECT)
    parser.add_argument(
        "--clearml-task-name",
        default="Requirement Model Data Audit",
    )
    parser.add_argument(
        "--clearml-dataset-name",
        default=DEFAULT_CLEARML_DATASET_NAME,
    )
    parser.add_argument("--clearml-dataset-version", default=None)
    return parser.parse_args()


def audit_data(data_dir: Path, include_reviewed_test: bool) -> dict[str, int]:
    """
    Audit generated data splits.

    Args:
        data_dir: The dataset directory.
        include_reviewed_test: Whether to include reviewed test records.

    Returns:
        Split record counts.
    """
    split_names = list(GENERATED_SPLITS)
    if include_reviewed_test:
        split_names.append("reviewed_test")
    split_records_by_name = {
        split_name: load_jsonl(data_dir / f"{split_name}.jsonl")
        for split_name in split_names
    }
    return validate_split_records(split_records_by_name)


def default_dataset_version() -> str:
    """
    Return a timestamp-based default ClearML Dataset version.

    Returns:
        The dataset version string.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def report_audit_to_clearml(
    tracker: ClearMlTracker,
    split_counts: dict[str, int],
    data_dir: Path,
    project_name: str,
    dataset_name: str,
    dataset_version: str,
) -> None:
    """
    Report audit counts and publish audited JSONL data to ClearML.

    Args:
        tracker: The optional ClearML tracker.
        split_counts: Record counts by split name.
        data_dir: Directory containing audited JSONL split files.
        project_name: The ClearML project name.
        dataset_name: The ClearML dataset name.
        dataset_version: The ClearML dataset version.
    """
    if not tracker.is_enabled:
        return
    for index, split_name in enumerate(sorted(split_counts)):
        tracker.report_scalar(
            title="records",
            series=split_name,
            value=float(split_counts[split_name]),
            iteration=index,
        )
    tracker.upload_artifact("split_counts", dict(split_counts))
    dataset_id = publish_clearml_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        project_name=project_name,
        split_counts=split_counts,
    )
    tracker.upload_artifact(
        "dataset",
        {
            "id": dataset_id,
            "name": dataset_name,
            "project": project_name,
            "version": dataset_version,
        },
    )


def main() -> None:
    """
    Run the dataset audit and print split counts.
    """
    args = parse_args()
    dataset_version = args.clearml_dataset_version or default_dataset_version()
    tracker = initialize_clearml_task(
        is_enabled=args.clearml,
        task_name=args.clearml_task_name,
        project_name=args.clearml_project,
        task_type="data_processing",
        tags=("requirement_model", "data_audit"),
        configuration={
            "data_dir": str(args.data_dir),
            "reviewed_test": args.reviewed_test,
            "strict": args.strict,
            "dataset_name": args.clearml_dataset_name,
            "dataset_version": dataset_version,
        },
    )
    try:
        split_counts = audit_data(args.data_dir, args.reviewed_test)
        report_audit_to_clearml(
            tracker=tracker,
            split_counts=split_counts,
            data_dir=args.data_dir,
            project_name=args.clearml_project,
            dataset_name=args.clearml_dataset_name,
            dataset_version=dataset_version,
        )
        counts_text = ", ".join(
            f"{split_name}={split_count}"
            for split_name, split_count in sorted(split_counts.items())
        )
        strict_text = " strict" if args.strict else ""
        print(f"audited{strict_text} requirement model data: {counts_text}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
