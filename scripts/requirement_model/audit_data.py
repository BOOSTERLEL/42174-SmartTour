"""Audit generated requirement model JSONL data."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

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
    REQUIRED_SLOT_FIELDS,
    validate_split_records,
)
from scripts.requirement_model.schema import RequirementTrainingRecord, load_jsonl

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
    parser.add_argument("--clearml-report-data-profile", action="store_true")
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
    split_records_by_name = load_split_records(data_dir, include_reviewed_test)
    return validate_split_records(split_records_by_name)


def load_split_records(
    data_dir: Path, include_reviewed_test: bool
) -> dict[str, list[RequirementTrainingRecord]]:
    """
    Load generated requirement model splits.

    Args:
        data_dir: The dataset directory.
        include_reviewed_test: Whether to include reviewed test records.

    Returns:
        Records keyed by split name.
    """
    split_names = list(GENERATED_SPLITS)
    if include_reviewed_test:
        split_names.append("reviewed_test")
    return {
        split_name: load_jsonl(data_dir / f"{split_name}.jsonl")
        for split_name in split_names
    }


def build_data_profile(
    split_records_by_name: Mapping[str, list[RequirementTrainingRecord]],
) -> dict[str, Any]:
    """
    Build deterministic profile statistics for requirement model data.

    Args:
        split_records_by_name: Records keyed by split name.

    Returns:
        JSON-compatible data profile statistics.
    """
    split_summaries: dict[str, dict[str, float | int]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    slot_coverage: dict[str, dict[str, int]] = {}
    token_lengths: dict[str, list[int]] = {}
    text_lengths: dict[str, list[int]] = {}
    for split_name, records in sorted(split_records_by_name.items()):
        split_token_lengths = [len(record.tokens) for record in records]
        split_text_lengths = [len(record.text) for record in records]
        token_lengths[split_name] = split_token_lengths
        text_lengths[split_name] = split_text_lengths
        split_summaries[split_name] = {
            "records": len(records),
            "average_tokens": average(split_token_lengths),
            "max_tokens": max(split_token_lengths, default=0),
            "average_characters": average(split_text_lengths),
            "max_characters": max(split_text_lengths, default=0),
        }
        label_counter: Counter[str] = Counter()
        slot_counter: Counter[str] = Counter()
        for record in records:
            label_counter.update(record.labels)
            slot_values = record.slots.model_dump()
            for field_name in REQUIRED_SLOT_FIELDS:
                value = slot_values[field_name]
                if value is None or value == []:
                    continue
                slot_counter[field_name] += 1
        label_counts[split_name] = dict(sorted(label_counter.items()))
        slot_coverage[split_name] = {
            field_name: slot_counter[field_name]
            for field_name in REQUIRED_SLOT_FIELDS
        }
    return {
        "split_summaries": split_summaries,
        "label_counts": label_counts,
        "slot_coverage": slot_coverage,
        "token_lengths": token_lengths,
        "text_lengths": text_lengths,
    }


def average(values: list[int]) -> float:
    """
    Return the arithmetic mean for integer values.

    Args:
        values: The values to average.

    Returns:
        The average, or 0.0 for an empty list.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


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


def report_data_profile_to_clearml(
    tracker: ClearMlTracker,
    data_profile: Mapping[str, Any],
) -> None:
    """
    Report data profile visualizations to ClearML.

    Args:
        tracker: The optional ClearML tracker.
        data_profile: Data profile statistics.
    """
    if not tracker.is_enabled:
        return
    tracker.report_table(
        title="data_profile",
        series="split_summary",
        rows=build_split_summary_rows(data_profile["split_summaries"]),
    )
    tracker.report_table(
        title="data_profile",
        series="slot_coverage",
        rows=build_nested_count_rows(data_profile["slot_coverage"], "slot"),
    )
    tracker.report_table(
        title="data_profile",
        series="label_distribution",
        rows=build_nested_count_rows(data_profile["label_counts"], "label"),
    )
    for split_name, values in sorted(data_profile["token_lengths"].items()):
        tracker.report_histogram(
            title="token_lengths",
            series=split_name,
            values=values,
            xaxis="tokens",
            yaxis="records",
        )
    for split_name, values in sorted(data_profile["text_lengths"].items()):
        tracker.report_histogram(
            title="text_lengths",
            series=split_name,
            values=values,
            xaxis="characters",
            yaxis="records",
        )
    tracker.upload_artifact("data_profile", dict(data_profile))


def build_split_summary_rows(
    split_summaries: Mapping[str, Mapping[str, float | int]],
) -> list[list[str | float | int]]:
    """
    Build table rows for split summary statistics.

    Args:
        split_summaries: Summary values keyed by split name.

    Returns:
        Table rows including a header row.
    """
    rows: list[list[str | float | int]] = [
        [
            "split",
            "records",
            "average_tokens",
            "max_tokens",
            "average_characters",
            "max_characters",
        ]
    ]
    for split_name, summary in sorted(split_summaries.items()):
        rows.append(
            [
                split_name,
                summary["records"],
                round(float(summary["average_tokens"]), 2),
                summary["max_tokens"],
                round(float(summary["average_characters"]), 2),
                summary["max_characters"],
            ]
        )
    return rows


def build_nested_count_rows(
    counts_by_split: Mapping[str, Mapping[str, int]], value_name: str
) -> list[list[str | int]]:
    """
    Build table rows for split-scoped count dictionaries.

    Args:
        counts_by_split: Counts keyed by split and value.
        value_name: Header name for the counted value column.

    Returns:
        Table rows including a header row.
    """
    rows: list[list[str | int]] = [["split", value_name, "count"]]
    for split_name, counts in sorted(counts_by_split.items()):
        for name, count in sorted(counts.items()):
            rows.append([split_name, name, count])
    return rows


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
            "report_data_profile": args.clearml_report_data_profile,
        },
    )
    try:
        split_records_by_name = load_split_records(args.data_dir, args.reviewed_test)
        split_counts = validate_split_records(split_records_by_name)
        report_audit_to_clearml(
            tracker=tracker,
            split_counts=split_counts,
            data_dir=args.data_dir,
            project_name=args.clearml_project,
            dataset_name=args.clearml_dataset_name,
            dataset_version=dataset_version,
        )
        if args.clearml_report_data_profile:
            report_data_profile_to_clearml(
                tracker,
                build_data_profile(split_records_by_name),
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
