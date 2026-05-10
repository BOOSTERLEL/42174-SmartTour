"""Audit generated requirement model JSONL data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.requirement_model.generate_data import (
    GENERATED_SPLITS,
    validate_split_records,
)
from scripts.requirement_model.schema import load_jsonl

DEFAULT_DATA_DIR = Path("data/requirement_model")


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


def main() -> None:
    """
    Run the dataset audit and print split counts.
    """
    args = parse_args()
    split_counts = audit_data(args.data_dir, args.reviewed_test)
    counts_text = ", ".join(
        f"{split_name}={split_count}"
        for split_name, split_count in sorted(split_counts.items())
    )
    strict_text = " strict" if args.strict else ""
    print(f"audited{strict_text} requirement model data: {counts_text}")


if __name__ == "__main__":
    main()
