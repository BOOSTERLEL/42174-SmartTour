"""Data schema shared by requirement model scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SLOT_NAMES: tuple[str, ...] = (
    "DESTINATION",
    "TRIP_DATES",
    "TRIP_LENGTH_DAYS",
    "ADULTS",
    "CHILDREN",
    "BUDGET_LEVEL",
    "TRAVEL_PACE",
    "INTEREST",
    "HOTEL_AREA",
    "TRANSPORTATION_MODE",
    "FOOD_PREFERENCE",
    "LANGUAGE",
)
LABEL_NAMES: tuple[str, ...] = ("O",) + tuple(
    f"{prefix}-{slot_name}" for slot_name in SLOT_NAMES for prefix in ("B", "I")
)
LABEL_TO_ID: dict[str, int] = {
    label_name: label_index for label_index, label_name in enumerate(LABEL_NAMES)
}
ID_TO_LABEL: dict[int, str] = {
    label_index: label_name for label_name, label_index in LABEL_TO_ID.items()
}
SLOT_TO_FIELD: dict[str, str] = {
    "DESTINATION": "destination",
    "TRIP_DATES": "trip_dates",
    "TRIP_LENGTH_DAYS": "trip_length_days",
    "ADULTS": "adults",
    "CHILDREN": "children",
    "BUDGET_LEVEL": "budget_level",
    "TRAVEL_PACE": "travel_pace",
    "INTEREST": "interests",
    "HOTEL_AREA": "hotel_area",
    "TRANSPORTATION_MODE": "transportation_mode",
    "FOOD_PREFERENCE": "food_preferences",
    "LANGUAGE": "language",
}
TEXT_TOKEN_PATTERN = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[\u4e00-\u9fff]|"
    r"[^\s]"
)


class RequirementSlots(BaseModel):
    """
    Canonical slot values for one travel requirement example.
    """

    model_config = ConfigDict(extra="forbid")

    destination: str | None = None
    trip_dates: str | None = None
    trip_length_days: int | None = Field(default=None, ge=1)
    adults: int | None = Field(default=None, ge=1)
    children: int | None = Field(default=None, ge=0)
    budget_level: str | None = None
    travel_pace: str | None = None
    interests: list[str] = Field(default_factory=list)
    hotel_area: str | None = None
    transportation_mode: str | None = None
    food_preferences: list[str] = Field(default_factory=list)
    language: str | None = None


class RequirementTrainingRecord(BaseModel):
    """
    One labeled token-classification example for requirement extraction.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    tokens: list[str] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    slots: RequirementSlots

    @model_validator(mode="after")
    def validate_token_labels(self) -> RequirementTrainingRecord:
        """
        Validate that token labels align with the tokens and known BIO label set.

        Returns:
            The validated training record.

        Raises:
            ValueError: Raised when labels do not align with tokens.
        """
        if len(self.tokens) != len(self.labels):
            raise ValueError("tokens and labels must have the same length")
        unknown_labels = sorted(set(self.labels).difference(LABEL_NAMES))
        if unknown_labels:
            joined_labels = ", ".join(unknown_labels)
            raise ValueError(f"unknown labels: {joined_labels}")
        return self


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize requirement text for synthetic label generation.

    Args:
        text: The source text.

    Returns:
        The token sequence.
    """
    return TEXT_TOKEN_PATTERN.findall(text)


def load_jsonl(path: Path) -> list[RequirementTrainingRecord]:
    """
    Load and validate requirement training records from a JSONL file.

    Args:
        path: The JSONL file path.

    Returns:
        The validated records.
    """
    records: list[RequirementTrainingRecord] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            records.append(RequirementTrainingRecord.model_validate_json(line))
    return records


def write_jsonl(path: Path, records: list[RequirementTrainingRecord]) -> None:
    """
    Write requirement training records to a JSONL file.

    Args:
        path: The JSONL output file path.
        records: The validated records to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(record.model_dump_json() + "\n")


def record_to_json_dict(record: RequirementTrainingRecord) -> dict[str, Any]:
    """
    Convert a record to a JSON-compatible dictionary.

    Args:
        record: The requirement training record.

    Returns:
        The JSON-compatible dictionary.
    """
    return json.loads(record.model_dump_json())
