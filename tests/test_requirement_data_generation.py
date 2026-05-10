"""Tests for requirement data generation parsing and validation."""

from __future__ import annotations

import pytest

from scripts.requirement_model.audit_data import build_data_profile
from scripts.requirement_model.generate_data import (
    DataGenerationError,
    deduplicate_records,
    generate_records,
    make_record_from_marked_text,
    validate_english_records,
    validate_split_records,
)


def test_make_record_from_marked_text_builds_bio_labels() -> None:
    """
    Verify that inline markers become plain text, slots, and BIO labels.
    """
    record = make_record_from_marked_text(
        "Plan [TRIP_LENGTH_DAYS|4 days] in [DESTINATION|Tokyo] "
        "for [ADULTS|2 adults] with [CHILDREN|no children]. Use "
        "[TRANSPORTATION_MODE|public transportation], stay in "
        "[HOTEL_AREA|city center], keep it [BUDGET_LEVEL|moderate], "
        "make it [TRAVEL_PACE|relaxed], include [INTEREST|museums] "
        "and [FOOD_PREFERENCE|ramen], and write in [LANGUAGE|English]."
    )

    assert "[" not in record.text
    assert record.slots.destination == "Tokyo"
    assert record.slots.trip_length_days == 4
    assert record.slots.adults == 2
    assert record.slots.children == 0
    assert record.slots.transportation_mode == "transit"
    assert record.slots.hotel_area == "city center"
    assert record.slots.budget_level == "medium"
    assert record.slots.travel_pace == "relaxed"
    assert record.slots.interests == ["museums"]
    assert record.slots.food_preferences == ["ramen"]
    assert record.slots.language == "en"
    assert "B-DESTINATION" in record.labels
    assert len(record.tokens) == len(record.labels)


def test_make_record_from_marked_text_rejects_bad_marker() -> None:
    """
    Verify that malformed or unsupported markers are rejected.
    """
    with pytest.raises(DataGenerationError, match="unsupported slot marker"):
        make_record_from_marked_text("Plan [CITY|Tokyo].")

    with pytest.raises(DataGenerationError, match="malformed marker syntax"):
        make_record_from_marked_text("Plan [DESTINATION|Tokyo.")


def test_make_record_from_marked_text_rejects_non_english_text() -> None:
    """
    Verify that marked candidates with Chinese text are rejected.
    """
    with pytest.raises(ValueError, match="English-only"):
        make_record_from_marked_text("Plan [DESTINATION|Tokyo]，谢谢。")


def test_deduplicate_records_removes_normalized_duplicates() -> None:
    """
    Verify that exact normalized duplicate texts are removed.
    """
    first_record = make_record_from_marked_text("Plan [DESTINATION|Tokyo].")
    second_record = make_record_from_marked_text("  plan [DESTINATION|Tokyo].  ")

    records = deduplicate_records([first_record, second_record])

    assert records == [first_record]


def test_validate_split_records_catches_train_test_leakage() -> None:
    """
    Verify that strict split validation rejects duplicate text across splits.
    """
    train_record = make_record_from_marked_text("Plan [DESTINATION|Tokyo].")
    test_record = make_record_from_marked_text("plan [DESTINATION|Tokyo].")

    with pytest.raises(ValueError, match="duplicate record text"):
        validate_split_records(
            {
                "train": [train_record],
                "test": [test_record],
                "reviewed_test": [],
            }
        )


def test_validate_split_records_rejects_unsupported_canonical_values() -> None:
    """
    Verify strict validation rejects unsupported canonical slot values.
    """
    record = make_record_from_marked_text("Plan [DESTINATION|Tokyo].")
    bad_record = record.model_copy(
        update={"slots": record.slots.model_copy(update={"interests": ["park"]})}
    )

    with pytest.raises(ValueError, match="unsupported interests value: park"):
        validate_split_records({"train": [bad_record]})


def test_validate_split_records_allows_open_food_preferences() -> None:
    """
    Verify food preferences can keep open English surface values.
    """
    record = make_record_from_marked_text("Plan [DESTINATION|Tokyo].")
    food_record = record.model_copy(
        update={
            "slots": record.slots.model_copy(update={"food_preferences": ["gelato"]})
        }
    )

    validate_english_records("train", [food_record])


def test_generate_records_supports_english_only_language() -> None:
    """
    Verify that template generation only accepts English.
    """
    assert generate_records(2, 42174, "en")

    with pytest.raises(ValueError, match="only English"):
        generate_records(1, 42174, "zh")


def test_build_data_profile_reports_lengths_labels_and_slots() -> None:
    """
    Verify data profile statistics summarize records deterministically.
    """
    first_record = make_record_from_marked_text(
        "Plan [DESTINATION|Tokyo] for [ADULTS|2 adults]."
    )
    second_record = make_record_from_marked_text(
        "Need [TRIP_LENGTH_DAYS|4 days] in [DESTINATION|Paris]."
    )

    profile = build_data_profile({"train": [first_record, second_record]})

    summary = profile["split_summaries"]["train"]
    assert summary["records"] == 2
    assert summary["max_tokens"] >= summary["average_tokens"]
    assert profile["label_counts"]["train"]["B-DESTINATION"] == 2
    assert profile["slot_coverage"]["train"]["destination"] == 2
    assert profile["slot_coverage"]["train"]["adults"] == 1
    assert len(profile["token_lengths"]["train"]) == 2
    assert len(profile["text_lengths"]["train"]) == 2
