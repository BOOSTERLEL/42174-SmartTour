"""Tests for requirement model evaluation decoding and metrics."""

from __future__ import annotations

import torch

from scripts.requirement_model.evaluate import (
    build_failure_rows,
    compute_confusion_matrix,
    compute_label_metric_rows,
    compute_metrics,
    compute_slot_accuracy_rows,
    decode_slots,
    move_encoding_to_device,
)
from scripts.requirement_model.schema import RequirementSlots


def test_decode_slots_uses_runtime_canonicalization() -> None:
    """
    Verify generated surface phrases decode to canonical slot values.
    """
    premium_slots = decode_slots(["premium"], ["B-BUDGET_LEVEL"])
    assert premium_slots.budget_level == "high"

    tokens = [
        "medium",
        "budget",
        "public",
        "transportation",
        "parks",
        "restaurants",
        "family",
        "activities",
        "historic",
        "sites",
    ]
    labels = [
        "B-BUDGET_LEVEL",
        "I-BUDGET_LEVEL",
        "B-TRANSPORTATION_MODE",
        "I-TRANSPORTATION_MODE",
        "B-INTEREST",
        "B-INTEREST",
        "B-INTEREST",
        "I-INTEREST",
        "B-INTEREST",
        "I-INTEREST",
    ]

    slots = decode_slots(tokens, labels)

    assert slots.budget_level == "medium"
    assert slots.transportation_mode == "transit"
    assert slots.interests == ["nature", "food", "family", "history"]


def test_compute_metrics_normalizes_list_field_order() -> None:
    """
    Verify list slot order does not reduce field or exact-match accuracy.
    """
    gold_slots = RequirementSlots(interests=["food", "nature"])
    predicted_slots = RequirementSlots(interests=["nature", "food"])

    metrics = compute_metrics(
        [gold_slots],
        [predicted_slots],
        [["O"]],
        [["O"]],
    )

    assert metrics["slot_accuracy"] == 1.0
    assert metrics["exact_match_accuracy"] == 1.0


def test_compute_confusion_matrix_counts_bio_pairs() -> None:
    """
    Verify BIO confusion matrices count gold and predicted labels.
    """
    matrix = compute_confusion_matrix(
        [["B-DESTINATION", "O"]],
        [["B-DESTINATION", "B-ADULTS"]],
    )

    assert matrix[1][1] == 1
    assert matrix[0][7] == 1


def test_compute_label_metric_rows_reports_precision_recall_f1() -> None:
    """
    Verify per-label metrics expose precision, recall, and support.
    """
    rows = compute_label_metric_rows(
        [["B-DESTINATION", "B-ADULTS"]],
        [["B-DESTINATION", "O"]],
    )
    rows_by_label = {str(row["label"]): row for row in rows}

    assert rows_by_label["B-DESTINATION"]["f1"] == 1.0
    assert rows_by_label["B-ADULTS"]["recall"] == 0.0
    assert rows_by_label["B-ADULTS"]["support"] == 1


def test_compute_slot_accuracy_rows_reports_mismatches() -> None:
    """
    Verify per-slot accuracy identifies field-level mismatches.
    """
    rows = compute_slot_accuracy_rows(
        [RequirementSlots(destination="Tokyo")],
        [RequirementSlots(destination="Paris")],
    )
    rows_by_field = {str(row["field"]): row for row in rows}

    assert rows_by_field["destination"]["accuracy"] == 0.0
    assert rows_by_field["destination"]["mismatches"] == 1


def test_build_failure_rows_records_mismatch_details() -> None:
    """
    Verify failed example rows include mismatch fields and serialized slots.
    """

    class Record:
        """
        Minimal record test double.
        """

        text = "Plan Tokyo."

    rows = build_failure_rows(
        [Record()],
        [RequirementSlots(destination="Tokyo")],
        [RequirementSlots(destination="Paris")],
    )

    assert rows[0]["text"] == "Plan Tokyo."
    assert rows[0]["mismatch_fields"] == "destination"
    assert "Tokyo" in rows[0]["gold_slots"]
    assert "Paris" in rows[0]["predicted_slots"]


def test_move_encoding_to_device_keeps_cpu_tensors_when_device_is_none() -> None:
    """
    Verify evaluation helpers can keep tokenizer outputs on their current device.
    """
    tensor = torch.tensor([[1, 2]])

    moved_encoding = move_encoding_to_device({"input_ids": tensor}, None)

    assert moved_encoding["input_ids"] is tensor
