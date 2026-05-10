"""Tests for requirement model evaluation decoding and metrics."""

from __future__ import annotations

from scripts.requirement_model.evaluate import compute_metrics, decode_slots
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
