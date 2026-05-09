"""Tests for the supervised requirement model extractor."""

from pathlib import Path

from smartour.integrations.requirement_model.decoder import (
    RequirementSlotSpan,
    TokenPrediction,
    decode_bio_spans,
)
from smartour.integrations.requirement_model.extractor import RequirementModelExtractor
from smartour.integrations.requirement_model.normalizer import (
    spans_to_requirement_update,
)


class StaticRequirementModelExtractor(RequirementModelExtractor):
    """
    Requirement model extractor that returns fixed spans for tests.
    """

    def __init__(self, spans: list[RequirementSlotSpan]) -> None:
        """
        Initialize the static test extractor.

        Args:
            spans: The fixed spans to return.
        """
        super().__init__(model_path=Path("missing-test-model"))
        self.spans = spans

    def predict_spans(self, message: str) -> list[RequirementSlotSpan]:
        """
        Return fixed spans without loading a model.

        Args:
            message: The raw user message.

        Returns:
            The fixed spans.
        """
        return self.spans


def test_decode_bio_spans_repairs_invalid_i_label() -> None:
    """
    Verify that an I label without an active span starts a valid span.
    """
    spans = decode_bio_spans(
        "Tokyo relaxed",
        [
            TokenPrediction("I-DESTINATION", 0, 5, 0.91),
            TokenPrediction("O", 5, 6, 0.99),
            TokenPrediction("B-TRAVEL_PACE", 6, 13, 0.94),
        ],
        confidence_threshold=0.5,
    )

    assert spans == [
        RequirementSlotSpan("DESTINATION", "Tokyo", 0.91),
        RequirementSlotSpan("TRAVEL_PACE", "relaxed", 0.94),
    ]


def test_decode_bio_spans_omits_low_confidence_span() -> None:
    """
    Verify that low-confidence model predictions are not returned.
    """
    spans = decode_bio_spans(
        "Tokyo",
        [TokenPrediction("B-DESTINATION", 0, 5, 0.2)],
        confidence_threshold=0.5,
    )

    assert spans == []


def test_spans_to_requirement_update_normalizes_supported_slots() -> None:
    """
    Verify that decoded spans map into the canonical requirement update model.
    """
    update = spans_to_requirement_update(
        [
            RequirementSlotSpan("DESTINATION", "Tokyo", 0.9),
            RequirementSlotSpan("TRIP_LENGTH_DAYS", "4 days", 0.9),
            RequirementSlotSpan("ADULTS", "2 adults", 0.9),
            RequirementSlotSpan("CHILDREN", "1 child", 0.9),
            RequirementSlotSpan("BUDGET_LEVEL", "moderate", 0.9),
            RequirementSlotSpan("TRAVEL_PACE", "relaxed", 0.9),
            RequirementSlotSpan("INTEREST", "food", 0.9),
            RequirementSlotSpan("INTEREST", "博物馆", 0.9),
            RequirementSlotSpan("HOTEL_AREA", "Shinjuku", 0.9),
            RequirementSlotSpan("TRANSPORTATION_MODE", "subway", 0.9),
            RequirementSlotSpan("FOOD_PREFERENCE", "ramen", 0.9),
            RequirementSlotSpan("LANGUAGE", "Chinese", 0.9),
        ]
    )

    assert update.destination == "Tokyo"
    assert update.trip_length_days == 4
    assert update.travelers is not None
    assert update.travelers.adults == 2
    assert update.travelers.children == 1
    assert update.budget_level == "medium"
    assert update.travel_pace == "relaxed"
    assert update.interests == ["food", "museums"]
    assert update.hotel_area == "Shinjuku"
    assert update.transportation_mode == "transit"
    assert update.food_preferences == ["ramen"]
    assert update.language == "zh"


def test_spans_to_requirement_update_omits_unknown_canonical_values() -> None:
    """
    Verify that unsupported scalar values do not corrupt requirement state.
    """
    update = spans_to_requirement_update(
        [
            RequirementSlotSpan("BUDGET_LEVEL", "surprising", 0.9),
            RequirementSlotSpan("TRAVEL_PACE", "chaotic", 0.9),
        ]
    )

    assert update.budget_level is None
    assert update.travel_pace is None


def test_requirement_model_extractor_preserves_contract() -> None:
    """
    Verify that the extractor returns a TravelRequirementUpdate.
    """
    extractor = StaticRequirementModelExtractor(
        [
            RequirementSlotSpan("DESTINATION", "Sydney", 0.9),
            RequirementSlotSpan("TRANSPORTATION_MODE", "public transit", 0.9),
        ]
    )

    update = extractor.extract("message")

    assert update.destination == "Sydney"
    assert update.transportation_mode == "transit"
