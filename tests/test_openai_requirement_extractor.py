"""Tests for OpenAI requirement extraction adapters."""

from smartour.domain.requirement import TravelRequirementUpdate
from smartour.integrations.openai.requirement_extractor import (
    HybridRequirementExtractor,
    OpenAIRequirementExtraction,
)


class FailingExtractor:
    """
    Test extractor that always fails.
    """

    def extract(self, message: str) -> TravelRequirementUpdate:
        """
        Raise an extraction failure.

        Args:
            message: The raw user message.

        Raises:
            RuntimeError: Always raised for fallback verification.
        """
        raise RuntimeError(message)


class StaticExtractor:
    """
    Test extractor that returns a static update.
    """

    def extract(self, message: str) -> TravelRequirementUpdate:
        """
        Return a fixed requirement update.

        Args:
            message: The raw user message.

        Returns:
            A fixed requirement update.
        """
        return TravelRequirementUpdate(destination="Sydney")


def test_openai_extraction_converts_to_requirement_update() -> None:
    """
    Verify that OpenAI structured output maps to the domain update model.
    """
    extraction = OpenAIRequirementExtraction(
        destination="Tokyo",
        trip_dates=None,
        trip_length_days=4,
        adults=2,
        children=1,
        budget_level="medium",
        travel_pace="relaxed",
        interests=["food", "museums"],
        hotel_area="Shinjuku",
        transportation_mode="transit",
        food_preferences=["ramen"],
        language="en",
    )

    update = extraction.to_requirement_update()

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
    assert update.language == "en"


def test_hybrid_extractor_uses_fallback_when_primary_fails() -> None:
    """
    Verify that the hybrid extractor keeps requirement collection available.
    """
    extractor = HybridRequirementExtractor(
        primary_extractor=FailingExtractor(),
        fallback_extractor=StaticExtractor(),
    )

    update = extractor.extract("message")

    assert update.destination == "Sydney"
