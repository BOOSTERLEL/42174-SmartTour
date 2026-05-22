"""Tests for itinerary report generation."""

import pytest
from fastapi import HTTPException, status

from smartour.api.routes.reports import get_itinerary_report
from smartour.application.report_service import ReportService
from smartour.domain.itinerary import (
    Coordinates,
    Itinerary,
    ItineraryDay,
    ItineraryItem,
    ItineraryItemType,
    PlaceRecommendation,
    RouteLeg,
    RouteSummary,
)
from smartour.infrastructure.repositories.itineraries import InMemoryItineraryRepository


@pytest.mark.asyncio
async def test_report_service_generates_markdown_report() -> None:
    """
    Verify that persisted itineraries produce deterministic Markdown reports.
    """
    itinerary_repository = InMemoryItineraryRepository()
    itinerary = _itinerary()
    await itinerary_repository.save(itinerary)
    service = ReportService(itinerary_repository)

    report = await service.generate_itinerary_report(itinerary.id)

    assert report is not None
    assert report.itinerary_id == itinerary.id
    assert report.format == "markdown"
    assert "# Sydney Weekend" in report.markdown
    assert "Destination: Sydney" in report.markdown
    assert "## Stay" in report.markdown
    assert "[Central Hotel](https://maps.example/hotel)" in report.markdown
    assert "### Day 1 - harbour views (2026-06-01)" in report.markdown
    assert "- 10:00 Attraction: [Harbour Museum](https://maps.example/museum)" in (
        report.markdown
    )
    assert "Route: 1.2 km, 12 min, 1 legs by transit." in report.markdown
    assert "Bring comfortable shoes." in report.markdown


@pytest.mark.asyncio
async def test_report_route_returns_not_found_for_missing_itinerary() -> None:
    """
    Verify that the report route returns 404 for missing itineraries.
    """
    service = ReportService(InMemoryItineraryRepository())

    with pytest.raises(HTTPException) as error_info:
        await get_itinerary_report("missing", service)

    assert error_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert error_info.value.detail == "Itinerary not found"


def _itinerary() -> Itinerary:
    """
    Build a reportable itinerary.

    Returns:
        A generated itinerary fixture.
    """
    hotel = PlaceRecommendation(
        place_id="hotel_1",
        name="Central Hotel",
        category="hotel",
        address="1 Harbour St",
        google_maps_uri="https://maps.example/hotel",
        rating=4.5,
        user_rating_count=200,
        price_level="PRICE_LEVEL_MODERATE",
    )
    museum = PlaceRecommendation(
        place_id="museum_1",
        name="Harbour Museum",
        category="attraction",
        location=Coordinates(latitude=-33.86, longitude=151.21),
        google_maps_uri="https://maps.example/museum",
    )
    day = ItineraryDay(
        day_number=1,
        date="2026-06-01",
        theme="harbour views",
        summary="Explore the harbour and museum district.",
        items=[
            ItineraryItem(
                time="10:00",
                type=ItineraryItemType.ATTRACTION,
                place=museum,
                duration_minutes=90,
            )
        ],
        route=RouteSummary(
            travel_mode="TRANSIT",
            distance_meters=1200,
            duration_seconds=720,
            legs=[
                RouteLeg(
                    origin_place_id="hotel_1",
                    destination_place_id="museum_1",
                    travel_mode="TRANSIT",
                    distance_meters=1200,
                    duration_seconds=720,
                )
            ],
        ),
    )
    return Itinerary(
        conversation_id="conv_1",
        title="Sydney Weekend",
        destination_name="Sydney",
        hotels=[hotel],
        days=[day],
        guide_markdown="Bring comfortable shoes.",
    )
