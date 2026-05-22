"""Tests for itinerary share links."""

import pytest
from fastapi import HTTPException, status

from smartour.api.routes.shares import (
    create_itinerary_share_link,
    get_shared_itinerary,
)
from smartour.application.report_service import ReportService
from smartour.application.share_service import ShareService
from smartour.domain.itinerary import Itinerary, ItineraryDay
from smartour.infrastructure.repositories.itineraries import InMemoryItineraryRepository
from smartour.infrastructure.repositories.shares import InMemoryItineraryShareRepository


@pytest.mark.asyncio
async def test_share_service_creates_opaque_share_link() -> None:
    """
    Verify that share links are opaque and resolve to a shared itinerary.
    """
    itinerary_repository = InMemoryItineraryRepository()
    share_repository = InMemoryItineraryShareRepository()
    itinerary = _itinerary()
    await itinerary_repository.save(itinerary)
    report_service = ReportService(itinerary_repository)
    share_service = ShareService(
        itinerary_repository=itinerary_repository,
        share_repository=share_repository,
        report_service=report_service,
    )

    share_link = await share_service.create_share_link(itinerary.id)
    assert share_link is not None
    assert share_link.itinerary_id == itinerary.id
    assert share_link.token not in itinerary.id
    assert itinerary.id not in share_link.token
    assert share_link.share_path == f"/share/{share_link.token}"

    shared_itinerary = await share_service.get_shared_itinerary(share_link.token)

    assert shared_itinerary is not None
    assert shared_itinerary.token == share_link.token
    assert shared_itinerary.itinerary.title == "Sydney Weekend"
    assert "# Sydney Weekend" in shared_itinerary.report.markdown


@pytest.mark.asyncio
async def test_share_route_returns_not_found_for_missing_itinerary() -> None:
    """
    Verify that missing itineraries cannot create share links.
    """
    share_service = _share_service()

    with pytest.raises(HTTPException) as error_info:
        await create_itinerary_share_link("missing", share_service)

    assert error_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert error_info.value.detail == "Itinerary not found"


@pytest.mark.asyncio
async def test_shared_itinerary_route_returns_not_found_for_unknown_token() -> None:
    """
    Verify that unknown share tokens return 404.
    """
    share_service = _share_service()

    with pytest.raises(HTTPException) as error_info:
        await get_shared_itinerary("unknown", share_service)

    assert error_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert error_info.value.detail == "Share link not found"


def _share_service() -> ShareService:
    """
    Create a share service with in-memory dependencies.

    Returns:
        The share service.
    """
    itinerary_repository = InMemoryItineraryRepository()
    return ShareService(
        itinerary_repository=itinerary_repository,
        share_repository=InMemoryItineraryShareRepository(),
        report_service=ReportService(itinerary_repository),
    )


def _itinerary() -> Itinerary:
    """
    Build a shareable itinerary.

    Returns:
        A generated itinerary fixture.
    """
    return Itinerary(
        conversation_id="conv_1",
        title="Sydney Weekend",
        destination_name="Sydney",
        days=[
            ItineraryDay(
                day_number=1,
                theme="harbour views",
                summary="Explore the harbour.",
            )
        ],
        guide_markdown="Bring comfortable shoes.",
    )
