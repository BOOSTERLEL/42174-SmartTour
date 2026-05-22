"""Itinerary report API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from smartour.api.dependencies import get_report_service
from smartour.application.report_service import ItineraryReport, ReportService

router = APIRouter(tags=["reports"])


@router.get("/itineraries/{itinerary_id}/report", response_model=ItineraryReport)
async def get_itinerary_report(
    itinerary_id: str,
    report_service: Annotated[ReportService, Depends(get_report_service)],
) -> ItineraryReport:
    """
    Return a generated Markdown report for an itinerary.

    Args:
        itinerary_id: The itinerary identifier.
        report_service: The report generation service.

    Returns:
        The generated itinerary report.
    """
    report = await report_service.generate_itinerary_report(itinerary_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Itinerary not found"
        )
    return report
