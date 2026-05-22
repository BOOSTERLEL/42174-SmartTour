"""Cost monitoring API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from smartour.api.dependencies import get_cost_monitoring_service, require_admin_user
from smartour.application.cost_monitoring_service import (
    CostMonitoringService,
    GoogleMapsCostSummary,
)
from smartour.domain.user import User

router = APIRouter(prefix="/costs", tags=["costs"])


@router.get("/google-maps", response_model=GoogleMapsCostSummary)
async def get_google_maps_cost_summary(
    cost_monitoring_service: Annotated[
        CostMonitoringService, Depends(get_cost_monitoring_service)
    ],
    current_user: Annotated[User, Depends(require_admin_user)],
    job_id: Annotated[str | None, Query()] = None,
    window_hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> GoogleMapsCostSummary:
    """
    Return a backend Google Maps usage and cost summary.

    Args:
        cost_monitoring_service: The cost monitoring service.
        current_user: The authenticated admin user.
        job_id: Optional itinerary job filter.
        window_hours: Lookback window in hours.

    Returns:
        A Google Maps usage and estimated cost summary.
    """
    return await cost_monitoring_service.get_google_maps_summary(
        job_id=job_id, window_hours=window_hours
    )
