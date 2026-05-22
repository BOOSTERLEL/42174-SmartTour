"""Admin dashboard API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from smartour.api.dependencies import get_admin_service, require_admin_user
from smartour.application.admin_service import AdminService, AdminStatsResponse
from smartour.domain.user import User

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    admin_service: Annotated[AdminService, Depends(get_admin_service)],
    current_user: Annotated[User, Depends(require_admin_user)],
    window_hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> AdminStatsResponse:
    """
    Return backend statistics for the admin dashboard.

    Args:
        admin_service: The admin statistics service.
        current_user: The authenticated admin user.
        window_hours: Google Maps cost-summary lookback window in hours.

    Returns:
        The admin dashboard statistics.
    """
    return await admin_service.get_stats(window_hours=window_hours)
