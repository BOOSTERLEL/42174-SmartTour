"""User dashboard API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends

from smartour.api.dependencies import get_user_dashboard_service, require_current_user
from smartour.application.user_dashboard_service import (
    UserDashboardResponse,
    UserDashboardService,
)
from smartour.domain.user import User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me/dashboard", response_model=UserDashboardResponse)
async def get_current_user_dashboard(
    current_user: Annotated[User, Depends(require_current_user)],
    dashboard_service: Annotated[
        UserDashboardService, Depends(get_user_dashboard_service)
    ],
) -> UserDashboardResponse:
    """
    Return dashboard data for the authenticated user.

    Args:
        current_user: The authenticated user.
        dashboard_service: The user dashboard service.

    Returns:
        The current user's dashboard data.
    """
    return await dashboard_service.get_dashboard(current_user.id)
