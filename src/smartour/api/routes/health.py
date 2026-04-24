"""Health check routes for the Smartour API."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """
    Response model for the API health check.
    """

    status: str
    service: str


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """
    Return the application health status.

    Returns:
        A simple health response.
    """
    return HealthResponse(status="ok", service="smartour")
