"""
Health check router.
"""

from fastapi import APIRouter

from smarttour.config import get_settings

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
def health_check() -> dict[str, str]:
    """
    Return service health status and version.

    Returns:
        A health payload including status and version.
    """

    settings = get_settings()
    return {"status": "ok", "version": settings.app_version}
