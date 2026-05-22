"""Itinerary share-link API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from smartour.api.dependencies import get_current_user, get_share_service
from smartour.application.share_service import (
    SharedItineraryResponse,
    ShareLinkResponse,
    ShareService,
)
from smartour.domain.user import User

router = APIRouter(tags=["shares"])


@router.post(
    "/itineraries/{itinerary_id}/share-links",
    response_model=ShareLinkResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_itinerary_share_link(
    itinerary_id: str,
    share_service: Annotated[ShareService, Depends(get_share_service)],
    current_user: Annotated[User | None, Depends(get_current_user)],
) -> ShareLinkResponse:
    """
    Create a public read-only share link for an itinerary.

    Args:
        itinerary_id: The itinerary identifier.
        share_service: The share service.
        current_user: The authenticated user when present.

    Returns:
        The created share-link response.
    """
    share_link = await share_service.create_share_link(
        itinerary_id,
        user_id=current_user.id if current_user else None,
    )
    if share_link is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Itinerary not found"
        )
    return share_link


@router.get(
    "/shared-itineraries/{token}",
    response_model=SharedItineraryResponse,
)
async def get_shared_itinerary(
    token: str,
    share_service: Annotated[ShareService, Depends(get_share_service)],
) -> SharedItineraryResponse:
    """
    Return a public read-only itinerary for a share token.

    Args:
        token: The share token.
        share_service: The share service.

    Returns:
        The shared itinerary and report.
    """
    shared_itinerary = await share_service.get_shared_itinerary(token)
    if shared_itinerary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Share link not found"
        )
    return shared_itinerary
