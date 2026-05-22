"""User dashboard service for owned plans and share links."""

from typing import Any

from pydantic import BaseModel


class UserItinerarySummary(BaseModel):
    """
    Summary of an itinerary created by the current user.
    """

    itinerary_id: str
    title: str
    destination_name: str
    created_at: str
    open_path: str


class UserShareLinkSummary(BaseModel):
    """
    Summary of a share link created by the current user.
    """

    token: str
    itinerary_id: str
    itinerary_title: str
    share_path: str
    created_at: str


class UserDashboardResponse(BaseModel):
    """
    Current user's saved plan and share-link dashboard.
    """

    created_itineraries: list[UserItinerarySummary]
    share_links: list[UserShareLinkSummary]


class UserDashboardService:
    """
    Builds account dashboard data for a local user.
    """

    def __init__(self, itinerary_repository: Any, share_repository: Any) -> None:
        """
        Initialize the user dashboard service.

        Args:
            itinerary_repository: Repository used to list owned itineraries.
            share_repository: Repository used to list owned share links.
        """
        self.itinerary_repository = itinerary_repository
        self.share_repository = share_repository

    async def get_dashboard(self, user_id: str) -> UserDashboardResponse:
        """
        Build dashboard data for a user.

        Args:
            user_id: The current user ID.

        Returns:
            The user's dashboard data.
        """
        itineraries = await self.itinerary_repository.list_by_user(user_id)
        share_links = await self.share_repository.list_by_user(user_id)
        return UserDashboardResponse(
            created_itineraries=[
                UserItinerarySummary(
                    itinerary_id=itinerary.id,
                    title=itinerary.title,
                    destination_name=itinerary.destination_name,
                    created_at=itinerary.created_at.isoformat(),
                    open_path=f"/itineraries/{itinerary.id}",
                )
                for itinerary in itineraries
            ],
            share_links=[
                await self._share_link_summary(share_link) for share_link in share_links
            ],
        )

    async def _share_link_summary(self, share_link: Any) -> UserShareLinkSummary:
        """
        Build a dashboard summary for a share link.

        Args:
            share_link: The share-link domain model.

        Returns:
            The share-link summary.
        """
        itinerary = await self.itinerary_repository.get(share_link.itinerary_id)
        itinerary_title = itinerary.title if itinerary is not None else "Deleted plan"
        return UserShareLinkSummary(
            token=share_link.token,
            itinerary_id=share_link.itinerary_id,
            itinerary_title=itinerary_title,
            share_path=share_link.share_path,
            created_at=share_link.created_at.isoformat(),
        )
