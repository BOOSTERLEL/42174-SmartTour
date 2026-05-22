"""Application service for itinerary share links."""

from typing import Any

from pydantic import BaseModel

from smartour.application.report_service import ItineraryReport, ReportService
from smartour.domain.itinerary import Itinerary
from smartour.domain.share import ItineraryShareLink


class ShareLinkResponse(BaseModel):
    """
    Response returned after creating an itinerary share link.
    """

    token: str
    itinerary_id: str
    share_path: str
    created_at: str


class SharedItineraryResponse(BaseModel):
    """
    Public read-only shared itinerary response.
    """

    token: str
    itinerary: Itinerary
    report: ItineraryReport


class ShareService:
    """
    Creates and resolves public read-only itinerary share links.
    """

    def __init__(
        self,
        itinerary_repository: Any,
        share_repository: Any,
        report_service: ReportService,
    ) -> None:
        """
        Initialize the share service.

        Args:
            itinerary_repository: Repository used to load itineraries.
            share_repository: Repository used to persist share links.
            report_service: Service used to generate shared reports.
        """
        self.itinerary_repository = itinerary_repository
        self.share_repository = share_repository
        self.report_service = report_service

    async def create_share_link(self, itinerary_id: str) -> ShareLinkResponse | None:
        """
        Create a public share link for a persisted itinerary.

        Args:
            itinerary_id: The itinerary identifier.

        Returns:
            The created share link response, or None when the itinerary is missing.
        """
        itinerary = await self.itinerary_repository.get(itinerary_id)
        if itinerary is None:
            return None
        share_link = ItineraryShareLink(itinerary_id=itinerary.id)
        await self.share_repository.save(share_link)
        return ShareLinkResponse(
            token=share_link.token,
            itinerary_id=share_link.itinerary_id,
            share_path=share_link.share_path,
            created_at=share_link.created_at.isoformat(),
        )

    async def get_shared_itinerary(
        self, token: str
    ) -> SharedItineraryResponse | None:
        """
        Return the itinerary and report for a public share token.

        Args:
            token: The share token.

        Returns:
            The shared itinerary response, or None when the token is unknown.
        """
        share_link = await self.share_repository.get_by_token(token)
        if share_link is None:
            return None
        itinerary = await self.itinerary_repository.get(share_link.itinerary_id)
        if itinerary is None:
            return None
        report = await self.report_service.generate_itinerary_report(itinerary.id)
        if report is None:
            return None
        return SharedItineraryResponse(
            token=share_link.token,
            itinerary=itinerary,
            report=report,
        )
