"""Share-link domain models."""

from datetime import UTC, datetime
from secrets import token_urlsafe

from pydantic import BaseModel, Field


def _new_share_token() -> str:
    """
    Generate an opaque URL-safe share token.

    Returns:
        The generated share token.
    """
    return token_urlsafe(24)


def _utc_now() -> datetime:
    """
    Return the current UTC datetime.

    Returns:
        The current UTC datetime.
    """
    return datetime.now(tz=UTC)


class ItineraryShareLink(BaseModel):
    """
    Public read-only share link for a generated itinerary.
    """

    token: str = Field(default_factory=_new_share_token)
    itinerary_id: str
    user_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    @property
    def share_path(self) -> str:
        """
        Return the frontend path for this share token.

        Returns:
            The frontend share path.
        """
        return f"/share/{self.token}"
