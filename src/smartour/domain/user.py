"""User identity domain models."""

from datetime import UTC, datetime
from secrets import token_urlsafe
from uuid import uuid4

from pydantic import BaseModel, Field


def normalize_username(username: str) -> str:
    """
    Normalize a username for uniqueness checks.

    Args:
        username: The raw username.

    Returns:
        The normalized username.
    """
    return username.strip().lower()


def _new_id(prefix: str) -> str:
    """
    Generate a prefixed unique identifier.

    Args:
        prefix: The identifier prefix.

    Returns:
        The generated identifier.
    """
    return f"{prefix}_{uuid4().hex}"


def _new_session_token() -> str:
    """
    Generate an opaque session token.

    Returns:
        The generated session token.
    """
    return token_urlsafe(32)


def _utc_now() -> datetime:
    """
    Return the current UTC datetime.

    Returns:
        The current UTC datetime.
    """
    return datetime.now(tz=UTC)


class User(BaseModel):
    """
    A local Smartour user account.
    """

    id: str = Field(default_factory=lambda: _new_id("usr"))
    username: str
    normalized_username: str
    password_hash: str
    password_salt: str
    is_admin: bool = False
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class UserSession(BaseModel):
    """
    A persisted browser session for a local user.
    """

    token: str = Field(default_factory=_new_session_token)
    user_id: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=_utc_now)

    def is_expired(self, now: datetime | None = None) -> bool:
        """
        Return whether the session has expired.

        Args:
            now: The comparison time. Current UTC time is used when omitted.

        Returns:
            True when the session expiry is in the past.
        """
        comparison_time = now or _utc_now()
        return self.expires_at <= comparison_time
