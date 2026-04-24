"""Itinerary generation job domain models."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    """
    Generate a prefixed unique identifier.

    Args:
        prefix: The identifier prefix.

    Returns:
        The generated identifier.
    """
    return f"{prefix}_{uuid4().hex}"


def _utc_now() -> datetime:
    """
    Return the current UTC datetime.

    Returns:
        The current UTC datetime.
    """
    return datetime.now(tz=UTC)


class ItineraryJobStatus(StrEnum):
    """
    Supported itinerary generation job states.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ItineraryJob(BaseModel):
    """
    A background itinerary generation job.
    """

    id: str = Field(default_factory=lambda: _new_id("job"))
    conversation_id: str
    status: ItineraryJobStatus = ItineraryJobStatus.QUEUED
    itinerary_id: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def mark_running(self) -> None:
        """
        Mark the job as running.
        """
        now = _utc_now()
        self.status = ItineraryJobStatus.RUNNING
        self.started_at = now
        self.updated_at = now
        self.error_message = None

    def mark_succeeded(self, itinerary_id: str) -> None:
        """
        Mark the job as successfully completed.

        Args:
            itinerary_id: The generated itinerary ID.
        """
        now = _utc_now()
        self.status = ItineraryJobStatus.SUCCEEDED
        self.itinerary_id = itinerary_id
        self.completed_at = now
        self.updated_at = now
        self.error_message = None

    def mark_failed(self, error_message: str) -> None:
        """
        Mark the job as failed.

        Args:
            error_message: The sanitized failure reason.
        """
        now = _utc_now()
        self.status = ItineraryJobStatus.FAILED
        self.error_message = error_message
        self.completed_at = now
        self.updated_at = now
