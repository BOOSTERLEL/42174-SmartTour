"""
Distance cache models.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Integer, Text
from sqlmodel import Field, SQLModel

from smarttour.models.transport import TransportSegment


class DistanceCacheRecord(SQLModel, table=True):
    """
    SQLite table for cached travel distances.
    """

    __tablename__ = "distance_cache"

    origin_id: str = Field(primary_key=True)
    dest_id: str = Field(primary_key=True)
    travel_mode: str = Field(primary_key=True)
    distance_m: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, default=0)
    )
    duration_s: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, default=0)
    )
    polyline: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    def to_model(self) -> TransportSegment:
        """
        Convert the record into a transport segment.

        Returns:
            The transport segment representation.
        """

        return TransportSegment(
            origin_id=self.origin_id,
            destination_id=self.dest_id,
            travel_mode=self.travel_mode,
            distance_m=self.distance_m,
            duration_s=self.duration_s,
            polyline=self.polyline,
        )

    @classmethod
    def from_model(cls, segment: TransportSegment) -> DistanceCacheRecord:
        """
        Create a cache record from a transport segment.

        Args:
            segment: The transport segment to serialize.

        Returns:
            A database-ready cache record.
        """

        return cls(
            origin_id=segment.origin_id,
            dest_id=segment.destination_id,
            travel_mode=segment.travel_mode,
            distance_m=segment.distance_m,
            duration_s=segment.duration_s,
            polyline=segment.polyline,
        )
