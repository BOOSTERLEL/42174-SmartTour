"""
Attraction and guidance models.
"""

from __future__ import annotations

import json

from sqlalchemy import Column, Float, Integer, Text
from sqlmodel import Field, SQLModel

from smarttour.models.preferences import TravelPreferences


class Attraction(SQLModel):
    """
    Public attraction model.
    """

    id: str
    name: str
    destination: str
    category: str
    latitude: float
    longitude: float
    description: str = ""
    opening_hours: dict[str, str] = Field(default_factory=dict)
    visit_duration: int = Field(default=90, ge=15)
    cost: float = Field(default=0.0, ge=0.0)
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    accessibility: str | None = None
    tags: list[str] = Field(default_factory=list)
    image_url: str | None = None
    source: str = "seed"


class AttractionRecord(SQLModel, table=True):
    """
    SQLite table for attractions.
    """

    __tablename__ = "attractions"

    id: str = Field(primary_key=True)
    name: str
    destination: str = Field(index=True)
    category: str = Field(index=True)
    latitude: float = Field(sa_column=Column(Float, nullable=False))
    longitude: float = Field(sa_column=Column(Float, nullable=False))
    description: str = Field(
        default="", sa_column=Column(Text, nullable=False, default="")
    )
    opening_hours_json: str = Field(
        default="{}",
        sa_column=Column("opening_hours", Text, nullable=False, default="{}"),
    )
    visit_duration: int = Field(
        default=90, sa_column=Column(Integer, nullable=False, default=90)
    )
    cost: float = Field(
        default=0.0, sa_column=Column(Float, nullable=False, default=0.0)
    )
    rating: float = Field(
        default=0.0, sa_column=Column(Float, nullable=False, default=0.0)
    )
    accessibility: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    tags_json: str = Field(
        default="[]",
        sa_column=Column("tags", Text, nullable=False, default="[]"),
    )
    image_url: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    source: str = Field(
        default="seed", sa_column=Column(Text, nullable=False, default="seed")
    )

    def to_model(self) -> Attraction:
        """
        Convert the record into a public attraction model.

        Returns:
            The public attraction representation.
        """

        return Attraction(
            id=self.id,
            name=self.name,
            destination=self.destination,
            category=self.category,
            latitude=self.latitude,
            longitude=self.longitude,
            description=self.description,
            opening_hours=json.loads(self.opening_hours_json),
            visit_duration=self.visit_duration,
            cost=self.cost,
            rating=self.rating,
            accessibility=self.accessibility,
            tags=json.loads(self.tags_json),
            image_url=self.image_url,
            source=self.source,
        )

    @classmethod
    def from_model(cls, attraction: Attraction) -> AttractionRecord:
        """
        Create a database record from a public attraction model.

        Args:
            attraction: The attraction model to serialize.

        Returns:
            A database-ready attraction record.
        """

        return cls(
            id=attraction.id,
            name=attraction.name,
            destination=attraction.destination,
            category=attraction.category,
            latitude=attraction.latitude,
            longitude=attraction.longitude,
            description=attraction.description,
            opening_hours_json=json.dumps(attraction.opening_hours),
            visit_duration=attraction.visit_duration,
            cost=attraction.cost,
            rating=attraction.rating,
            accessibility=attraction.accessibility,
            tags_json=json.dumps(attraction.tags),
            image_url=attraction.image_url,
            source=attraction.source,
        )


class AttractionSearchResponse(SQLModel):
    """
    Attraction search response payload.
    """

    results: list[Attraction]
    total: int


class GuidanceRequest(SQLModel):
    """
    Payload for AI-style attraction guidance generation.
    """

    attraction_id: str
    preferences: TravelPreferences | None = None


class GuidanceResponse(SQLModel):
    """
    Generated attraction guidance.
    """

    attraction: Attraction
    historical_background: str
    visiting_tips: list[str] = Field(default_factory=list)
    practical_notes: list[str] = Field(default_factory=list)
