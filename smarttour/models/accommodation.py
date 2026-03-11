"""
Accommodation models.
"""

from __future__ import annotations

import json

from sqlalchemy import Column, Float, Integer, Text
from sqlmodel import Field, SQLModel


class Hotel(SQLModel):
    """
    Public accommodation model.
    """

    id: str
    name: str
    destination: str
    latitude: float
    longitude: float
    price_per_night: float = Field(default=0.0, ge=0.0)
    star_rating: int = Field(default=3, ge=1, le=5)
    category: str = "hotel"
    amenities: list[str] = Field(default_factory=list)
    booking_url: str | None = None


class AccommodationOption(SQLModel):
    """
    Wrapper model for accommodation candidates.
    """

    options: list[Hotel] = Field(default_factory=list)


class AccommodationRecord(SQLModel, table=True):
    """
    SQLite table for accommodations.
    """

    __tablename__ = "accommodations"

    id: str = Field(primary_key=True)
    name: str
    destination: str = Field(index=True)
    latitude: float = Field(sa_column=Column(Float, nullable=False))
    longitude: float = Field(sa_column=Column(Float, nullable=False))
    price_per_night: float = Field(
        default=0.0, sa_column=Column(Float, nullable=False, default=0.0)
    )
    star_rating: int = Field(
        default=3, sa_column=Column(Integer, nullable=False, default=3)
    )
    category: str = Field(
        default="hotel", sa_column=Column(Text, nullable=False, default="hotel")
    )
    amenities_json: str = Field(
        default="[]",
        sa_column=Column("amenities", Text, nullable=False, default="[]"),
    )
    booking_url: str | None = Field(default=None, sa_column=Column(Text, nullable=True))

    def to_model(self) -> Hotel:
        """
        Convert the record into a public hotel model.

        Returns:
            The hotel representation.
        """

        return Hotel(
            id=self.id,
            name=self.name,
            destination=self.destination,
            latitude=self.latitude,
            longitude=self.longitude,
            price_per_night=self.price_per_night,
            star_rating=self.star_rating,
            category=self.category,
            amenities=json.loads(self.amenities_json),
            booking_url=self.booking_url,
        )

    @classmethod
    def from_model(cls, hotel: Hotel) -> AccommodationRecord:
        """
        Create a database record from a hotel model.

        Args:
            hotel: The hotel to serialize.

        Returns:
            A database-ready accommodation record.
        """

        return cls(
            id=hotel.id,
            name=hotel.name,
            destination=hotel.destination,
            latitude=hotel.latitude,
            longitude=hotel.longitude,
            price_per_night=hotel.price_per_night,
            star_rating=hotel.star_rating,
            category=hotel.category,
            amenities_json=json.dumps(hotel.amenities),
            booking_url=hotel.booking_url,
        )
