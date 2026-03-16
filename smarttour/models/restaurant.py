"""
Restaurant models for itinerary meal planning.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, Text
from sqlmodel import Field, SQLModel


class Restaurant(SQLModel):
    """
    Public restaurant model.
    """

    id: str
    name: str
    destination: str
    cuisine: str
    meal_types: list[str] = Field(default_factory=list)
    latitude: float
    longitude: float
    description: str = ""
    opening_hours: dict[str, str] = Field(default_factory=dict)
    average_cost: float = Field(default=0.0, ge=0.0)
    visit_duration: int = Field(default=60, ge=15)
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    tags: list[str] = Field(default_factory=list)
    image_url: str | None = None
    source: str = "seed"


class RestaurantRecord(SQLModel, table=True):
    """
    SQLite table for restaurants.
    """

    __tablename__ = "restaurants"

    id: str = Field(primary_key=True)
    name: str
    destination: str = Field(index=True)
    cuisine: str = Field(index=True)
    meal_types_json: str = Field(
        default="[]",
        sa_column=Column("meal_types", Text, nullable=False, default="[]"),
    )
    latitude: float = Field(sa_column=Column(Float, nullable=False))
    longitude: float = Field(sa_column=Column(Float, nullable=False))
    description: str = Field(
        default="", sa_column=Column(Text, nullable=False, default="")
    )
    opening_hours_json: str = Field(
        default="{}",
        sa_column=Column("opening_hours", Text, nullable=False, default="{}"),
    )
    average_cost: float = Field(
        default=0.0, sa_column=Column(Float, nullable=False, default=0.0)
    )
    visit_duration: int = Field(
        default=60, sa_column=Column(Integer, nullable=False, default=60)
    )
    rating: float = Field(
        default=0.0, sa_column=Column(Float, nullable=False, default=0.0)
    )
    tags_json: str = Field(
        default="[]",
        sa_column=Column("tags", Text, nullable=False, default="[]"),
    )
    image_url: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    fetched_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    source: str = Field(
        default="seed", sa_column=Column(Text, nullable=False, default="seed")
    )

    def to_model(self) -> Restaurant:
        """
        Convert the record into a public restaurant model.

        Returns:
            The public restaurant representation.
        """

        return Restaurant(
            id=self.id,
            name=self.name,
            destination=self.destination,
            cuisine=self.cuisine,
            meal_types=json.loads(self.meal_types_json),
            latitude=self.latitude,
            longitude=self.longitude,
            description=self.description,
            opening_hours=json.loads(self.opening_hours_json),
            average_cost=self.average_cost,
            visit_duration=self.visit_duration,
            rating=self.rating,
            tags=json.loads(self.tags_json),
            image_url=self.image_url,
            source=self.source,
        )

    @classmethod
    def from_model(
        cls,
        restaurant: Restaurant,
        fetched_at: datetime | None = None,
    ) -> RestaurantRecord:
        """
        Create a database record from a public restaurant model.

        Args:
            restaurant: The restaurant model to serialize.
            fetched_at: Optional cache timestamp.

        Returns:
            A database-ready restaurant record.
        """

        return cls(
            id=restaurant.id,
            name=restaurant.name,
            destination=restaurant.destination,
            cuisine=restaurant.cuisine,
            meal_types_json=json.dumps(restaurant.meal_types),
            latitude=restaurant.latitude,
            longitude=restaurant.longitude,
            description=restaurant.description,
            opening_hours_json=json.dumps(restaurant.opening_hours),
            average_cost=restaurant.average_cost,
            visit_duration=restaurant.visit_duration,
            rating=restaurant.rating,
            tags_json=json.dumps(restaurant.tags),
            image_url=restaurant.image_url,
            fetched_at=fetched_at,
            source=restaurant.source,
        )
