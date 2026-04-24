"""Itinerary domain models."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
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


class ItineraryItemType(StrEnum):
    """
    Supported itinerary item types.
    """

    ATTRACTION = "attraction"
    LUNCH = "lunch"
    DINNER = "dinner"


class Coordinates(BaseModel):
    """
    Geographic coordinates.
    """

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class PlaceRecommendation(BaseModel):
    """
    A normalized recommended place from Google Maps Platform.
    """

    place_id: str
    name: str
    category: str
    address: str | None = None
    location: Coordinates | None = None
    google_maps_uri: str | None = None
    rating: float | None = None
    user_rating_count: int | None = None
    price_level: str | None = None
    types: list[str] = Field(default_factory=list)
    regular_opening_hours: dict[str, Any] | None = None
    current_opening_hours: dict[str, Any] | None = None
    score: float = 0.0


class RouteLeg(BaseModel):
    """
    A route leg between two itinerary places.
    """

    origin_place_id: str
    destination_place_id: str
    travel_mode: str
    distance_meters: int | None = None
    duration_seconds: int | None = None
    encoded_polyline: str | None = None


class RouteSummary(BaseModel):
    """
    Aggregated route data for an itinerary day.
    """

    travel_mode: str
    distance_meters: int = 0
    duration_seconds: int = 0
    legs: list[RouteLeg] = Field(default_factory=list)


class ItineraryItem(BaseModel):
    """
    A scheduled itinerary item.
    """

    time: str
    type: ItineraryItemType
    place: PlaceRecommendation
    duration_minutes: int


class ItineraryDay(BaseModel):
    """
    A single day in a generated itinerary.
    """

    day_number: int = Field(ge=1)
    date: str | None = None
    theme: str = "highlights"
    summary: str
    items: list[ItineraryItem] = Field(default_factory=list)
    route: RouteSummary | None = None


class Itinerary(BaseModel):
    """
    A generated travel itinerary.
    """

    id: str = Field(default_factory=lambda: _new_id("itin"))
    conversation_id: str
    title: str
    destination_name: str
    destination_location: Coordinates | None = None
    hotels: list[PlaceRecommendation] = Field(default_factory=list)
    days: list[ItineraryDay] = Field(default_factory=list)
    guide_markdown: str
    created_at: datetime = Field(default_factory=_utc_now)
