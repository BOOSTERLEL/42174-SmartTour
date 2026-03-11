"""
Itinerary models.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel

from smarttour.models.accommodation import Hotel
from smarttour.models.attractions import Attraction
from smarttour.models.preferences import TravelPreferences
from smarttour.models.transport import Route, TransportSegment


class TimeSlot(SQLModel):
    """
    A scheduled stop in a day plan.
    """

    start_time: str
    end_time: str
    title: str
    description: str
    attraction: Attraction | None = None
    transport_from_previous: TransportSegment | None = None
    cost: float = Field(default=0.0, ge=0.0)


class DayPlan(SQLModel):
    """
    One day of an itinerary.
    """

    day_number: int = Field(ge=1)
    label: str
    slots: list[TimeSlot] = Field(default_factory=list)
    route: Route
    estimated_cost: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class Itinerary(SQLModel):
    """
    Multi-day itinerary returned to clients.
    """

    id: str
    destination: str
    created_at: datetime
    preferences: TravelPreferences
    accommodation: Hotel | None = None
    days: list[DayPlan] = Field(default_factory=list)
    total_estimated_cost: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class ItineraryGenerateRequest(SQLModel):
    """
    Payload for itinerary generation.
    """

    preferences: TravelPreferences
    user_input: str = ""


class ItineraryRegenerateRequest(SQLModel):
    """
    Payload for itinerary regeneration.
    """

    session_id: str
    preferences: TravelPreferences
    user_input: str = ""
