"""
Itinerary models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlmodel import Field, SQLModel

from smarttour.models.accommodation import Hotel
from smarttour.models.attractions import Attraction
from smarttour.models.preferences import TravelPreferences
from smarttour.models.restaurant import Restaurant
from smarttour.models.transport import Route, TransportSegment

SlotType = Literal[
    "attraction",
    "breakfast",
    "lunch",
    "dinner",
    "hotel_checkin",
    "hotel_checkout",
]


class TimeSlot(SQLModel):
    """
    A scheduled stop in a day plan.
    """

    slot_type: SlotType = "attraction"
    start_time: str
    end_time: str
    title: str
    description: str
    attraction: Attraction | None = None
    restaurant: Restaurant | None = None
    hotel: Hotel | None = None
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


class DayTheme(SQLModel):
    """
    AI-generated theme and attraction grouping for one day.
    """

    day_number: int = Field(default=1, ge=1)
    theme: str
    attraction_ids: list[str] = Field(default_factory=list)


class ItineraryPlan(SQLModel):
    """
    Structured AI planning guidance used before route optimization.
    """

    reasoning: str = ""
    day_themes: list[DayTheme] = Field(default_factory=list)
    priority_attractions: list[str] = Field(default_factory=list)
    pacing_notes: str = ""


class ItineraryReview(SQLModel):
    """
    Structured AI review attached after itinerary generation.
    """

    overall_score: int = Field(default=0, ge=0, le=10)
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    revised_warnings: list[str] = Field(default_factory=list)


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
    ai_plan_reasoning: str = ""
    ai_review: ItineraryReview | None = None


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
