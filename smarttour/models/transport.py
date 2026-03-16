"""
Transport and route optimization models.
"""

from typing import Literal

from sqlmodel import Field, SQLModel

from smarttour.models.attractions import Attraction


class TransportSegment(SQLModel):
    """
    Travel segment between two attractions.
    """

    origin_id: str
    destination_id: str
    travel_mode: Literal["walking", "transit", "driving"] = "walking"
    distance_m: int = Field(default=0, ge=0)
    duration_s: int = Field(default=0, ge=0)
    navigation_hint: str = ""
    polyline: str | None = None


class Route(SQLModel):
    """
    Optimized route for one day of attractions.
    """

    attraction_order: list[str] = Field(default_factory=list)
    segments: list[TransportSegment] = Field(default_factory=list)
    total_distance_m: int = Field(default=0, ge=0)
    total_duration_s: int = Field(default=0, ge=0)


class RouteOptimizationRequest(SQLModel):
    """
    Payload for route optimization.
    """

    attractions: list[Attraction]
    travel_mode: Literal["walking", "transit", "driving", "walking_transit"] = (
        "walking_transit"
    )


class RouteOptimizationResponse(SQLModel):
    """
    Route optimization result.
    """

    route: Route
