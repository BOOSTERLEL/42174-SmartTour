"""
Shared data models.
"""

from smarttour.models.accommodation import (
    AccommodationOption,
    AccommodationRecord,
    Hotel,
)
from smarttour.models.attractions import (
    Attraction,
    AttractionRecord,
    AttractionSearchResponse,
    GuidanceRequest,
    GuidanceResponse,
)
from smarttour.models.distance_cache import DistanceCacheRecord
from smarttour.models.itinerary import (
    DayPlan,
    Itinerary,
    ItineraryGenerateRequest,
    ItineraryRegenerateRequest,
    TimeSlot,
)
from smarttour.models.preferences import (
    PreferenceParseRequest,
    PreferenceParseResponse,
    PreferenceRefineRequest,
    TravelPreferences,
    UserInput,
)
from smarttour.models.session import PlanningSession, PlanningSessionRecord
from smarttour.models.transport import (
    Route,
    RouteOptimizationRequest,
    RouteOptimizationResponse,
    TransportSegment,
)

__all__ = [
    "AccommodationOption",
    "AccommodationRecord",
    "Attraction",
    "AttractionRecord",
    "AttractionSearchResponse",
    "DayPlan",
    "DistanceCacheRecord",
    "GuidanceRequest",
    "GuidanceResponse",
    "Hotel",
    "Itinerary",
    "ItineraryGenerateRequest",
    "ItineraryRegenerateRequest",
    "PlanningSession",
    "PlanningSessionRecord",
    "PreferenceParseRequest",
    "PreferenceParseResponse",
    "PreferenceRefineRequest",
    "Route",
    "RouteOptimizationRequest",
    "RouteOptimizationResponse",
    "TimeSlot",
    "TransportSegment",
    "TravelPreferences",
    "UserInput",
]
