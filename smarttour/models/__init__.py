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
    DayTheme,
    Itinerary,
    ItineraryGenerateRequest,
    ItineraryPlan,
    ItineraryRegenerateRequest,
    ItineraryReview,
    TimeSlot,
)
from smarttour.models.preferences import (
    PreferenceParseRequest,
    PreferenceParseResponse,
    PreferenceRefineRequest,
    TravelPreferences,
    UserInput,
)
from smarttour.models.restaurant import Restaurant, RestaurantRecord
from smarttour.models.session import PlanningSession, PlanningSessionRecord
from smarttour.models.transport import (
    Route,
    RouteOptimizationRequest,
    RouteOptimizationResponse,
    TransportSegment,
)
from smarttour.models.xhs_popularity import XhsDiscovery, XhsPopularityRecord

__all__ = [
    "AccommodationOption",
    "AccommodationRecord",
    "Attraction",
    "AttractionRecord",
    "AttractionSearchResponse",
    "DayPlan",
    "DayTheme",
    "DistanceCacheRecord",
    "GuidanceRequest",
    "GuidanceResponse",
    "Hotel",
    "Itinerary",
    "ItineraryGenerateRequest",
    "ItineraryPlan",
    "ItineraryRegenerateRequest",
    "ItineraryReview",
    "PlanningSession",
    "PlanningSessionRecord",
    "PreferenceParseRequest",
    "PreferenceParseResponse",
    "PreferenceRefineRequest",
    "Restaurant",
    "RestaurantRecord",
    "Route",
    "RouteOptimizationRequest",
    "RouteOptimizationResponse",
    "TimeSlot",
    "TransportSegment",
    "TravelPreferences",
    "UserInput",
    "XhsDiscovery",
    "XhsPopularityRecord",
]
