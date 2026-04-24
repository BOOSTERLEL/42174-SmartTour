"""Itinerary planning orchestration service."""

import re
from datetime import UTC, date, datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any

from smartour.core.errors import PlanningInputError
from smartour.domain.itinerary import (
    Coordinates,
    Itinerary,
    ItineraryDay,
    ItineraryItem,
    ItineraryItemType,
    PlaceRecommendation,
    RouteLeg,
    RouteSummary,
)
from smartour.domain.requirement import TravelRequirement
from smartour.integrations.google_maps.client import GoogleMapsClient
from smartour.integrations.google_maps.field_masks import (
    PLACES_RECOMMENDATION_FIELD_MASK,
)

PRICE_LEVELS_BY_BUDGET = {
    "low": ["PRICE_LEVEL_INEXPENSIVE"],
    "medium": ["PRICE_LEVEL_MODERATE"],
    "high": ["PRICE_LEVEL_EXPENSIVE", "PRICE_LEVEL_VERY_EXPENSIVE"],
}
TRAVEL_MODES_BY_REQUIREMENT = {
    "walking": "WALK",
    "transit": "TRANSIT",
    "drive": "DRIVE",
}
ICONIC_DISCOVERY_QUERIES = [
    ("famous landmarks in {destination}", None),
    ("scenic viewpoints and waterfront walks in {destination}", "tourist_attraction"),
]
ICONIC_DISCOVERY_SCORE_BOOST = 8.0
INTEREST_INCLUDED_TYPES = {
    "architecture": "historical_landmark",
    "art": "art_gallery",
    "beach": "beach",
    "beaches": "beach",
    "culture": "museum",
    "family": "aquarium",
    "food": "market",
    "foodie": "market",
    "garden": "garden",
    "gardens": "garden",
    "heritage": "historical_landmark",
    "hiking": "hiking_area",
    "history": "historical_landmark",
    "kids": "aquarium",
    "market": "market",
    "markets": "market",
    "museum": "museum",
    "museums": "museum",
    "nature": "park",
    "nightlife": "night_club",
    "outdoors": "hiking_area",
    "park": "park",
    "parks": "park",
    "shopping": "shopping_mall",
    "wildlife": "wildlife_park",
}
CANDIDATE_DISTANCE_LIMITS_METERS = {
    "walking": 5000.0,
    "transit": 12000.0,
    "drive": 30000.0,
}
CLUSTER_RADIUS_METERS = {
    "walking": 1800.0,
    "transit": 4000.0,
    "drive": 9000.0,
}
DAILY_ROUTE_DURATION_LIMITS_SECONDS = {
    "walking": 18000,
    "transit": 7200,
    "drive": 10800,
}
DAILY_ROUTE_DISTANCE_LIMITS_METERS = {
    "walking": 10000,
    "transit": 18000,
    "drive": 60000,
}
ATTRACTION_TIME_SLOTS = ["10:00", "14:00", "16:15"]
LUNCH_TIME = "12:15"
DINNER_TIME = "18:30"
OPENING_HOURS_DAY_OFFSET = 1
THEME_TYPE_WEIGHT = 3.0
THEME_TEXT_WEIGHT = 1.25
THEME_INTEREST_WEIGHT = 1.25
COMBINED_THEME_MIN_SCORE = 3.0
COMBINED_THEME_MAX_SCORE_GAP = 3.5
CLASSIC_SIGHTSEEING_THEME = "classic sightseeing"
WATERFRONT_DINING_THEME = "waterfront and dining"
WATERFRONT_DINING_MIN_FOOD_SCORE = 4.0
WATERFRONT_DINING_MIN_WATERFRONT_SCORE = 2.5
THEME_TYPE_KEYWORDS = {
    CLASSIC_SIGHTSEEING_THEME: [
        "bridge",
        "cultural_landmark",
        "historical_landmark",
        "monument",
        "observation_deck",
        "opera_house",
        "plaza",
        "scenic_spot",
    ],
    "arts and museums": [
        "art_gallery",
        "art_museum",
        "art_studio",
        "auditorium",
        "cultural_center",
        "museum",
        "opera_house",
        "performing_arts_theater",
    ],
    "heritage and landmarks": [
        "buddhist_temple",
        "castle",
        "church",
        "cultural_landmark",
        "historical_landmark",
        "historical_place",
        "history_museum",
        "hindu_temple",
        "monument",
        "mosque",
        "shinto_shrine",
        "synagogue",
    ],
    "parks and gardens": [
        "botanical_garden",
        "city_park",
        "garden",
        "national_park",
        "park",
        "state_park",
        "wildlife_park",
        "wildlife_refuge",
        "zoo",
    ],
    "waterfront and views": [
        "beach",
        "bridge",
        "ferry_service",
        "ferry_terminal",
        "island",
        "lake",
        "marina",
        "observation_deck",
        "river",
        "scenic_spot",
    ],
    "food and markets": [
        "bar",
        "brunch_restaurant",
        "cafe",
        "coffee_shop",
        "farmers_market",
        "food_store",
        "market",
        "pub",
        "restaurant",
        "seafood_restaurant",
    ],
    "shopping and neighborhoods": [
        "department_store",
        "flea_market",
        "market",
        "plaza",
        "shopping_mall",
        "store",
    ],
    "family attractions": [
        "amusement_park",
        "aquarium",
        "ferris_wheel",
        "indoor_playground",
        "playground",
        "water_park",
        "wildlife_park",
        "zoo",
    ],
    "nightlife and entertainment": [
        "bar",
        "casino",
        "comedy_club",
        "concert_hall",
        "live_music_venue",
        "night_club",
        "pub",
    ],
    "outdoors and adventure": [
        "adventure_sports_center",
        "beach",
        "cycling_park",
        "hiking_area",
        "mountain_peak",
        "nature_preserve",
        "ski_resort",
        "woods",
    ],
}
THEME_TEXT_KEYWORDS = {
    CLASSIC_SIGHTSEEING_THEME: [
        "bridge",
        "cathedral",
        "icon",
        "iconic",
        "landmark",
        "lookout",
        "monument",
        "old town",
        "opera house",
        "palace",
        "plaza",
        "quay",
        "square",
        "tower",
    ],
    "arts and museums": [
        "art",
        "arts",
        "culture",
        "cultural",
        "gallery",
        "museum",
        "museums",
        "opera",
        "theater",
        "theatre",
    ],
    "heritage and landmarks": [
        "barracks",
        "castle",
        "cathedral",
        "church",
        "heritage",
        "historic",
        "historical",
        "history",
        "landmark",
        "monument",
        "old town",
        "shrine",
        "temple",
    ],
    "parks and gardens": [
        "botanic",
        "botanical",
        "garden",
        "gardens",
        "green",
        "park",
        "parks",
        "reserve",
        "wildlife",
        "zoo",
    ],
    "waterfront and views": [
        "bay",
        "beach",
        "coast",
        "coastal",
        "cove",
        "harbor",
        "harborside",
        "harbour",
        "harbourside",
        "head",
        "island",
        "lake",
        "lookout",
        "pier",
        "promenade",
        "quay",
        "river",
        "scenic",
        "view",
        "views",
        "waterfront",
        "wharf",
    ],
    "food and markets": [
        "bakery",
        "bar",
        "brunch",
        "cafe",
        "coffee",
        "dining",
        "food",
        "market",
        "pub",
        "restaurant",
        "restaurants",
        "seafood",
    ],
    "shopping and neighborhoods": [
        "boutique",
        "district",
        "mall",
        "market",
        "neighborhood",
        "neighbourhood",
        "plaza",
        "shop",
        "shopping",
        "shops",
        "souvenir",
        "street",
    ],
    "family attractions": [
        "amusement",
        "aquarium",
        "children",
        "family",
        "kids",
        "playground",
        "theme park",
        "wildlife",
        "zoo",
    ],
    "nightlife and entertainment": [
        "bar",
        "casino",
        "club",
        "comedy",
        "concert",
        "music",
        "night",
        "nightlife",
        "pub",
        "theater",
        "theatre",
    ],
    "outdoors and adventure": [
        "adventure",
        "cycling",
        "forest",
        "hike",
        "hiking",
        "mountain",
        "nature",
        "outdoor",
        "outdoors",
        "reserve",
        "trail",
        "walk",
        "woods",
    ],
}
THEME_INTEREST_ALIASES = {
    CLASSIC_SIGHTSEEING_THEME: [
        "classic",
        "first time",
        "first-time",
        "highlights",
        "iconic",
        "landmark",
        "sightseeing",
    ],
    "arts and museums": [
        "art",
        "arts",
        "culture",
        "museum",
        "museums",
        "theater",
        "theatre",
    ],
    "heritage and landmarks": [
        "architecture",
        "heritage",
        "historic",
        "history",
        "landmark",
    ],
    "parks and gardens": [
        "garden",
        "nature",
        "park",
        "parks",
        "wildlife",
    ],
    "waterfront and views": [
        "beach",
        "coast",
        "harbor",
        "harbour",
        "scenic",
        "views",
        "waterfront",
    ],
    "food and markets": [
        "cafe",
        "coffee",
        "dining",
        "food",
        "foodie",
        "market",
        "restaurant",
        "restaurants",
    ],
    "shopping and neighborhoods": [
        "fashion",
        "market",
        "markets",
        "shopping",
    ],
    "family attractions": [
        "children",
        "family",
        "kids",
    ],
    "nightlife and entertainment": [
        "bar",
        "bars",
        "music",
        "nightlife",
    ],
    "outdoors and adventure": [
        "adventure",
        "hiking",
        "nature",
        "outdoor",
        "outdoors",
    ],
}


class PlanningService:
    """
    Generates a first-pass itinerary from confirmed travel requirements.
    """

    def __init__(
        self,
        conversation_repository: Any,
        itinerary_repository: Any,
    ) -> None:
        """
        Initialize the planning service.

        Args:
            conversation_repository: The repository used to read conversations.
            itinerary_repository: The repository used to persist itineraries.
        """
        self.conversation_repository = conversation_repository
        self.itinerary_repository = itinerary_repository

    async def generate_for_conversation(
        self, conversation_id: str, google_maps_client: GoogleMapsClient
    ) -> Itinerary | None:
        """
        Generate and persist an itinerary for a conversation.

        Args:
            conversation_id: The source conversation ID.
            google_maps_client: The Google Maps client group.

        Returns:
            The generated itinerary, or None when the conversation is missing.

        Raises:
            PlanningInputError: Raised when required slots are incomplete.
        """
        conversation = await self.conversation_repository.get(conversation_id)
        if conversation is None:
            return None
        requirement = conversation.requirement
        missing_slots = requirement.missing_required_slots()
        if missing_slots:
            raise PlanningInputError(
                "Cannot generate an itinerary until requirements are complete"
            )

        destination_location = await self._resolve_destination(
            requirement, google_maps_client
        )
        hotels = await self._discover_hotels(
            requirement, destination_location, google_maps_client
        )
        attractions = await self._discover_attractions(
            requirement, destination_location, google_maps_client
        )
        restaurants = await self._discover_restaurants(
            requirement, destination_location, google_maps_client
        )
        if not hotels:
            raise PlanningInputError("No hotel candidates were found")
        if not attractions:
            raise PlanningInputError("No attraction candidates were found")
        if not restaurants:
            raise PlanningInputError("No restaurant candidates were found")

        days = await self._build_days(
            requirement,
            hotels[0],
            attractions,
            restaurants,
            google_maps_client,
        )
        itinerary = Itinerary(
            conversation_id=conversation_id,
            title=self._build_title(requirement),
            destination_name=requirement.destination or "Destination",
            destination_location=destination_location,
            hotels=hotels[:3],
            days=days,
            guide_markdown=self._render_guide(requirement, hotels[:3], days),
        )
        await self.itinerary_repository.save(itinerary)
        return itinerary

    async def get_itinerary(self, itinerary_id: str) -> Itinerary | None:
        """
        Return a generated itinerary by ID.

        Args:
            itinerary_id: The itinerary ID.

        Returns:
            The itinerary when found.
        """
        return await self.itinerary_repository.get(itinerary_id)

    async def _resolve_destination(
        self, requirement: TravelRequirement, google_maps_client: GoogleMapsClient
    ) -> Coordinates | None:
        """
        Resolve destination text to coordinates using Geocoding API.

        Args:
            requirement: The confirmed travel requirement snapshot.
            google_maps_client: The Google Maps client group.

        Returns:
            Destination coordinates when available.
        """
        if not requirement.destination:
            return None
        payload = await google_maps_client.geocoding.geocode(
            requirement.destination, language=requirement.language
        )
        results = payload.get("results", [])
        if not results:
            return None
        geometry = results[0].get("geometry", {})
        location = geometry.get("location", {})
        return _coordinates_from_google_location(location)

    async def _discover_hotels(
        self,
        requirement: TravelRequirement,
        destination_location: Coordinates | None,
        google_maps_client: GoogleMapsClient,
    ) -> list[PlaceRecommendation]:
        """
        Discover hotel candidates with Places Text Search.

        Args:
            requirement: The confirmed travel requirement snapshot.
            destination_location: Destination coordinates for location bias.
            google_maps_client: The Google Maps client group.

        Returns:
            Ranked hotel recommendations.
        """
        budget_text = _budget_search_text(requirement.budget_level)
        query = (
            f"{budget_text} hotels near {requirement.hotel_area} "
            f"in {requirement.destination}"
        )
        return await self._search_places(
            google_maps_client=google_maps_client,
            text_query=query,
            category="hotel",
            requirement=requirement,
            destination_location=destination_location,
            page_size=5,
        )

    async def _discover_attractions(
        self,
        requirement: TravelRequirement,
        destination_location: Coordinates | None,
        google_maps_client: GoogleMapsClient,
    ) -> list[PlaceRecommendation]:
        """
        Discover attraction candidates with Places Text Search.

        Args:
            requirement: The confirmed travel requirement snapshot.
            destination_location: Destination coordinates for location bias.
            google_maps_client: The Google Maps client group.

        Returns:
            Ranked attraction recommendations.
        """
        discovered_places: list[PlaceRecommendation] = []
        for query_template, included_type in ICONIC_DISCOVERY_QUERIES:
            query = query_template.format(destination=requirement.destination)
            discovered_places = _merge_unique_places(
                discovered_places,
                _boost_places(
                    await self._search_places(
                        google_maps_client=google_maps_client,
                        text_query=query,
                        category="attraction",
                        requirement=requirement,
                        destination_location=destination_location,
                        page_size=8,
                        included_type=included_type,
                    ),
                    ICONIC_DISCOVERY_SCORE_BOOST,
                ),
            )
        interests = requirement.interests[:3] or ["top"]
        for interest in interests:
            query = f"{interest} attractions in {requirement.destination}"
            discovered_places = _merge_unique_places(
                discovered_places,
                await self._search_places(
                    google_maps_client=google_maps_client,
                    text_query=query,
                    category="attraction",
                    requirement=requirement,
                    destination_location=destination_location,
                    page_size=8,
                    included_type=_included_type_for_interest(interest),
                ),
            )
        return sorted(discovered_places, key=lambda place: place.score, reverse=True)

    async def _discover_restaurants(
        self,
        requirement: TravelRequirement,
        destination_location: Coordinates | None,
        google_maps_client: GoogleMapsClient,
    ) -> list[PlaceRecommendation]:
        """
        Discover restaurant candidates with Places Text Search.

        Args:
            requirement: The confirmed travel requirement snapshot.
            destination_location: Destination coordinates for location bias.
            google_maps_client: The Google Maps client group.

        Returns:
            Ranked restaurant recommendations.
        """
        food_terms = requirement.food_preferences or ["local food"]
        query = f"{' '.join(food_terms[:3])} restaurants in {requirement.destination}"
        return await self._search_places(
            google_maps_client=google_maps_client,
            text_query=query,
            category="restaurant",
            requirement=requirement,
            destination_location=destination_location,
            page_size=10,
        )

    async def _search_places(
        self,
        google_maps_client: GoogleMapsClient,
        text_query: str,
        category: str,
        requirement: TravelRequirement,
        destination_location: Coordinates | None,
        page_size: int,
        included_type: str | None = None,
    ) -> list[PlaceRecommendation]:
        """
        Search, normalize, and rank Google Places candidates.

        Args:
            google_maps_client: The Google Maps client group.
            text_query: The Places Text Search query.
            category: The recommendation category.
            requirement: The confirmed travel requirement snapshot.
            destination_location: Destination coordinates for location bias.
            page_size: The requested result count.
            included_type: Optional Google place type used to bias results.

        Returns:
            Ranked place recommendations.
        """
        payload = await google_maps_client.places.search_text(
            text_query=text_query,
            page_size=page_size,
            field_mask=PLACES_RECOMMENDATION_FIELD_MASK,
            language_code=requirement.language,
            location_bias=_location_bias(destination_location),
            included_type=included_type,
        )
        places = [
            _place_from_google_payload(place, category, requirement)
            for place in payload.get("places", [])
            if _is_open_candidate(place)
        ]
        return sorted(places, key=lambda place: place.score, reverse=True)

    async def _build_days(
        self,
        requirement: TravelRequirement,
        primary_hotel: PlaceRecommendation,
        attractions: list[PlaceRecommendation],
        restaurants: list[PlaceRecommendation],
        google_maps_client: GoogleMapsClient,
    ) -> list[ItineraryDay]:
        """
        Build daily itinerary items and route summaries.

        Args:
            requirement: The confirmed travel requirement snapshot.
            primary_hotel: The hotel used as daily route anchor.
            attractions: Ranked attraction candidates.
            restaurants: Ranked restaurant candidates.
            google_maps_client: The Google Maps client group.

        Returns:
            Generated itinerary days.
        """
        day_count = _trip_day_count(requirement)
        attraction_count = _attractions_per_day(requirement.travel_pace)
        ranked_attractions = _nearby_ranked_places(
            primary_hotel, attractions, requirement
        )
        attraction_clusters = _rank_clusters(
            primary_hotel,
            _cluster_places(ranked_attractions, requirement),
        )
        preferred_themes = _preferred_themes(requirement, day_count)
        ranked_restaurants = _nearby_ranked_places(
            primary_hotel, restaurants, requirement
        )
        used_attraction_ids: set[str] = set()
        used_restaurant_ids: set[str] = set()
        days: list[ItineraryDay] = []
        for day_index in range(day_count):
            day_date = _day_date(requirement, day_index)
            preferred_theme = _preferred_theme_for_day(preferred_themes, day_index)
            day_cluster = _select_day_cluster(
                attraction_clusters,
                day_index,
                used_attraction_ids,
                preferred_theme,
            )
            day_attractions = _select_cluster_places(
                day_cluster,
                used_attraction_ids,
                attraction_count,
                preferred_theme,
                day_date,
            )
            day_theme = _cluster_theme(day_attractions, requirement)
            lunch, dinner = _select_daily_restaurants(
                day_attractions,
                ranked_restaurants,
                used_restaurant_ids,
                day_date,
            )
            items = _scheduled_items(day_attractions, lunch, dinner)
            route = await self._build_route_summary(
                requirement,
                primary_hotel,
                [item.place for item in items],
                google_maps_client,
            )
            while _route_exceeds_limit(route, requirement) and len(day_attractions) > 1:
                day_attractions = _without_farthest_place(
                    primary_hotel, day_attractions
                )
                lunch, dinner = _select_daily_restaurants(
                    day_attractions,
                    ranked_restaurants,
                    used_restaurant_ids,
                    day_date,
                )
                day_theme = _cluster_theme(day_attractions, requirement)
                items = _scheduled_items(day_attractions, lunch, dinner)
                route = await self._build_route_summary(
                    requirement,
                    primary_hotel,
                    [item.place for item in items],
                    google_maps_client,
                )
            for attraction in day_attractions:
                used_attraction_ids.add(attraction.place_id)
            used_restaurant_ids.add(lunch.place_id)
            used_restaurant_ids.add(dinner.place_id)
            days.append(
                ItineraryDay(
                    day_number=day_index + 1,
                    date=day_date,
                    theme=day_theme,
                    summary=_day_summary(day_index + 1, day_theme, day_attractions),
                    items=items,
                    route=route,
                )
            )
        return days

    async def _build_route_summary(
        self,
        requirement: TravelRequirement,
        primary_hotel: PlaceRecommendation,
        places: list[PlaceRecommendation],
        google_maps_client: GoogleMapsClient,
    ) -> RouteSummary | None:
        """
        Compute daily route legs between the hotel and scheduled places.

        Args:
            requirement: The confirmed travel requirement snapshot.
            primary_hotel: The hotel used as daily route anchor.
            places: The scheduled daily places.
            google_maps_client: The Google Maps client group.

        Returns:
            Aggregated route summary when coordinates are available.
        """
        routed_places = [primary_hotel, *places, primary_hotel]
        routed_places = [place for place in routed_places if place.location is not None]
        if len(routed_places) < 2:
            return None
        travel_mode = TRAVEL_MODES_BY_REQUIREMENT.get(
            requirement.transportation_mode or "", "DRIVE"
        )
        route_summary = RouteSummary(travel_mode=travel_mode)
        for origin, destination in zip(routed_places, routed_places[1:], strict=False):
            leg = await self._compute_route_leg(
                google_maps_client, origin, destination, travel_mode
            )
            route_summary.legs.append(leg)
            route_summary.distance_meters += leg.distance_meters or 0
            route_summary.duration_seconds += leg.duration_seconds or 0
        actual_modes = {leg.travel_mode for leg in route_summary.legs}
        if len(actual_modes) == 1:
            route_summary.travel_mode = next(iter(actual_modes))
        elif actual_modes:
            route_summary.travel_mode = "MIXED"
        return route_summary

    async def _compute_route_leg(
        self,
        google_maps_client: GoogleMapsClient,
        origin: PlaceRecommendation,
        destination: PlaceRecommendation,
        travel_mode: str,
    ) -> RouteLeg:
        """
        Compute one route leg between two places.

        Args:
            google_maps_client: The Google Maps client group.
            origin: The origin place.
            destination: The destination place.
            travel_mode: The Google Routes travel mode.

        Returns:
            The normalized route leg.
        """
        if origin.location is None or destination.location is None:
            return RouteLeg(
                origin_place_id=origin.place_id,
                destination_place_id=destination.place_id,
                travel_mode=travel_mode,
            )
        payload = await self._compute_route_payload(
            google_maps_client, origin, destination, travel_mode
        )
        if not payload.get("routes") and travel_mode != "DRIVE":
            return await self._compute_route_leg(
                google_maps_client, origin, destination, "DRIVE"
            )
        return _route_leg_from_payload(origin, destination, travel_mode, payload)

    async def _compute_route_payload(
        self,
        google_maps_client: GoogleMapsClient,
        origin: PlaceRecommendation,
        destination: PlaceRecommendation,
        travel_mode: str,
    ) -> dict[str, Any]:
        """
        Compute a raw route payload between two places.

        Args:
            google_maps_client: The Google Maps client group.
            origin: The origin place.
            destination: The destination place.
            travel_mode: The Google Routes travel mode.

        Returns:
            The raw Compute Routes payload.
        """
        if origin.location is None or destination.location is None:
            return {}
        routing_preference = "TRAFFIC_AWARE" if travel_mode == "DRIVE" else None
        departure_time = _transit_departure_time(travel_mode)
        return await google_maps_client.routes.compute_routes(
            origin.location.latitude,
            origin.location.longitude,
            destination.location.latitude,
            destination.location.longitude,
            travel_mode=travel_mode,
            routing_preference=routing_preference,
            departure_time=departure_time,
        )

    def _build_title(self, requirement: TravelRequirement) -> str:
        """
        Build a concise itinerary title.

        Args:
            requirement: The confirmed travel requirement snapshot.

        Returns:
            The itinerary title.
        """
        day_count = _trip_day_count(requirement)
        destination = requirement.destination or "Destination"
        return f"{day_count}-Day {destination} Travel Guide"

    def _render_guide(
        self,
        requirement: TravelRequirement,
        hotels: list[PlaceRecommendation],
        days: list[ItineraryDay],
    ) -> str:
        """
        Render a human-readable guide from structured itinerary data.

        Args:
            requirement: The confirmed travel requirement snapshot.
            hotels: The selected hotel recommendations.
            days: The generated itinerary days.

        Returns:
            The rendered guide in Markdown.
        """
        lines = [f"# {self._build_title(requirement)}", "", "## Hotels"]
        for hotel in hotels:
            lines.append(f"- {hotel.name} - {hotel.address or 'Address unavailable'}")
        lines.extend(["", "## Daily Plan"])
        for day in days:
            lines.append(f"### Day {day.day_number}")
            lines.append(f"Theme: {day.theme}")
            lines.append(day.summary)
            for item in day.items:
                lines.append(f"- {item.time} {item.type}: {item.place.name}")
            if day.route:
                lines.append(
                    f"- Route: {day.route.distance_meters} meters, "
                    f"{day.route.duration_seconds} seconds by {day.route.travel_mode}"
                )
        return "\n".join(lines)


def _location_bias(coordinates: Coordinates | None) -> dict[str, Any] | None:
    """
    Build a Places Text Search location bias.

    Args:
        coordinates: The destination coordinates.

    Returns:
        A Google location bias payload when coordinates are available.
    """
    if coordinates is None:
        return None
    return {
        "circle": {
            "center": {
                "latitude": coordinates.latitude,
                "longitude": coordinates.longitude,
            },
            "radius": 50000.0,
        }
    }


def _place_from_google_payload(
    payload: dict[str, Any], category: str, requirement: TravelRequirement
) -> PlaceRecommendation:
    """
    Normalize a Google Place payload.

    Args:
        payload: The Google Place payload.
        category: The recommendation category.
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The normalized place recommendation.
    """
    display_name = payload.get("displayName", {})
    name = display_name.get("text") or payload.get("id") or "Unnamed place"
    location = _coordinates_from_google_location(payload.get("location", {}))
    place = PlaceRecommendation(
        place_id=str(payload.get("id") or name),
        name=str(name),
        category=category,
        address=payload.get("formattedAddress"),
        location=location,
        google_maps_uri=payload.get("googleMapsUri"),
        rating=payload.get("rating"),
        user_rating_count=payload.get("userRatingCount"),
        price_level=payload.get("priceLevel"),
        types=payload.get("types") or [],
        regular_opening_hours=payload.get("regularOpeningHours"),
        current_opening_hours=payload.get("currentOpeningHours"),
    )
    place.score = _score_place(place, requirement)
    return place


def _merge_unique_places(
    current_places: list[PlaceRecommendation],
    new_places: list[PlaceRecommendation],
) -> list[PlaceRecommendation]:
    """
    Merge place lists by Google place ID while preserving first-seen order.

    Args:
        current_places: The existing places.
        new_places: The new places to merge.

    Returns:
        The merged unique places.
    """
    merged_places = list(current_places)
    seen_place_ids = {place.place_id for place in merged_places}
    for place in new_places:
        if place.place_id not in seen_place_ids:
            merged_places.append(place)
            seen_place_ids.add(place.place_id)
    return merged_places


def _boost_places(
    places: list[PlaceRecommendation], score_boost: float
) -> list[PlaceRecommendation]:
    """
    Return places with an additional discovery score boost.

    Args:
        places: The places to boost.
        score_boost: The score amount to add.

    Returns:
        Copies of the places with boosted scores.
    """
    return [
        place.model_copy(update={"score": place.score + score_boost})
        for place in places
    ]


def _coordinates_from_google_location(payload: dict[str, Any]) -> Coordinates | None:
    """
    Convert a Google location object to coordinates.

    Args:
        payload: The Google location payload.

    Returns:
        Coordinates when latitude and longitude are present.
    """
    latitude = payload.get("lat", payload.get("latitude"))
    longitude = payload.get("lng", payload.get("longitude"))
    if latitude is None or longitude is None:
        return None
    return Coordinates(latitude=float(latitude), longitude=float(longitude))


def _route_leg_from_payload(
    origin: PlaceRecommendation,
    destination: PlaceRecommendation,
    travel_mode: str,
    payload: dict[str, Any],
) -> RouteLeg:
    """
    Normalize a Compute Routes payload into a route leg.

    Args:
        origin: The origin place.
        destination: The destination place.
        travel_mode: The actual route travel mode.
        payload: The Compute Routes payload.

    Returns:
        The normalized route leg.
    """
    routes = payload.get("routes", [])
    if not routes:
        return RouteLeg(
            origin_place_id=origin.place_id,
            destination_place_id=destination.place_id,
            travel_mode=travel_mode,
        )
    route = routes[0]
    polyline = route.get("polyline", {})
    return RouteLeg(
        origin_place_id=origin.place_id,
        destination_place_id=destination.place_id,
        travel_mode=travel_mode,
        distance_meters=route.get("distanceMeters"),
        duration_seconds=_duration_to_seconds(route.get("duration")),
        encoded_polyline=polyline.get("encodedPolyline"),
    )


def _score_place(place: PlaceRecommendation, requirement: TravelRequirement) -> float:
    """
    Score a candidate place against user requirements.

    Args:
        place: The place recommendation.
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The deterministic candidate score.
    """
    rating_score = ((place.rating or 0.0) / 5.0) * 50.0
    review_score = min(place.user_rating_count or 0, 500) / 500.0 * 20.0
    budget_score = _budget_fit_score(place.price_level, requirement.budget_level)
    preference_score = _preference_score(place, requirement)
    return round(rating_score + review_score + budget_score + preference_score, 2)


def _nearby_ranked_places(
    primary_hotel: PlaceRecommendation,
    places: list[PlaceRecommendation],
    requirement: TravelRequirement,
) -> list[PlaceRecommendation]:
    """
    Filter distant candidates and rank remaining places around the hotel.

    Args:
        primary_hotel: The hotel used as the route anchor.
        places: The candidate places.
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The nearby candidates sorted by route-friendly score.
    """
    nearby_places = _filter_places_by_distance(
        primary_hotel.location,
        places,
        _candidate_distance_limit(requirement),
    )
    if not nearby_places:
        nearby_places = places
    return sorted(
        nearby_places,
        key=lambda place: _route_friendly_score(primary_hotel.location, place),
        reverse=True,
    )


def _filter_places_by_distance(
    reference: Coordinates | None,
    places: list[PlaceRecommendation],
    max_distance_meters: float,
) -> list[PlaceRecommendation]:
    """
    Keep places close enough to a reference point.

    Args:
        reference: The route anchor coordinates.
        places: The candidate places.
        max_distance_meters: The maximum straight-line distance.

    Returns:
        The candidates within the distance limit.
    """
    if reference is None:
        return places
    return [
        place
        for place in places
        if _distance_from_reference(reference, place) <= max_distance_meters
    ]


def _candidate_distance_limit(requirement: TravelRequirement) -> float:
    """
    Return a city candidate distance limit for the travel mode.

    Args:
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The maximum straight-line candidate distance in meters.
    """
    return CANDIDATE_DISTANCE_LIMITS_METERS.get(
        requirement.transportation_mode or "", 30000.0
    )


def _route_friendly_score(
    reference: Coordinates | None, place: PlaceRecommendation
) -> float:
    """
    Score a place by quality while penalizing distance from the route anchor.

    Args:
        reference: The route anchor coordinates.
        place: The candidate place.

    Returns:
        The route-friendly score.
    """
    distance_penalty = min(
        _distance_from_reference(reference, place) / 1000.0 * 2.0, 40
    )
    return place.score - distance_penalty


def _cluster_places(
    places: list[PlaceRecommendation], requirement: TravelRequirement
) -> list[list[PlaceRecommendation]]:
    """
    Group nearby places into route-friendly day clusters.

    Args:
        places: The ranked candidate places.
        requirement: The confirmed travel requirement snapshot.

    Returns:
        Nearby place clusters.
    """
    remaining_places = list(places)
    clusters: list[list[PlaceRecommendation]] = []
    radius_meters = _cluster_radius(requirement)
    while remaining_places:
        seed_place = remaining_places.pop(0)
        cluster = [seed_place]
        index = 0
        while index < len(remaining_places):
            candidate = remaining_places[index]
            if _distance_to_cluster(candidate, cluster) <= radius_meters:
                cluster.append(candidate)
                remaining_places.pop(index)
            else:
                index += 1
        clusters.append(cluster)
    return clusters


def _rank_clusters(
    primary_hotel: PlaceRecommendation,
    clusters: list[list[PlaceRecommendation]],
) -> list[list[PlaceRecommendation]]:
    """
    Rank clusters by size, quality, and distance from the hotel.

    Args:
        primary_hotel: The hotel used as the route anchor.
        clusters: The place clusters.

    Returns:
        Ranked place clusters.
    """
    return sorted(
        clusters,
        key=lambda cluster: _cluster_score(primary_hotel, cluster),
        reverse=True,
    )


def _cluster_score(
    primary_hotel: PlaceRecommendation, cluster: list[PlaceRecommendation]
) -> float:
    """
    Score a cluster by quality, size, and route anchor distance.

    Args:
        primary_hotel: The hotel used as the route anchor.
        cluster: The place cluster.

    Returns:
        The cluster score.
    """
    if not cluster:
        return 0.0
    average_score = sum(place.score for place in cluster) / len(cluster)
    size_bonus = min(len(cluster), 4) * 5.0
    center = _cluster_center(cluster)
    distance_penalty = (
        _distance_between_coordinates(primary_hotel.location, center) / 1000.0
    )
    return average_score + size_bonus - distance_penalty


def _cluster_radius(requirement: TravelRequirement) -> float:
    """
    Return the clustering radius for the requested travel mode.

    Args:
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The cluster radius in meters.
    """
    return CLUSTER_RADIUS_METERS.get(requirement.transportation_mode or "", 9000.0)


def _preferred_themes(requirement: TravelRequirement, day_count: int) -> list[str]:
    """
    Resolve user interests to ordered travel themes.

    Args:
        requirement: The confirmed travel requirement snapshot.
        day_count: The itinerary day count.

    Returns:
        Unique preferred themes in user-interest order.
    """
    interest_themes: list[str] = []
    if day_count >= 3:
        interest_themes.append(CLASSIC_SIGHTSEEING_THEME)
    for interest in requirement.interests:
        theme = _theme_for_interest(interest)
        if theme and theme not in interest_themes:
            interest_themes.append(theme)
    if day_count < 3:
        return interest_themes
    food_themes = [theme for theme in interest_themes if theme == "food and markets"]
    primary_themes = [theme for theme in interest_themes if theme != "food and markets"]
    return [*primary_themes, *food_themes]


def _theme_for_interest(interest: str) -> str | None:
    """
    Resolve one user interest to a mainstream travel theme.

    Args:
        interest: The raw user interest.

    Returns:
        A theme label when the interest maps to one.
    """
    normalized_interest = interest.lower()
    for theme, aliases in THEME_INTEREST_ALIASES.items():
        if any(alias in normalized_interest for alias in aliases):
            return theme
    return None


def _preferred_theme_for_day(preferred_themes: list[str], day_index: int) -> str | None:
    """
    Return the preferred theme for a day.

    Args:
        preferred_themes: Ordered preferred themes.
        day_index: The zero-based day index.

    Returns:
        The preferred theme, cycling when the trip has more days than themes.
    """
    if not preferred_themes:
        return None
    return preferred_themes[day_index % len(preferred_themes)]


def _select_day_cluster(
    clusters: list[list[PlaceRecommendation]],
    day_index: int,
    used_place_ids: set[str],
    preferred_theme: str | None,
) -> list[PlaceRecommendation]:
    """
    Select the best cluster for a day while avoiding prior assignments.

    Args:
        clusters: The ranked place clusters.
        day_index: The zero-based day index.
        used_place_ids: Place IDs already assigned to prior days.
        preferred_theme: The preferred theme for this day.

    Returns:
        The selected cluster.
    """
    if not clusters:
        return []
    available_clusters = [
        cluster
        for cluster in clusters
        if any(place.place_id not in used_place_ids for place in cluster)
    ]
    if preferred_theme and available_clusters:
        best_cluster = max(
            available_clusters,
            key=lambda cluster: _theme_context_score(preferred_theme, cluster),
        )
        if _theme_context_score(preferred_theme, best_cluster) > 0:
            return best_cluster
    for cluster in clusters:
        if any(place.place_id not in used_place_ids for place in cluster):
            return cluster
    return clusters[day_index % len(clusters)]


def _theme_ranked_places(
    places: list[PlaceRecommendation], preferred_theme: str
) -> list[PlaceRecommendation]:
    """
    Rank places by how strongly they match a preferred theme.

    Args:
        places: The candidate places.
        preferred_theme: The preferred theme for this day.

    Returns:
        Places ordered by theme affinity, preserving prior order for ties.
    """
    if not places:
        return []
    if max(_theme_context_score(preferred_theme, [place]) for place in places) == 0:
        return places
    return sorted(
        places,
        key=lambda place: _theme_context_score(preferred_theme, [place]),
        reverse=True,
    )


def _select_cluster_places(
    cluster: list[PlaceRecommendation],
    used_place_ids: set[str],
    count: int,
    preferred_theme: str | None,
    day_date: str | None,
) -> list[PlaceRecommendation]:
    """
    Select unused places from a day cluster.

    Args:
        cluster: The selected day cluster.
        used_place_ids: Place IDs already assigned to prior days.
        count: The requested selection count.
        preferred_theme: The preferred theme for this day.
        day_date: The concrete itinerary date when known.

    Returns:
        The selected places.
    """
    available_places = [
        place for place in cluster if place.place_id not in used_place_ids
    ]
    if not available_places:
        available_places = cluster
    if preferred_theme:
        available_places = _theme_ranked_places(available_places, preferred_theme)
    selected_places: list[PlaceRecommendation] = []
    remaining_places = list(available_places)
    for time_text in ATTRACTION_TIME_SLOTS[: max(1, count)]:
        if not remaining_places:
            break
        selected_place = _first_open_place(remaining_places, day_date, time_text)
        selected_places.append(selected_place)
        remaining_places = [
            place
            for place in remaining_places
            if place.place_id != selected_place.place_id
        ]
    return selected_places


def _cluster_theme(
    cluster: list[PlaceRecommendation], requirement: TravelRequirement
) -> str:
    """
    Infer a human-readable theme for a place cluster.

    Args:
        cluster: The selected day cluster.
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The inferred cluster theme.
    """
    if not cluster:
        return "local highlights"
    place_type_counts = _place_type_counts(cluster)
    place_text = _cluster_searchable_text(cluster)
    interest_text = " ".join(requirement.interests).lower()
    theme_scores = {
        theme: _theme_score(theme, place_type_counts, place_text, interest_text)
        for theme in THEME_TYPE_KEYWORDS
    }
    best_theme, best_score = max(theme_scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "city highlights"
    if _should_prefer_classic_sightseeing(theme_scores, best_theme):
        return CLASSIC_SIGHTSEEING_THEME
    if _is_waterfront_dining_theme(theme_scores, best_theme):
        return WATERFRONT_DINING_THEME
    combined_theme = _combined_theme(theme_scores, best_theme)
    if combined_theme:
        return combined_theme
    return best_theme


def _should_prefer_classic_sightseeing(
    theme_scores: dict[str, float], best_theme: str
) -> bool:
    """
    Return whether classic sightseeing should override a close theme score.

    Args:
        theme_scores: The calculated theme scores.
        best_theme: The highest scoring base theme.

    Returns:
        True when classic sightseeing is close enough to be more useful.
    """
    classic_score = theme_scores[CLASSIC_SIGHTSEEING_THEME]
    best_score = theme_scores[best_theme]
    return (
        best_theme != CLASSIC_SIGHTSEEING_THEME
        and classic_score >= COMBINED_THEME_MIN_SCORE
        and best_score - classic_score <= COMBINED_THEME_MAX_SCORE_GAP
    )


def _combined_theme(theme_scores: dict[str, float], best_theme: str) -> str | None:
    """
    Return a mixed theme when a day has two strong place contexts.

    Args:
        theme_scores: The calculated theme scores.
        best_theme: The highest scoring base theme.

    Returns:
        A combined theme label when appropriate.
    """
    ranked_themes = sorted(theme_scores.items(), key=lambda item: item[1], reverse=True)
    second_theme, second_score = ranked_themes[1]
    best_score = theme_scores[best_theme]
    if second_score < COMBINED_THEME_MIN_SCORE:
        return None
    if best_score - second_score > COMBINED_THEME_MAX_SCORE_GAP:
        return None
    theme_pairs = {
        frozenset({"arts and museums", "food and markets"}): "food and arts",
        frozenset({"arts and museums", "waterfront and views"}): "waterfront and arts",
        frozenset({"food and markets", "parks and gardens"}): "markets and gardens",
        frozenset(
            {"heritage and landmarks", "parks and gardens"}
        ): "parks and heritage",
        frozenset(
            {"heritage and landmarks", "waterfront and views"}
        ): "waterfront heritage",
        frozenset(
            {"parks and gardens", "waterfront and views"}
        ): "waterfront parks and views",
    }
    return theme_pairs.get(frozenset({best_theme, second_theme}))


def _is_waterfront_dining_theme(
    theme_scores: dict[str, float], best_theme: str
) -> bool:
    """
    Return whether a food-led cluster should be labeled as waterfront dining.

    Args:
        theme_scores: The calculated theme scores.
        best_theme: The highest scoring base theme.

    Returns:
        True when the cluster is food-led with clear waterfront context.
    """
    return (
        best_theme == "food and markets"
        and theme_scores["food and markets"] >= WATERFRONT_DINING_MIN_FOOD_SCORE
        and theme_scores["waterfront and views"]
        >= WATERFRONT_DINING_MIN_WATERFRONT_SCORE
    )


def _theme_context_score(theme: str, cluster: list[PlaceRecommendation]) -> float:
    """
    Score a cluster against a theme without user-interest bias.

    Args:
        theme: The travel theme label.
        cluster: The selected place cluster.

    Returns:
        The place-only theme score.
    """
    return _theme_score(
        theme,
        _place_type_counts(cluster),
        _cluster_searchable_text(cluster),
        "",
    )


def _theme_score(
    theme: str,
    place_type_counts: dict[str, int],
    place_text: str,
    interest_text: str,
) -> float:
    """
    Score how strongly a cluster matches a travel theme.

    Args:
        theme: The travel theme label.
        place_type_counts: Counts of Google place types in the cluster.
        place_text: Searchable place names.
        interest_text: Searchable user interest text.

    Returns:
        The weighted theme score.
    """
    type_score = sum(
        place_type_counts.get(type_name, 0) * THEME_TYPE_WEIGHT
        for type_name in THEME_TYPE_KEYWORDS[theme]
    )
    text_score = sum(
        place_text.count(keyword) * THEME_TEXT_WEIGHT
        for keyword in THEME_TEXT_KEYWORDS[theme]
    )
    interest_score = 0.0
    if any(keyword in interest_text for keyword in THEME_INTEREST_ALIASES[theme]):
        interest_score = THEME_INTEREST_WEIGHT
    return type_score + text_score + interest_score


def _place_type_counts(cluster: list[PlaceRecommendation]) -> dict[str, int]:
    """
    Count normalized Google place types across a cluster.

    Args:
        cluster: The selected place cluster.

    Returns:
        A mapping from place type to count.
    """
    counts: dict[str, int] = {}
    for place in cluster:
        for place_type in place.types:
            normalized_type = place_type.lower()
            counts[normalized_type] = counts.get(normalized_type, 0) + 1
    return counts


def _cluster_searchable_text(cluster: list[PlaceRecommendation]) -> str:
    """
    Build searchable text from place names.

    Args:
        cluster: The selected place cluster.

    Returns:
        Lowercase text for theme keyword matching.
    """
    return " ".join(place.name.lower() for place in cluster)


def _cluster_center(cluster: list[PlaceRecommendation]) -> Coordinates | None:
    """
    Calculate the coordinate center of a place cluster.

    Args:
        cluster: The place cluster.

    Returns:
        The cluster center when coordinates are available.
    """
    coordinates = [place.location for place in cluster if place.location is not None]
    if not coordinates:
        return None
    return Coordinates(
        latitude=sum(coordinate.latitude for coordinate in coordinates)
        / len(coordinates),
        longitude=sum(coordinate.longitude for coordinate in coordinates)
        / len(coordinates),
    )


def _distance_to_cluster(
    place: PlaceRecommendation, cluster: list[PlaceRecommendation]
) -> float:
    """
    Return the shortest distance from a place to any place in a cluster.

    Args:
        place: The candidate place.
        cluster: The place cluster.

    Returns:
        The shortest distance in meters.
    """
    if not cluster:
        return 0.0
    return min(
        _distance_between_places(place, cluster_place) for cluster_place in cluster
    )


def _select_daily_restaurants(
    day_attractions: list[PlaceRecommendation],
    restaurants: list[PlaceRecommendation],
    used_restaurant_ids: set[str],
    day_date: str | None,
) -> tuple[PlaceRecommendation, PlaceRecommendation]:
    """
    Select restaurants near the day's attraction cluster.

    Args:
        day_attractions: The selected attractions for the day.
        restaurants: The ranked restaurant candidates.
        used_restaurant_ids: Restaurant IDs already assigned to prior days.
        day_date: The concrete itinerary date when known.

    Returns:
        Lunch and dinner restaurant recommendations.
    """
    available_restaurants = [
        restaurant
        for restaurant in restaurants
        if restaurant.place_id not in used_restaurant_ids
    ]
    if len(available_restaurants) < 2:
        available_restaurants = restaurants
    lunch_anchor = day_attractions[0].location if day_attractions else None
    dinner_anchor = day_attractions[-1].location if day_attractions else None
    lunch_options = _places_open_at(available_restaurants, day_date, LUNCH_TIME)
    if not lunch_options:
        lunch_options = available_restaurants
    lunch = _nearest_place(lunch_anchor, lunch_options)
    dinner_options = [
        restaurant
        for restaurant in available_restaurants
        if restaurant.place_id != lunch.place_id
    ]
    if not dinner_options:
        dinner_options = restaurants
    open_dinner_options = _places_open_at(dinner_options, day_date, DINNER_TIME)
    if open_dinner_options:
        dinner_options = open_dinner_options
    dinner = _nearest_place(dinner_anchor, dinner_options)
    return lunch, dinner


def _first_open_place(
    places: list[PlaceRecommendation], day_date: str | None, time_text: str
) -> PlaceRecommendation:
    """
    Return the first place open at the requested itinerary time.

    Args:
        places: The ranked candidate places.
        day_date: The concrete itinerary date when known.
        time_text: The local time in HH:MM format.

    Returns:
        The first open place, or the first candidate when hours are unavailable.
    """
    for place in places:
        if _place_is_open_at(place, day_date, time_text):
            return place
    return places[0]


def _places_open_at(
    places: list[PlaceRecommendation], day_date: str | None, time_text: str
) -> list[PlaceRecommendation]:
    """
    Filter places to those open at the requested itinerary time.

    Args:
        places: The candidate places.
        day_date: The concrete itinerary date when known.
        time_text: The local time in HH:MM format.

    Returns:
        Places open at the requested time.
    """
    return [place for place in places if _place_is_open_at(place, day_date, time_text)]


def _place_is_open_at(
    place: PlaceRecommendation, day_date: str | None, time_text: str
) -> bool:
    """
    Return whether a place is expected to be open at a local date and time.

    Args:
        place: The candidate place.
        day_date: The concrete itinerary date when known.
        time_text: The local time in HH:MM format.

    Returns:
        True when hours are unavailable or the place is open at that time.
    """
    if day_date is None:
        return True
    opening_hours = place.regular_opening_hours or place.current_opening_hours
    if opening_hours is None:
        return True
    periods = opening_hours.get("periods")
    if periods is None:
        return True
    if not periods:
        return False
    requested_day = _google_weekday(day_date)
    requested_minutes = _time_to_minutes(time_text)
    return any(
        _period_contains_time(period, requested_day, requested_minutes)
        for period in periods
    )


def _period_contains_time(
    period: dict[str, Any], requested_day: int, requested_minutes: int
) -> bool:
    """
    Return whether an opening-hours period contains a requested time.

    Args:
        period: The Google Places opening-hours period.
        requested_day: The Google weekday value.
        requested_minutes: The requested minute of day.

    Returns:
        True when the period contains the requested time.
    """
    open_point = period.get("open", {})
    close_point = period.get("close")
    if close_point is None:
        return True
    open_day = int(open_point.get("day", requested_day))
    close_day = int(close_point.get("day", requested_day))
    open_minutes = _point_minutes(open_point)
    close_minutes = _point_minutes(close_point)
    requested_value = requested_day * 1440 + requested_minutes
    open_value = open_day * 1440 + open_minutes
    close_value = close_day * 1440 + close_minutes
    if close_value <= open_value:
        close_value += 7 * 1440
        if requested_value < open_value:
            requested_value += 7 * 1440
    return open_value <= requested_value < close_value


def _google_weekday(day_date: str) -> int:
    """
    Convert an ISO date to a Google Places weekday number.

    Args:
        day_date: The ISO date string.

    Returns:
        The Google weekday value, where 0 is Sunday.
    """
    parsed_date = date.fromisoformat(day_date)
    return (parsed_date.weekday() + OPENING_HOURS_DAY_OFFSET) % 7


def _point_minutes(point: dict[str, Any]) -> int:
    """
    Convert a Google opening-hours point to minutes after midnight.

    Args:
        point: The Google opening-hours point.

    Returns:
        Minutes after midnight.
    """
    return int(point.get("hour", 0)) * 60 + int(point.get("minute", 0))


def _time_to_minutes(time_text: str) -> int:
    """
    Convert HH:MM text to minutes after midnight.

    Args:
        time_text: The local time in HH:MM format.

    Returns:
        Minutes after midnight.
    """
    hour_text, minute_text = time_text.split(":", maxsplit=1)
    return int(hour_text) * 60 + int(minute_text)


def _nearest_place(
    reference: Coordinates | None, places: list[PlaceRecommendation]
) -> PlaceRecommendation:
    """
    Select the nearest place to a reference, using score as a tie-breaker.

    Args:
        reference: The reference coordinates.
        places: The candidate places.

    Returns:
        The selected place.
    """
    return min(
        places,
        key=lambda place: (
            _distance_from_reference(reference, place),
            -place.score,
        ),
    )


def _without_farthest_place(
    primary_hotel: PlaceRecommendation, places: list[PlaceRecommendation]
) -> list[PlaceRecommendation]:
    """
    Remove the farthest selected place from the hotel.

    Args:
        primary_hotel: The hotel used as the route anchor.
        places: The selected places.

    Returns:
        The selected places without the farthest candidate.
    """
    if len(places) <= 1:
        return places
    farthest_place = max(
        places,
        key=lambda place: _distance_from_reference(primary_hotel.location, place),
    )
    return [place for place in places if place.place_id != farthest_place.place_id]


def _route_exceeds_limit(
    route: RouteSummary | None, requirement: TravelRequirement
) -> bool:
    """
    Return whether a daily route exceeds the configured time budget.

    Args:
        route: The computed route summary.
        requirement: The confirmed travel requirement snapshot.

    Returns:
        True when the route should be reduced.
    """
    if route is None:
        return False
    duration_limit = DAILY_ROUTE_DURATION_LIMITS_SECONDS.get(
        requirement.transportation_mode or "", 10800
    )
    distance_limit = DAILY_ROUTE_DISTANCE_LIMITS_METERS.get(
        requirement.transportation_mode or "", 60000
    )
    return (
        route.duration_seconds > duration_limit
        or route.distance_meters > distance_limit
    )


def _distance_from_reference(
    reference: Coordinates | None, place: PlaceRecommendation
) -> float:
    """
    Return straight-line distance from a reference point to a place.

    Args:
        reference: The reference coordinates.
        place: The candidate place.

    Returns:
        The straight-line distance in meters.
    """
    if reference is None or place.location is None:
        return 0.0
    return _haversine_distance_meters(reference, place.location)


def _distance_between_places(
    first_place: PlaceRecommendation, second_place: PlaceRecommendation
) -> float:
    """
    Return straight-line distance between two places.

    Args:
        first_place: The first place.
        second_place: The second place.

    Returns:
        The straight-line distance in meters.
    """
    return _distance_between_coordinates(first_place.location, second_place.location)


def _distance_between_coordinates(
    first_coordinates: Coordinates | None, second_coordinates: Coordinates | None
) -> float:
    """
    Return straight-line distance between two optional coordinate pairs.

    Args:
        first_coordinates: The first coordinates.
        second_coordinates: The second coordinates.

    Returns:
        The straight-line distance in meters.
    """
    if first_coordinates is None or second_coordinates is None:
        return 0.0
    return _haversine_distance_meters(first_coordinates, second_coordinates)


def _haversine_distance_meters(
    first_coordinates: Coordinates, second_coordinates: Coordinates
) -> float:
    """
    Calculate straight-line distance between two coordinates.

    Args:
        first_coordinates: The first coordinates.
        second_coordinates: The second coordinates.

    Returns:
        The distance in meters.
    """
    earth_radius_meters = 6371000.0
    first_latitude = radians(first_coordinates.latitude)
    second_latitude = radians(second_coordinates.latitude)
    latitude_delta = radians(second_coordinates.latitude - first_coordinates.latitude)
    longitude_delta = radians(
        second_coordinates.longitude - first_coordinates.longitude
    )
    haversine_value = (
        sin(latitude_delta / 2) ** 2
        + cos(first_latitude) * cos(second_latitude) * sin(longitude_delta / 2) ** 2
    )
    return earth_radius_meters * 2 * asin(sqrt(haversine_value))


def _budget_fit_score(price_level: str | None, budget_level: str | None) -> float:
    """
    Score how well a place price level fits the user budget.

    Args:
        price_level: The Google price level.
        budget_level: The user budget level.

    Returns:
        The price fit score.
    """
    if price_level is None or budget_level is None:
        return 5.0
    expected_levels = PRICE_LEVELS_BY_BUDGET.get(budget_level, [])
    if price_level in expected_levels:
        return 15.0
    if budget_level == "medium" and price_level == "PRICE_LEVEL_INEXPENSIVE":
        return 10.0
    return 0.0


def _preference_score(
    place: PlaceRecommendation, requirement: TravelRequirement
) -> float:
    """
    Score how well a place matches explicit interests.

    Args:
        place: The place recommendation.
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The preference match score.
    """
    searchable_text = " ".join([place.name, *place.types]).lower()
    matches = [
        interest for interest in requirement.interests if interest in searchable_text
    ]
    if not matches:
        return 0.0
    return min(len(matches) * 5.0, 15.0)


def _budget_search_text(budget_level: str | None) -> str:
    """
    Convert budget level to search text.

    Args:
        budget_level: The user budget level.

    Returns:
        Search query text.
    """
    if budget_level == "low":
        return "budget-friendly"
    if budget_level == "high":
        return "luxury"
    return "moderate"


def _included_type_for_interest(interest: str) -> str | None:
    """
    Resolve a user interest to a Google Places included type.

    Args:
        interest: The raw user interest.

    Returns:
        A Google Places type when the interest maps to one.
    """
    normalized_interest = interest.lower()
    for keyword, included_type in INTEREST_INCLUDED_TYPES.items():
        if keyword in normalized_interest:
            return included_type
    return None


def _is_open_candidate(payload: dict[str, Any]) -> bool:
    """
    Return whether a Google Place payload is usable.

    Args:
        payload: The Google Place payload.

    Returns:
        True when the candidate should be considered.
    """
    return payload.get("businessStatus") != "CLOSED_PERMANENTLY"


def _trip_day_count(requirement: TravelRequirement) -> int:
    """
    Determine the number of itinerary days.

    Args:
        requirement: The confirmed travel requirement snapshot.

    Returns:
        The bounded trip day count.
    """
    if requirement.trip_length_days:
        return min(requirement.trip_length_days, 7)
    return min(_day_count_from_dates(requirement.trip_dates), 7)


def _day_count_from_dates(trip_dates: str | None) -> int:
    """
    Parse day count from a free-form date range.

    Args:
        trip_dates: The raw trip date text.

    Returns:
        The parsed day count, defaulting to one.
    """
    if not trip_dates:
        return 1
    matches = re.findall(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", trip_dates)
    if len(matches) < 2:
        return 1
    start_date = date.fromisoformat(matches[0].replace("/", "-"))
    end_date = date.fromisoformat(matches[1].replace("/", "-"))
    return max((end_date - start_date).days + 1, 1)


def _day_date(requirement: TravelRequirement, day_index: int) -> str | None:
    """
    Return the concrete date for a day when trip dates are available.

    Args:
        requirement: The confirmed travel requirement snapshot.
        day_index: The zero-based day index.

    Returns:
        The ISO date string when available.
    """
    if not requirement.trip_dates:
        return None
    matches = re.findall(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", requirement.trip_dates)
    if not matches:
        return None
    start_date = date.fromisoformat(matches[0].replace("/", "-"))
    return start_date.fromordinal(start_date.toordinal() + day_index).isoformat()


def _attractions_per_day(travel_pace: str | None) -> int:
    """
    Return the target number of major attractions per day.

    Args:
        travel_pace: The user travel pace.

    Returns:
        The attraction count per day.
    """
    if travel_pace == "relaxed":
        return 2
    if travel_pace == "packed":
        return 4
    return 3


def _scheduled_items(
    attractions: list[PlaceRecommendation],
    lunch: PlaceRecommendation,
    dinner: PlaceRecommendation,
) -> list[ItineraryItem]:
    """
    Build scheduled itinerary items for one day.

    Args:
        attractions: The selected attractions.
        lunch: The lunch restaurant.
        dinner: The dinner restaurant.

    Returns:
        The scheduled itinerary items.
    """
    schedule = [
        (ATTRACTION_TIME_SLOTS[0], ItineraryItemType.ATTRACTION, 105),
        (LUNCH_TIME, ItineraryItemType.LUNCH, 75),
        (ATTRACTION_TIME_SLOTS[1], ItineraryItemType.ATTRACTION, 105),
        (ATTRACTION_TIME_SLOTS[2], ItineraryItemType.ATTRACTION, 75),
        (DINNER_TIME, ItineraryItemType.DINNER, 90),
    ]
    attraction_index = 0
    items: list[ItineraryItem] = []
    for time_text, item_type, duration_minutes in schedule:
        if item_type == ItineraryItemType.LUNCH:
            place = lunch
        elif item_type == ItineraryItemType.DINNER:
            place = dinner
        elif attraction_index < len(attractions):
            place = attractions[attraction_index]
            attraction_index += 1
        else:
            continue
        items.append(
            ItineraryItem(
                time=time_text,
                type=item_type,
                place=place,
                duration_minutes=duration_minutes,
            )
        )
    return items


def _day_summary(
    day_number: int,
    theme: str,
    attractions: list[PlaceRecommendation],
) -> str:
    """
    Build a short daily itinerary summary.

    Args:
        day_number: The itinerary day number.
        theme: The inferred day theme.
        attractions: The selected attractions for the day.

    Returns:
        The daily summary.
    """
    if attractions:
        focus = ", ".join(attraction.name for attraction in attractions[:2])
    else:
        focus = theme
    return f"Day {day_number} focuses on {theme}: {focus}."


def _transit_departure_time(travel_mode: str) -> str | None:
    """
    Return an RFC3339 departure time for transit route requests.

    Args:
        travel_mode: The Google Routes travel mode.

    Returns:
        A UTC departure time when the route mode is transit.
    """
    if travel_mode != "TRANSIT":
        return None
    departure_time = datetime.now(tz=UTC) + timedelta(hours=1)
    return departure_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_to_seconds(duration: str | None) -> int | None:
    """
    Convert a Google duration string to seconds.

    Args:
        duration: A Google duration string such as `165s`.

    Returns:
        The duration in seconds when parseable.
    """
    if not duration:
        return None
    match = re.fullmatch(r"(\d+)s", duration)
    if not match:
        return None
    return int(match.group(1))
