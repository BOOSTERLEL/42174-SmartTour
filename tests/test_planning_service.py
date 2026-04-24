"""Tests for itinerary planning service."""

import pytest

from smartour.application.planning_service import (
    PlanningService,
    _cluster_places,
    _cluster_theme,
    _place_is_open_at,
    _preferred_themes,
)
from smartour.core.errors import PlanningInputError
from smartour.domain.conversation import Conversation
from smartour.domain.itinerary import Coordinates, PlaceRecommendation
from smartour.domain.requirement import Travelers, TravelRequirement
from smartour.infrastructure.repositories.conversations import (
    InMemoryConversationRepository,
)
from smartour.infrastructure.repositories.itineraries import InMemoryItineraryRepository
from smartour.integrations.google_maps.client import GoogleMapsClient


class FakeGeocodingClient:
    """
    Fake Geocoding client for planning tests.
    """

    async def geocode(
        self, address: str, language: str | None = None, region: str | None = None
    ) -> dict:
        """
        Return a stable geocoding response.

        Args:
            address: The address query.
            language: The requested language.
            region: The requested region.

        Returns:
            A fake Geocoding API response.
        """
        return {
            "results": [
                {
                    "formatted_address": address,
                    "geometry": {"location": {"lat": 35.6764, "lng": 139.65}},
                }
            ],
            "status": "OK",
        }


class FakePlacesClient:
    """
    Fake Places client for planning tests.
    """

    def __init__(self) -> None:
        """
        Initialize the fake Places client.
        """
        self.included_types: list[str | None] = []

    async def search_text(
        self,
        text_query: str,
        page_size: int = 5,
        field_mask: str = "",
        language_code: str | None = None,
        region_code: str | None = None,
        location_bias: dict | None = None,
        included_type: str | None = None,
    ) -> dict:
        """
        Return category-specific fake Places responses.

        Args:
            text_query: The search query.
            page_size: The requested page size.
            field_mask: The requested field mask.
            language_code: The requested language code.
            region_code: The requested region code.
            location_bias: The requested location bias.
            included_type: The requested type filter.

        Returns:
            A fake Places Text Search response.
        """
        self.included_types.append(included_type)
        if "hotels" in text_query:
            places = [
                _place("hotel_1", "Central Hotel", "hotel", 35.67, 139.7, 4.5, 300),
                _place("hotel_2", "Budget Hotel", "hotel", 35.66, 139.71, 4.0, 100),
            ]
        elif "restaurants" in text_query:
            places = [
                _place(
                    "restaurant_1", "Ramen House", "restaurant", 35.68, 139.72, 4.6, 400
                ),
                _place(
                    "restaurant_2", "Sushi Bar", "restaurant", 35.69, 139.73, 4.4, 350
                ),
                _place(
                    "restaurant_3",
                    "Remote Seafood",
                    "restaurant",
                    36.4,
                    140.4,
                    5.0,
                    500,
                ),
            ]
        else:
            places = [
                _place("attraction_1", "Food Museum", "museum", 35.7, 139.74, 4.7, 500),
                _place("attraction_2", "City Garden", "park", 35.71, 139.75, 4.3, 250),
                _place(
                    "attraction_3",
                    "Remote Mountain",
                    "tourist_attraction",
                    36.5,
                    140.5,
                    5.0,
                    500,
                ),
            ]
        return {"places": places[:page_size]}


class FakeRoutesClient:
    """
    Fake Routes client for planning tests.
    """

    def __init__(self) -> None:
        """
        Initialize the fake routes client.
        """
        self.routing_preferences: list[str | None] = []
        self.travel_modes: list[str] = []

    async def compute_routes(
        self,
        origin_latitude: float,
        origin_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
        travel_mode: str = "DRIVE",
        routing_preference: str | None = "TRAFFIC_AWARE",
        departure_time: str | None = None,
        field_mask: str = "",
    ) -> dict:
        """
        Return a stable route response.

        Args:
            origin_latitude: The origin latitude.
            origin_longitude: The origin longitude.
            destination_latitude: The destination latitude.
            destination_longitude: The destination longitude.
            travel_mode: The requested travel mode.
            routing_preference: The requested routing preference.
            departure_time: The requested departure time.
            field_mask: The requested field mask.

        Returns:
            A fake Routes API response.
        """
        self.travel_modes.append(travel_mode)
        self.routing_preferences.append(routing_preference)
        return {
            "routes": [
                {
                    "distanceMeters": 1000,
                    "duration": "600s",
                    "polyline": {"encodedPolyline": "encoded"},
                }
            ]
        }


class FakeTimeZoneClient:
    """
    Fake Time Zone client placeholder.
    """


@pytest.mark.asyncio
async def test_planning_service_generates_structured_itinerary() -> None:
    """
    Verify that planning creates hotels, daily items, and route summaries.
    """
    conversation_repository = InMemoryConversationRepository()
    itinerary_repository = InMemoryItineraryRepository()
    conversation = Conversation(requirement=_complete_requirement())
    conversation_repository.save(conversation)
    places_client = FakePlacesClient()
    routes_client = FakeRoutesClient()
    google_maps_client = GoogleMapsClient(
        places=places_client,
        routes=routes_client,
        geocoding=FakeGeocodingClient(),
        timezone=FakeTimeZoneClient(),
    )
    service = PlanningService(conversation_repository, itinerary_repository)

    itinerary = await service.generate_for_conversation(
        conversation.id, google_maps_client
    )

    assert itinerary is not None
    assert itinerary.title == "1-Day Tokyo Travel Guide"
    assert itinerary.destination_location is not None
    assert itinerary.hotels[0].name == "Central Hotel"
    assert len(itinerary.days) == 1
    assert itinerary.days[0].theme == "arts and museums"
    assert [item.place.name for item in itinerary.days[0].items] == [
        "Food Museum",
        "Sushi Bar",
        "City Garden",
        "Ramen House",
    ]
    assert "Remote Mountain" not in {
        item.place.name for item in itinerary.days[0].items
    }
    assert "Remote Seafood" not in {item.place.name for item in itinerary.days[0].items}
    assert itinerary.days[0].route is not None
    assert itinerary.days[0].route.distance_meters == 5000
    assert itinerary.days[0].route.duration_seconds == 3000
    assert routes_client.travel_modes == ["TRANSIT"] * 5
    assert routes_client.routing_preferences == [None] * 5
    assert places_client.included_types == [
        None,
        None,
        "tourist_attraction",
        "market",
        "museum",
        None,
    ]
    assert service.get_itinerary(itinerary.id) is not None


def test_cluster_places_groups_nearby_attractions_by_area() -> None:
    """
    Verify that day clustering separates distinct geographic areas.
    """
    requirement = _complete_requirement()
    places = [
        _recommendation("museum_1", "Museum One", "museum", 35.700, 139.740),
        _recommendation("museum_2", "Gallery Two", "art_gallery", 35.705, 139.745),
        _recommendation("park_1", "Garden One", "park", 35.760, 139.760),
        _recommendation("park_2", "Garden Two", "garden", 35.765, 139.765),
    ]

    clusters = _cluster_places(places, requirement)

    assert len(clusters) == 2
    assert {place.name for place in clusters[0]} == {"Museum One", "Gallery Two"}
    assert _cluster_theme(clusters[0], requirement) == "arts and museums"


def test_cluster_theme_prefers_place_context_over_global_interests() -> None:
    """
    Verify that global interests do not dominate clear place context.
    """
    requirement = _complete_requirement()
    places = [
        _recommendation(
            "wharf_1",
            "Cockle Bay Wharf",
            "tourist_attraction",
            35.700,
            139.740,
            types=[
                "tourist_attraction",
                "food_store",
                "bar",
                "restaurant",
                "point_of_interest",
            ],
        ),
        _recommendation(
            "wharf_2",
            "King Street Wharf Darling Harbour",
            "tourist_attraction",
            35.705,
            139.745,
            types=["tourist_attraction", "point_of_interest"],
        ),
    ]

    assert _cluster_theme(places, requirement) == "waterfront and dining"


def test_cluster_theme_uses_google_place_types() -> None:
    """
    Verify that Google place types can identify mainstream travel themes.
    """
    requirement = _complete_requirement()
    places = [
        _recommendation(
            "aquarium_1",
            "SEA LIFE Aquarium",
            "tourist_attraction",
            35.700,
            139.740,
            types=["aquarium", "tourist_attraction", "point_of_interest"],
        ),
    ]

    assert _cluster_theme(places, requirement) == "family attractions"


def test_cluster_theme_treats_history_museums_as_heritage() -> None:
    """
    Verify that history museum types produce a heritage-led day theme.
    """
    requirement = _complete_requirement()
    places = [
        _recommendation(
            "history_1",
            "Hyde Park Barracks",
            "history_museum",
            35.700,
            139.740,
            types=[
                "history_museum",
                "historical_landmark",
                "museum",
                "point_of_interest",
            ],
        ),
    ]

    assert _cluster_theme(places, requirement) == "heritage and landmarks"


def test_cluster_theme_prefers_classic_for_symbolic_landmarks() -> None:
    """
    Verify that iconic mixed-type landmarks remain classic sightseeing.
    """
    requirement = _complete_requirement()
    places = [
        _recommendation(
            "opera_1",
            "Sydney Opera House",
            "opera_house",
            35.700,
            139.740,
            types=[
                "performing_arts_theater",
                "opera_house",
                "concert_hall",
                "auditorium",
                "tourist_attraction",
            ],
        ),
        _recommendation(
            "bridge_1",
            "Sydney Harbour Bridge",
            "bridge",
            35.705,
            139.745,
            types=["bridge", "tourist_attraction", "point_of_interest"],
        ),
    ]

    assert _cluster_theme(places, requirement) == "classic sightseeing"


def test_preferred_themes_reserve_classic_sightseeing_for_longer_trips() -> None:
    """
    Verify that longer trips reserve a first-visit sightseeing theme.
    """
    requirement = _complete_requirement().model_copy(
        update={"trip_length_days": 3, "interests": ["food", "museums", "nature"]}
    )

    assert _preferred_themes(requirement, 3) == [
        "classic sightseeing",
        "arts and museums",
        "parks and gardens",
        "food and markets",
    ]


def test_place_is_open_at_uses_regular_opening_hours() -> None:
    """
    Verify that date-aware opening hours are evaluated by local weekday.
    """
    place = _recommendation(
        "museum_1",
        "Monday Museum",
        "museum",
        35.700,
        139.740,
        regular_opening_hours=_opening_hours(1, 10, 0, 1, 17, 0),
    )

    assert _place_is_open_at(place, "2026-04-27", "10:00")
    assert not _place_is_open_at(place, "2026-04-27", "09:30")
    assert not _place_is_open_at(place, "2026-04-26", "10:00")


@pytest.mark.asyncio
async def test_build_days_themes_actual_selected_attractions() -> None:
    """
    Verify that day themes use selected attractions instead of the full cluster.
    """
    conversation_repository = InMemoryConversationRepository()
    itinerary_repository = InMemoryItineraryRepository()
    service = PlanningService(conversation_repository, itinerary_repository)
    requirement = _complete_requirement().model_copy(
        update={"trip_length_days": 2, "interests": ["museums", "nature"]}
    )
    primary_hotel = _recommendation(
        "hotel_1", "Central Hotel", "hotel", 35.700, 139.740
    )
    attractions = [
        _recommendation(
            "museum_1",
            "Modern Art Museum",
            "art_museum",
            35.701,
            139.741,
            types=["art_museum", "museum", "point_of_interest"],
        ),
        _recommendation(
            "museum_2",
            "City Gallery",
            "art_gallery",
            35.702,
            139.742,
            types=["art_gallery", "point_of_interest"],
        ),
        _recommendation(
            "park_1",
            "Harbour Park",
            "park",
            35.703,
            139.743,
            types=["park", "point_of_interest"],
        ),
        _recommendation(
            "park_2",
            "Botanical Garden",
            "botanical_garden",
            35.704,
            139.744,
            types=["botanical_garden", "point_of_interest"],
        ),
    ]
    restaurants = [
        _recommendation(
            "restaurant_1",
            "Cafe One",
            "restaurant",
            35.705,
            139.745,
            types=["restaurant", "point_of_interest"],
        ),
        _recommendation(
            "restaurant_2",
            "Bistro Two",
            "restaurant",
            35.706,
            139.746,
            types=["restaurant", "point_of_interest"],
        ),
    ]
    google_maps_client = GoogleMapsClient(
        places=FakePlacesClient(),
        routes=FakeRoutesClient(),
        geocoding=FakeGeocodingClient(),
        timezone=FakeTimeZoneClient(),
    )

    days = await service._build_days(
        requirement,
        primary_hotel,
        attractions,
        restaurants,
        google_maps_client,
    )

    assert [day.theme for day in days] == ["arts and museums", "parks and gardens"]


@pytest.mark.asyncio
async def test_build_days_avoids_closed_attractions_when_dates_exist() -> None:
    """
    Verify that dated plans prefer attractions open at scheduled visit times.
    """
    conversation_repository = InMemoryConversationRepository()
    itinerary_repository = InMemoryItineraryRepository()
    service = PlanningService(conversation_repository, itinerary_repository)
    requirement = _complete_requirement().model_copy(
        update={
            "trip_dates": "2026-04-27 to 2026-04-27",
            "trip_length_days": 1,
            "interests": ["museums"],
        }
    )
    primary_hotel = _recommendation(
        "hotel_1", "Central Hotel", "hotel", 35.700, 139.740
    )
    attractions = [
        _recommendation(
            "closed_museum",
            "Closed Museum",
            "museum",
            35.701,
            139.741,
            regular_opening_hours=_opening_hours(2, 10, 0, 2, 17, 0),
        ),
        _recommendation(
            "open_museum",
            "Open Museum",
            "museum",
            35.702,
            139.742,
            regular_opening_hours=_opening_hours(1, 10, 0, 1, 17, 0),
        ),
    ]
    restaurants = [
        _recommendation(
            "restaurant_1",
            "Open Lunch",
            "restaurant",
            35.703,
            139.743,
            types=["restaurant", "point_of_interest"],
            regular_opening_hours=_opening_hours(1, 12, 0, 1, 22, 0),
        ),
        _recommendation(
            "restaurant_2",
            "Dinner Only",
            "restaurant",
            35.704,
            139.744,
            types=["restaurant", "point_of_interest"],
            regular_opening_hours=_opening_hours(1, 17, 0, 1, 22, 0),
        ),
    ]
    google_maps_client = GoogleMapsClient(
        places=FakePlacesClient(),
        routes=FakeRoutesClient(),
        geocoding=FakeGeocodingClient(),
        timezone=FakeTimeZoneClient(),
    )

    days = await service._build_days(
        requirement,
        primary_hotel,
        attractions,
        restaurants,
        google_maps_client,
    )

    assert days[0].date == "2026-04-27"
    assert days[0].items[0].time == "10:00"
    assert days[0].items[0].place.name == "Open Museum"
    assert days[0].items[1].time == "12:15"
    assert days[0].items[1].place.name == "Open Lunch"


@pytest.mark.asyncio
async def test_planning_service_rejects_incomplete_requirements() -> None:
    """
    Verify that planning does not run with missing required slots.
    """
    conversation_repository = InMemoryConversationRepository()
    itinerary_repository = InMemoryItineraryRepository()
    conversation = Conversation()
    conversation_repository.save(conversation)
    service = PlanningService(conversation_repository, itinerary_repository)

    with pytest.raises(PlanningInputError):
        await service.generate_for_conversation(
            conversation.id,
            GoogleMapsClient(
                places=FakePlacesClient(),
                routes=FakeRoutesClient(),
                geocoding=FakeGeocodingClient(),
                timezone=FakeTimeZoneClient(),
            ),
        )


def _complete_requirement() -> TravelRequirement:
    """
    Create a complete travel requirement for tests.

    Returns:
        A complete travel requirement.
    """
    return TravelRequirement(
        destination="Tokyo",
        trip_length_days=1,
        travelers=Travelers(adults=2),
        budget_level="medium",
        travel_pace="relaxed",
        interests=["food", "museums"],
        hotel_area="Shinjuku",
        transportation_mode="transit",
        language="en",
    )


def _place(
    place_id: str,
    name: str,
    type_name: str,
    latitude: float,
    longitude: float,
    rating: float,
    user_rating_count: int,
) -> dict:
    """
    Build a fake Google Place payload.

    Args:
        place_id: The Google place ID.
        name: The display name.
        type_name: The primary type.
        latitude: The latitude.
        longitude: The longitude.
        rating: The rating.
        user_rating_count: The user rating count.

    Returns:
        A fake Google Place payload.
    """
    return {
        "id": place_id,
        "displayName": {"text": name, "languageCode": "en"},
        "formattedAddress": f"{name} address",
        "location": {"latitude": latitude, "longitude": longitude},
        "googleMapsUri": f"https://maps.google.com/?cid={place_id}",
        "rating": rating,
        "userRatingCount": user_rating_count,
        "priceLevel": "PRICE_LEVEL_MODERATE",
        "businessStatus": "OPERATIONAL",
        "types": [type_name, "point_of_interest"],
    }


def _recommendation(
    place_id: str,
    name: str,
    type_name: str,
    latitude: float,
    longitude: float,
    types: list[str] | None = None,
    regular_opening_hours: dict | None = None,
) -> PlaceRecommendation:
    """
    Build a place recommendation for pure planning tests.

    Args:
        place_id: The Google place ID.
        name: The display name.
        type_name: The primary type.
        latitude: The latitude.
        longitude: The longitude.
        types: The Google place types to attach.
        regular_opening_hours: The fake Google opening hours payload.

    Returns:
        A normalized place recommendation.
    """
    place_types = types or [type_name, "point_of_interest"]
    return PlaceRecommendation(
        place_id=place_id,
        name=name,
        category="attraction",
        location=Coordinates(latitude=latitude, longitude=longitude),
        rating=4.5,
        user_rating_count=200,
        types=place_types,
        regular_opening_hours=regular_opening_hours,
        score=70.0,
    )


def _opening_hours(
    open_day: int,
    open_hour: int,
    open_minute: int,
    close_day: int,
    close_hour: int,
    close_minute: int,
) -> dict:
    """
    Build a fake Google opening hours payload.

    Args:
        open_day: The Google opening weekday.
        open_hour: The opening hour.
        open_minute: The opening minute.
        close_day: The Google closing weekday.
        close_hour: The closing hour.
        close_minute: The closing minute.

    Returns:
        A fake regularOpeningHours payload.
    """
    return {
        "periods": [
            {
                "open": {
                    "day": open_day,
                    "hour": open_hour,
                    "minute": open_minute,
                },
                "close": {
                    "day": close_day,
                    "hour": close_hour,
                    "minute": close_minute,
                },
            }
        ]
    }
