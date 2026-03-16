"""
Google Places API integration for live attraction, restaurant, and hotel discovery.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, overload

from smarttour.config import get_settings
from smarttour.models import Attraction, Hotel, Restaurant

logger = logging.getLogger(__name__)

PLACE_DETAILS_FIELDS = [
    "name",
    "geometry",
    "rating",
    "opening_hours",
    "price_level",
    "type",
    "editorial_summary",
    "formatted_address",
]

ATTRACTION_SEARCH_PLANS: dict[str, list[tuple[str, str | None]]] = {
    "museum": [("museum", "museum")],
    "park": [("park", "park")],
    "food": [("food market", "tourist_attraction")],
    "shopping": [("shopping district", "tourist_attraction")],
    "landmark": [("landmark", "tourist_attraction")],
    "wildlife": [("zoo", "zoo"), ("aquarium", "aquarium")],
    "beach": [("beach", "tourist_attraction")],
    "nightlife": [("nightlife", "tourist_attraction")],
    "culture": [("cultural attraction", "tourist_attraction")],
    "sport": [("stadium", "stadium")],
}

DIVERSITY_BASE_PLANS: list[tuple[str, str | None]] = [
    ("popular park", "park"),
    ("cultural attraction", "tourist_attraction"),
    ("popular market", "tourist_attraction"),
    ("must see landmarks", "tourist_attraction"),
]

ATTRACTION_TYPE_PRIORITY = [
    ("museum", "museum"),
    ("art_gallery", "museum"),
    ("library", "culture"),
    ("cultural_landmark", "culture"),
    ("park", "park"),
    ("zoo", "wildlife"),
    ("aquarium", "wildlife"),
    ("beach", "beach"),
    ("shopping_mall", "shopping"),
    ("market", "shopping"),
    ("stadium", "sport"),
    ("amusement_park", "landmark"),
    ("night_club", "nightlife"),
    ("bar", "nightlife"),
    ("tourist_attraction", "landmark"),
    ("restaurant", "food"),
    ("cafe", "food"),
]

CUISINE_TYPE_PRIORITY = [
    ("cafe", "cafe"),
    ("bakery", "bakery"),
    ("meal_takeaway", "casual"),
    ("bar", "bar"),
    ("restaurant", "restaurant"),
]

VISIT_DURATION_BY_TYPE = {
    "museum": 90,
    "tourist_attraction": 90,
    "park": 120,
    "restaurant": 60,
    "cafe": 60,
    "zoo": 180,
    "aquarium": 180,
}

PRICE_LEVEL_TO_ATTRACTION_COST = {
    0: 0.0,
    1: 10.0,
    2: 25.0,
    3: 50.0,
    4: 80.0,
}

PRICE_LEVEL_TO_RESTAURANT_COST = {
    0: 10.0,
    1: 20.0,
    2: 35.0,
    3: 55.0,
    4: 80.0,
}

PRICE_LEVEL_TO_HOTEL_COST = {
    0: 150.0,
    1: 80.0,
    2: 150.0,
    3: 250.0,
    4: 400.0,
}

MEAL_SEARCH_KEYWORDS = {
    "breakfast": "breakfast cafe",
    "lunch": "lunch restaurant",
    "dinner": "dinner restaurant",
}
ALL_MEAL_TYPES = ["breakfast", "lunch", "dinner"]
ATTRACTION_NAME_SEARCH_TYPES = {
    "tourist_attraction",
    "museum",
    "park",
    "zoo",
    "aquarium",
    "stadium",
}
RESTAURANT_NAME_SEARCH_TYPES = {"restaurant", "cafe"}

HOTEL_AMENITY_TYPES = {
    "spa": "spa",
    "gym": "gym",
    "pool": "pool",
}


class PlacesClient:
    """
    Retrieve live place data from Google Places when an API key is configured.
    """

    def __init__(self) -> None:
        """
        Initialize the Places client if Google Maps credentials are available.
        """

        settings = get_settings()
        self.api_key = settings.google_maps_api_key
        self._client = None
        if self.api_key:
            try:
                import googlemaps

                self._client = googlemaps.Client(key=self.api_key)
            except Exception:
                logger.exception("Failed to initialize Google Places client")

    @property
    def enabled(self) -> bool:
        """
        Return whether live Places lookups are available.

        Returns:
            ``True`` when a Google client is initialized.
        """

        return self._client is not None

    def search_attractions(
        self,
        destination: str,
        interests: list[str],
        limit: int = 12,
    ) -> list[Attraction]:
        """
        Search live attractions for a destination.

        Args:
            destination: Destination name.
            interests: Desired interest categories.
            limit: Maximum number of results.

        Returns:
            Live attractions from Google Places.
        """

        if not self.enabled:
            return []

        logger.info(
            "Places search: type=%s destination='%s' limit=%d",
            "attraction",
            destination,
            limit,
        )
        seen_place_ids: set[str] = set()
        attractions: list[Attraction] = []
        search_plans = self._build_attraction_search_plans(interests)
        assert self._client is not None

        for query_term, place_type in search_plans:
            query = f"{query_term} in {destination}"
            try:
                response = self._client.places(
                    query=query,
                    type=place_type,
                )
            except Exception:
                logger.exception(
                    "Places attraction search failed for query '%s'", query
                )
                continue
            raw_results = response.get("results", [])
            logger.debug(
                "Places API response: %d raw results for '%s'",
                len(raw_results) if isinstance(raw_results, list) else 0,
                query,
            )

            for raw_place in raw_results:
                if not isinstance(raw_place, dict):
                    continue
                place_id = raw_place.get("place_id")
                if not isinstance(place_id, str) or place_id in seen_place_ids:
                    continue
                place_payload = raw_place
                if not self._has_weekday_text(raw_place):
                    detail_payload = self._fetch_place_details(place_id)
                    if detail_payload:
                        place_payload = self._merge_place_payloads(
                            raw_place,
                            detail_payload,
                        )
                attractions.append(
                    self._build_attraction(
                        destination=destination,
                        raw_place=place_payload,
                    )
                )
                seen_place_ids.add(place_id)
                if len(attractions) >= limit:
                    return attractions
        return attractions

    def search_restaurants(
        self,
        destination: str,
        meal_type: str,
        near_lat: float,
        near_lon: float,
        limit: int = 6,
    ) -> list[Restaurant]:
        """
        Search live restaurants near a route point.

        Args:
            destination: Destination name.
            meal_type: Meal slot name.
            near_lat: Reference latitude.
            near_lon: Reference longitude.
            limit: Maximum number of results.

        Returns:
            Live restaurant candidates from Google Places.
        """

        if not self.enabled:
            return []

        logger.info(
            "Places search: type=%s destination='%s' limit=%d",
            "restaurant",
            destination,
            limit,
        )
        assert self._client is not None
        keyword = MEAL_SEARCH_KEYWORDS.get(meal_type, f"{meal_type} restaurant")
        try:
            response = self._client.places_nearby(
                location=(near_lat, near_lon),
                radius=2000,
                keyword=keyword,
                type="restaurant",
                rank_by="prominence",
            )
        except Exception:
            logger.exception(
                "Places restaurant search failed for %s near (%s, %s)",
                meal_type,
                near_lat,
                near_lon,
            )
            return []
        raw_results = response.get("results", [])
        logger.debug(
            "Places API response: %d raw results for '%s'",
            len(raw_results) if isinstance(raw_results, list) else 0,
            keyword,
        )

        restaurants: list[Restaurant] = []
        for raw_place in raw_results:
            if not isinstance(raw_place, dict):
                continue
            place_id = raw_place.get("place_id")
            if not isinstance(place_id, str):
                continue
            place_payload = raw_place
            if not self._has_weekday_text(raw_place):
                detail_payload = self._fetch_place_details(place_id)
                if detail_payload:
                    place_payload = self._merge_place_payloads(
                        raw_place, detail_payload
                    )
            restaurants.append(
                self._build_restaurant(
                    destination=destination,
                    raw_place=place_payload,
                    meal_type=meal_type,
                )
            )
            if len(restaurants) >= limit:
                break
        return restaurants

    def search_hotels(
        self,
        destination: str,
        limit: int = 5,
    ) -> list[Hotel]:
        """
        Search live hotels for a destination.

        Args:
            destination: Destination name.
            limit: Maximum number of hotel results.

        Returns:
            Live hotel candidates from Google Places.
        """

        if not self.enabled:
            return []

        logger.info(
            "Places search: type=%s destination='%s' limit=%d",
            "lodging",
            destination,
            limit,
        )
        assert self._client is not None
        try:
            response = self._client.places(
                query=f"{destination} hotel",
                type="lodging",
            )
        except Exception:
            logger.exception(
                "Places hotel search failed for destination '%s'", destination
            )
            return []
        raw_results = response.get("results", [])
        logger.debug(
            "Places API response: %d raw results for '%s'",
            len(raw_results) if isinstance(raw_results, list) else 0,
            f"{destination} hotel",
        )

        seen_place_ids: set[str] = set()
        hotels: list[Hotel] = []
        for raw_place in raw_results:
            if not isinstance(raw_place, dict):
                continue
            place_id = raw_place.get("place_id")
            if not isinstance(place_id, str) or place_id in seen_place_ids:
                continue
            hotels.append(
                self._build_hotel(
                    destination=destination,
                    raw_place=raw_place,
                )
            )
            seen_place_ids.add(place_id)
            if len(hotels) >= limit:
                break
        return hotels

    @overload
    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: Literal[
            "tourist_attraction", "museum", "park", "zoo", "aquarium", "stadium"
        ],
        limit: int = 3,
    ) -> list[Attraction]: ...

    @overload
    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: Literal["lodging"],
        limit: int = 3,
    ) -> list[Hotel]: ...

    @overload
    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: Literal["restaurant", "cafe"],
        limit: int = 3,
    ) -> list[Restaurant]: ...

    @overload
    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: str = "",
        limit: int = 3,
    ) -> list[Attraction] | list[Hotel] | list[Restaurant]: ...

    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: str = "",
        limit: int = 3,
    ) -> list[Attraction] | list[Hotel] | list[Restaurant]:
        """
        Search for a specific place name within a destination.

        Args:
            destination: Destination name.
            name: Place name hint from Xiaohongshu discovery.
            place_type: Optional Google Places type.
            limit: Maximum number of results.

        Returns:
            Structured place results for the requested place type.
        """

        results: list[Attraction] | list[Hotel] | list[Restaurant]
        if place_type == "lodging":
            results = self._search_hotels_by_name(destination, name, limit)
        elif place_type in RESTAURANT_NAME_SEARCH_TYPES:
            results = self._search_restaurants_by_name(
                destination,
                name,
                place_type,
                limit,
            )
        else:
            results = self._search_attractions_by_name(
                destination,
                name,
                place_type if place_type in ATTRACTION_NAME_SEARCH_TYPES else "",
                limit,
            )
        logger.info(
            "Places search_by_name: name='%s' type=%s returned %d results",
            name,
            place_type or "any",
            len(results),
        )
        return results

    def _search_attractions_by_name(
        self,
        destination: str,
        name: str,
        place_type: str,
        limit: int,
    ) -> list[Attraction]:
        """
        Search for targeted attractions by name.

        Args:
            destination: Destination name.
            name: Place name hint.
            place_type: Optional Google Places type.
            limit: Maximum number of results.

        Returns:
            Targeted attraction matches.
        """

        payloads = self._search_place_payloads(destination, name, place_type, limit)
        return [
            self._build_attraction(destination=destination, raw_place=payload)
            for payload in payloads
        ]

    def _search_hotels_by_name(
        self,
        destination: str,
        name: str,
        limit: int,
    ) -> list[Hotel]:
        """
        Search for targeted hotels by name.

        Args:
            destination: Destination name.
            name: Hotel name hint.
            limit: Maximum number of results.

        Returns:
            Targeted hotel matches.
        """

        payloads = self._search_place_payloads(destination, name, "lodging", limit)
        return [
            self._build_hotel(destination=destination, raw_place=payload)
            for payload in payloads
        ]

    def _search_restaurants_by_name(
        self,
        destination: str,
        name: str,
        place_type: str,
        limit: int,
    ) -> list[Restaurant]:
        """
        Search for targeted restaurants by name.

        Args:
            destination: Destination name.
            name: Restaurant name hint.
            place_type: Google Places type.
            limit: Maximum number of results.

        Returns:
            Targeted restaurant matches.
        """

        payloads = self._search_place_payloads(destination, name, place_type, limit)
        restaurants: list[Restaurant] = []
        for payload in payloads:
            restaurant = self._build_restaurant(
                destination=destination,
                raw_place=payload,
                meal_type="lunch",
            )
            restaurants.append(
                restaurant.model_copy(
                    update={
                        "meal_types": list(ALL_MEAL_TYPES),
                        "tags": list(
                            dict.fromkeys([*ALL_MEAL_TYPES, *restaurant.tags])
                        ),
                    }
                )
            )
        return restaurants

    def _search_place_payloads(
        self,
        destination: str,
        name: str,
        place_type: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Run a targeted Places text search and normalize the payloads.

        Args:
            destination: Destination name.
            name: Place name hint.
            place_type: Optional Google Places type.
            limit: Maximum number of results.

        Returns:
            Deduplicated Places payloads.
        """

        if not self.enabled:
            return []

        assert self._client is not None
        query = f"{destination} {name}".strip()
        try:
            if place_type:
                response = self._client.places(query=query, type=place_type)
            else:
                response = self._client.places(query=query)
        except Exception:
            logger.exception(
                "Places targeted search failed for query '%s' and type '%s'",
                query,
                place_type,
            )
            return []
        raw_results = response.get("results", [])
        logger.debug(
            "Places API response: %d raw results for '%s'",
            len(raw_results) if isinstance(raw_results, list) else 0,
            query,
        )

        payloads: list[dict[str, Any]] = []
        seen_place_ids: set[str] = set()
        for raw_place in raw_results:
            if not isinstance(raw_place, dict):
                continue
            place_id = raw_place.get("place_id")
            if not isinstance(place_id, str) or place_id in seen_place_ids:
                continue
            place_payload = raw_place
            if not self._has_weekday_text(raw_place):
                detail_payload = self._fetch_place_details(place_id)
                if detail_payload:
                    place_payload = self._merge_place_payloads(
                        raw_place,
                        detail_payload,
                    )
            payloads.append(place_payload)
            seen_place_ids.add(place_id)
            if len(payloads) >= limit:
                break
        return payloads

    def _fetch_place_details(self, place_id: str) -> dict[str, Any]:
        """
        Fetch additional details for a place when search results are incomplete.

        Args:
            place_id: Google place identifier.

        Returns:
            Place details payload or an empty dict on failure.
        """

        if not self.enabled:
            return {}

        assert self._client is not None
        try:
            response = self._client.place(
                place_id=place_id, fields=PLACE_DETAILS_FIELDS
            )
        except Exception:
            logger.exception("Places details lookup failed for '%s'", place_id)
            return {}

        result = response.get("result", {})
        if isinstance(result, dict):
            return result
        return {}

    def _build_attraction(
        self,
        destination: str,
        raw_place: dict[str, Any],
    ) -> Attraction:
        """
        Convert a Places payload into an attraction model.

        Args:
            destination: Destination name.
            raw_place: Places result payload.

        Returns:
            Normalized attraction model.
        """

        place_types = self._extract_types(raw_place)
        category = self._map_attraction_category(place_types)
        return Attraction(
            id=self._prefixed_place_id(raw_place),
            name=self._string_value(raw_place.get("name"), "Unknown Place"),
            destination=destination,
            category=category,
            latitude=self._latitude(raw_place),
            longitude=self._longitude(raw_place),
            description=self._place_description(raw_place),
            opening_hours=self._extract_opening_hours(raw_place),
            visit_duration=self._estimate_visit_duration(category, place_types),
            cost=self._map_price_level(
                raw_place.get("price_level"), for_restaurant=False
            ),
            rating=self._float_value(raw_place.get("rating")),
            accessibility=None,
            tags=place_types,
            image_url=self._photo_url(raw_place),
            source="google_places",
        )

    def _build_restaurant(
        self,
        destination: str,
        raw_place: dict[str, Any],
        meal_type: str,
    ) -> Restaurant:
        """
        Convert a Places payload into a restaurant model.

        Args:
            destination: Destination name.
            raw_place: Places result payload.
            meal_type: Meal slot used for the query.

        Returns:
            Normalized restaurant model.
        """

        place_types = self._extract_types(raw_place)
        cuisine = self._map_cuisine(place_types)
        tags = list(dict.fromkeys([meal_type, *place_types]))
        return Restaurant(
            id=self._prefixed_place_id(raw_place),
            name=self._string_value(raw_place.get("name"), "Unknown Restaurant"),
            destination=destination,
            cuisine=cuisine,
            meal_types=[meal_type],
            latitude=self._latitude(raw_place),
            longitude=self._longitude(raw_place),
            description=self._place_description(raw_place),
            opening_hours=self._extract_opening_hours(raw_place),
            average_cost=self._map_price_level(
                raw_place.get("price_level"),
                for_restaurant=True,
            ),
            visit_duration=self._estimate_visit_duration("restaurant", place_types),
            rating=self._float_value(raw_place.get("rating")),
            tags=tags,
            image_url=self._photo_url(raw_place),
            source="google_places",
        )

    def _build_hotel(
        self,
        destination: str,
        raw_place: dict[str, Any],
    ) -> Hotel:
        """
        Convert a Places payload into a hotel model.

        Args:
            destination: Destination name.
            raw_place: Places result payload.

        Returns:
            Normalized hotel model.
        """

        place_types = self._extract_types(raw_place)
        return Hotel(
            id=self._prefixed_place_id(raw_place),
            name=self._string_value(raw_place.get("name"), "Unknown Hotel"),
            destination=destination,
            latitude=self._latitude(raw_place),
            longitude=self._longitude(raw_place),
            description=self._place_description(raw_place),
            price_per_night=self._map_hotel_price(raw_place.get("price_level")),
            star_rating=self._hotel_star_rating(raw_place.get("rating")),
            category="hotel",
            amenities=self._hotel_amenities(place_types),
            booking_url=None,
            source="google_places",
        )

    def _build_attraction_search_plans(
        self,
        interests: list[str],
    ) -> list[tuple[str, str | None]]:
        """
        Build a compact set of search plans based on user interests.

        Args:
            interests: Interest categories.

        Returns:
            Query terms paired with optional Places types.
        """

        plans: list[tuple[str, str | None]] = []
        for interest in interests:
            plans.extend(ATTRACTION_SEARCH_PLANS.get(interest, []))
        if not plans:
            plans.append(("tourist attractions", "tourist_attraction"))
        for base_plan in DIVERSITY_BASE_PLANS:
            if base_plan not in plans:
                plans.append(base_plan)
        return list(dict.fromkeys(plans))

    def _prefixed_place_id(self, raw_place: dict[str, Any]) -> str:
        """
        Prefix Google place IDs to avoid collisions with seed IDs.

        Args:
            raw_place: Places result payload.

        Returns:
            Prefixed place identifier.
        """

        place_id = self._string_value(raw_place.get("place_id"), "unknown-place")
        return f"gp-{place_id}"

    def _latitude(self, raw_place: dict[str, Any]) -> float:
        """
        Extract latitude from a Places payload.

        Args:
            raw_place: Places result payload.

        Returns:
            Latitude value.
        """

        geometry = raw_place.get("geometry")
        if isinstance(geometry, dict):
            location = geometry.get("location")
            if isinstance(location, dict):
                return self._float_value(location.get("lat"))
        return 0.0

    def _longitude(self, raw_place: dict[str, Any]) -> float:
        """
        Extract longitude from a Places payload.

        Args:
            raw_place: Places result payload.

        Returns:
            Longitude value.
        """

        geometry = raw_place.get("geometry")
        if isinstance(geometry, dict):
            location = geometry.get("location")
            if isinstance(location, dict):
                return self._float_value(location.get("lng"))
        return 0.0

    def _extract_types(self, raw_place: dict[str, Any]) -> list[str]:
        """
        Extract the raw Google place types list.

        Args:
            raw_place: Places result payload.

        Returns:
            Normalized types list.
        """

        raw_types = raw_place.get("types", [])
        if isinstance(raw_types, str):
            return [raw_types]
        if not isinstance(raw_types, list):
            raw_type = raw_place.get("type")
            if isinstance(raw_type, str):
                return [raw_type]
            if isinstance(raw_type, list):
                return [item for item in raw_type if isinstance(item, str)]
            return []
        return [item for item in raw_types if isinstance(item, str)]

    def _map_attraction_category(self, place_types: list[str]) -> str:
        """
        Map Google place types onto the project's attraction categories.

        Args:
            place_types: Google place types.

        Returns:
            Internal attraction category.
        """

        lowered = set(place_types)
        for google_type, mapped_category in ATTRACTION_TYPE_PRIORITY:
            if google_type in lowered:
                return mapped_category
        return "landmark"

    def _map_cuisine(self, place_types: list[str]) -> str:
        """
        Map Google place types onto a simple cuisine label.

        Args:
            place_types: Google place types.

        Returns:
            Cuisine label.
        """

        lowered = set(place_types)
        for google_type, mapped_cuisine in CUISINE_TYPE_PRIORITY:
            if google_type in lowered:
                return mapped_cuisine
        return "restaurant"

    def _estimate_visit_duration(
        self,
        category: str,
        place_types: list[str],
    ) -> int:
        """
        Estimate visit duration from category and raw place types.

        Args:
            category: Internal category.
            place_types: Google place types.

        Returns:
            Estimated visit duration in minutes.
        """

        for place_type in place_types:
            if place_type in VISIT_DURATION_BY_TYPE:
                return VISIT_DURATION_BY_TYPE[place_type]
        if category in VISIT_DURATION_BY_TYPE:
            return VISIT_DURATION_BY_TYPE[category]
        return 60

    def _hotel_star_rating(self, rating: Any) -> int:
        """
        Convert a Google rating into a coarse hotel star rating.

        Args:
            rating: Raw Google rating value.

        Returns:
            A star rating clamped between ``1`` and ``5``.
        """

        normalized_rating = round(self._float_value(rating))
        return max(1, min(5, normalized_rating))

    def _map_hotel_price(self, price_level: Any) -> float:
        """
        Map Google price levels to nightly hotel price estimates.

        Args:
            price_level: Raw Places price level value.

        Returns:
            Approximate nightly hotel price.
        """

        try:
            normalized_level = int(price_level)
        except (TypeError, ValueError):
            normalized_level = 0
        return PRICE_LEVEL_TO_HOTEL_COST.get(
            normalized_level,
            PRICE_LEVEL_TO_HOTEL_COST[0],
        )

    def _hotel_amenities(self, place_types: list[str]) -> list[str]:
        """
        Filter hotel-relevant amenities from Google place types.

        Args:
            place_types: Google place types.

        Returns:
            A normalized amenity list.
        """

        amenities: list[str] = []
        for place_type in place_types:
            amenity = HOTEL_AMENITY_TYPES.get(place_type)
            if amenity is not None and amenity not in amenities:
                amenities.append(amenity)
        return amenities

    def _map_price_level(
        self,
        price_level: Any,
        *,
        for_restaurant: bool,
    ) -> float:
        """
        Map Google price levels to application costs.

        Args:
            price_level: Raw Places price level value.
            for_restaurant: Whether to use restaurant pricing.

        Returns:
            Approximate cost in local currency units.
        """

        try:
            normalized_level = int(price_level)
        except (TypeError, ValueError):
            normalized_level = 0
        mapping = (
            PRICE_LEVEL_TO_RESTAURANT_COST
            if for_restaurant
            else PRICE_LEVEL_TO_ATTRACTION_COST
        )
        return mapping.get(normalized_level, mapping[0])

    def _extract_opening_hours(self, raw_place: dict[str, Any]) -> dict[str, str]:
        """
        Normalize opening hours from a Places payload.

        Args:
            raw_place: Places result payload.

        Returns:
            Weekly opening hours map.
        """

        for key in ("opening_hours", "current_opening_hours"):
            opening_hours = raw_place.get(key)
            if not isinstance(opening_hours, dict):
                continue
            weekday_text = opening_hours.get("weekday_text")
            if isinstance(weekday_text, list):
                return self._weekday_text_to_map(weekday_text)
        return {}

    def _weekday_text_to_map(self, weekday_text: list[Any]) -> dict[str, str]:
        """
        Convert Google weekday strings into a short-key dictionary.

        Args:
            weekday_text: List of weekday strings.

        Returns:
            Opening hours keyed by weekday abbreviation.
        """

        day_mapping = {
            "Monday": "mon",
            "Tuesday": "tue",
            "Wednesday": "wed",
            "Thursday": "thu",
            "Friday": "fri",
            "Saturday": "sat",
            "Sunday": "sun",
        }
        hours_map: dict[str, str] = {}
        for item in weekday_text:
            if not isinstance(item, str) or ":" not in item:
                continue
            day_name, hours_value = item.split(":", 1)
            key = day_mapping.get(day_name.strip())
            if key is not None:
                hours_map[key] = self._normalize_hours_text(hours_value.strip())
        return hours_map

    def _normalize_hours_text(self, hours_value: str) -> str:
        """
        Normalize Google opening-hours text into planner-friendly values.

        Args:
            hours_value: Human-readable hours text.

        Returns:
            Normalized hours string.
        """

        if hours_value.lower() == "open 24 hours":
            return "00:00-23:59"
        return hours_value

    def _place_description(self, raw_place: dict[str, Any]) -> str:
        """
        Build a concise description from available Places fields.

        Args:
            raw_place: Places result payload.

        Returns:
            Description text.
        """

        editorial_summary = raw_place.get("editorial_summary")
        if isinstance(editorial_summary, dict):
            overview = editorial_summary.get("overview")
            if isinstance(overview, str) and overview.strip():
                return overview.strip()
        for key in ("formatted_address", "vicinity"):
            value = raw_place.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "Live result from Google Places."

    def _photo_url(self, raw_place: dict[str, Any]) -> str | None:
        """
        Build a photo URL when Google returns a photo reference.

        Args:
            raw_place: Places result payload.

        Returns:
            A photo URL or ``None``.
        """

        if not self.api_key:
            return None
        photos = raw_place.get("photos")
        if not isinstance(photos, list) or not photos:
            return None
        first_photo = photos[0]
        if not isinstance(first_photo, dict):
            return None
        photo_reference = first_photo.get("photo_reference")
        if not isinstance(photo_reference, str) or not photo_reference:
            return None
        return (
            "https://maps.googleapis.com/maps/api/place/photo"
            f"?maxwidth=800&photo_reference={photo_reference}&key={self.api_key}"
        )

    def _has_weekday_text(self, raw_place: dict[str, Any]) -> bool:
        """
        Return whether a Places payload already includes detailed weekday text.

        Args:
            raw_place: Places result payload.

        Returns:
            ``True`` when weekday text is present.
        """

        for key in ("opening_hours", "current_opening_hours"):
            opening_hours = raw_place.get(key)
            if not isinstance(opening_hours, dict):
                continue
            weekday_text = opening_hours.get("weekday_text")
            if isinstance(weekday_text, list) and bool(weekday_text):
                return True
        return False

    def _merge_place_payloads(
        self,
        base_payload: dict[str, Any],
        detail_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge search and detail payloads, preferring detail values when present.

        Args:
            base_payload: Search payload.
            detail_payload: Detail payload.

        Returns:
            Combined payload.
        """

        merged = dict(base_payload)
        for key, value in detail_payload.items():
            if value in (None, "", [], {}):
                continue
            merged[key] = value
        return merged

    def _string_value(self, value: Any, default: str) -> str:
        """
        Coerce a raw Places field to a string.

        Args:
            value: Raw value.
            default: Fallback string.

        Returns:
            A normalized string.
        """

        if isinstance(value, str) and value.strip():
            return value.strip()
        return default

    def _float_value(self, value: Any) -> float:
        """
        Coerce a raw Places field to a float.

        Args:
            value: Raw value.

        Returns:
            Float representation or ``0.0``.
        """

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
