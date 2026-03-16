"""
Database-backed retrieval with Google Places data and Xiaohongshu popularity boosts.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, overload

from sqlmodel import Session, select

from smarttour.config import get_settings
from smarttour.models import (
    AccommodationRecord,
    Attraction,
    AttractionRecord,
    Hotel,
    Restaurant,
    RestaurantRecord,
    XhsDiscovery,
)
from smarttour.services.places_client import PlacesClient
from smarttour.services.xiaohongshu_client import XiaohongshuClient

logger = logging.getLogger(__name__)


class PlacesLookup(Protocol):
    """
    Protocol for live Places lookups used by the retrieval service.
    """

    enabled: bool

    def search_attractions(
        self,
        destination: str,
        interests: list[str],
        limit: int = 12,
    ) -> list[Attraction]:
        """
        Return live attractions for a destination.
        """

        ...

    def search_restaurants(
        self,
        destination: str,
        meal_type: str,
        near_lat: float,
        near_lon: float,
        limit: int = 6,
    ) -> list[Restaurant]:
        """
        Return live restaurants for a meal slot.
        """

        ...

    def search_hotels(
        self,
        destination: str,
        limit: int = 5,
    ) -> list[Hotel]:
        """
        Return live hotels for a destination.
        """

        ...

    @overload
    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: Literal[
            "tourist_attraction", "museum", "park", "zoo", "aquarium", "stadium"
        ],
        limit: int = 3,
    ) -> list[Attraction]:
        """
        Return targeted attraction matches.
        """

        ...

    @overload
    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: Literal["lodging"],
        limit: int = 3,
    ) -> list[Hotel]:
        """
        Return targeted hotel matches.
        """

        ...

    @overload
    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: Literal["restaurant", "cafe"],
        limit: int = 3,
    ) -> list[Restaurant]:
        """
        Return targeted restaurant matches.
        """

        ...

    def search_by_name(
        self,
        destination: str,
        name: str,
        place_type: str = "",
        limit: int = 3,
    ) -> list[Attraction] | list[Hotel] | list[Restaurant]:
        """
        Return targeted place matches.
        """

        ...


class PopularityLookup(Protocol):
    """
    Protocol for destination popularity lookups used in ranking.
    """

    enabled: bool

    def search_popularity(
        self,
        session: Session,
        destination: str,
    ) -> dict[str, float]:
        """
        Return popularity scores keyed by lowercase Xiaohongshu note titles.
        """

        ...

    def discover_destination(
        self,
        session: Session,
        destination: str,
    ) -> XhsDiscovery:
        """
        Return structured Xiaohongshu discovery data for a destination.
        """

        ...

    def search_place_notes(
        self,
        place_name: str,
        limit: int = 5,
    ) -> list[str]:
        """
        Return XHS note titles for a specific place.
        """

        ...


class DataRetrievalService:
    """
    Retrieve attractions and restaurants from cache, Google Places, or seed data.
    """

    def __init__(
        self,
        places_client: PlacesLookup | None = None,
        xiaohongshu_client: PopularityLookup | None = None,
    ) -> None:
        """
        Initialize the retrieval service.

        Args:
            places_client: Optional Places client override.
            xiaohongshu_client: Optional Xiaohongshu client override.
        """

        self.settings = get_settings()
        self.places_client = places_client or PlacesClient()
        self.xiaohongshu_client = xiaohongshu_client or XiaohongshuClient()

    def search_attractions(
        self,
        session: Session,
        destination: str,
        categories: list[str] | None = None,
        max_cost: float | None = None,
        limit: int = 12,
    ) -> list[Attraction]:
        """
        Search attractions using cache-first retrieval with live fallback.

        Args:
            session: Active database session.
            destination: Destination name.
            categories: Optional desired categories or tags.
            max_cost: Optional cost ceiling per attraction.
            limit: Maximum number of results.

        Returns:
            Matching attractions ordered by quality.
        """

        logger.info(
            "search_attractions: destination='%s' categories=%s max_cost=%s limit=%d",
            destination,
            categories,
            max_cost,
            limit,
        )
        popularity = self.xiaohongshu_client.search_popularity(session, destination)
        fresh_cached = self._load_cached_attractions(
            session=session,
            destination=destination,
            fresh_only=True,
        )
        filtered_fresh = self._filter_attractions(
            attractions=fresh_cached,
            categories=categories,
            max_cost=max_cost,
            limit=limit,
            popularity=popularity,
        )
        if filtered_fresh:
            self._log_search_results(
                "search_attractions", "fresh_cache", len(filtered_fresh)
            )
            return filtered_fresh

        live_results = self.places_client.search_attractions(
            destination=destination,
            interests=categories or [],
            limit=limit,
        )
        if live_results:
            self._cache_attractions(session, live_results)
            filtered_live = self._filter_attractions(
                attractions=live_results,
                categories=categories,
                max_cost=max_cost,
                limit=limit,
                popularity=popularity,
            )
            if filtered_live:
                self._log_search_results(
                    "search_attractions", "live_api", len(filtered_live)
                )
                return filtered_live

        stale_cached = self._load_cached_attractions(
            session=session,
            destination=destination,
            fresh_only=False,
        )
        filtered_stale = self._filter_attractions(
            attractions=stale_cached,
            categories=categories,
            max_cost=max_cost,
            limit=limit,
            popularity=popularity,
        )
        if filtered_stale:
            self._log_search_results(
                "search_attractions", "stale_cache", len(filtered_stale)
            )
            return filtered_stale

        seed_records = self._load_seed_attractions(session, destination)
        filtered_seed = self._filter_attractions(
            attractions=seed_records,
            categories=categories,
            max_cost=max_cost,
            limit=limit,
            popularity=popularity,
        )
        self._log_search_results("search_attractions", "seed", len(filtered_seed))
        return filtered_seed

    def get_attraction(self, session: Session, attraction_id: str) -> Attraction | None:
        """
        Retrieve a single attraction by identifier.

        Args:
            session: Active database session.
            attraction_id: Attraction primary key.

        Returns:
            The matching attraction or `None`.
        """

        record = session.get(AttractionRecord, attraction_id)
        if record is None:
            return None
        return record.to_model()

    def search_restaurants(
        self,
        session: Session,
        destination: str,
        meal_type: str,
        near_lat: float | None = None,
        near_lon: float | None = None,
        max_cost: float | None = None,
        limit: int = 3,
    ) -> list[Restaurant]:
        """
        Search restaurants using cache-first retrieval with live fallback.

        Args:
            session: Active database session.
            destination: Destination name.
            meal_type: Desired meal slot, such as ``breakfast`` or ``dinner``.
            near_lat: Optional latitude used for proximity ranking.
            near_lon: Optional longitude used for proximity ranking.
            max_cost: Optional per-person budget ceiling.
            limit: Maximum number of results.

        Returns:
            Matching restaurants ranked by suitability.
        """

        logger.info(
            "search_restaurants: destination='%s' meal_type=%s max_cost=%s limit=%d",
            destination,
            meal_type,
            max_cost,
            limit,
        )
        popularity = self.xiaohongshu_client.search_popularity(session, destination)
        fresh_cached = self._load_cached_restaurants(
            session=session,
            destination=destination,
            fresh_only=True,
        )
        filtered_fresh = self._filter_restaurants(
            restaurants=fresh_cached,
            meal_type=meal_type,
            near_lat=near_lat,
            near_lon=near_lon,
            max_cost=max_cost,
            limit=limit,
            popularity=popularity,
        )
        if filtered_fresh:
            self._log_search_results(
                "search_restaurants", "fresh_cache", len(filtered_fresh)
            )
            return filtered_fresh

        if near_lat is not None and near_lon is not None:
            live_results = self.places_client.search_restaurants(
                destination=destination,
                meal_type=meal_type,
                near_lat=near_lat,
                near_lon=near_lon,
                limit=max(limit, 6),
            )
            if live_results:
                self._cache_restaurants(session, live_results)
                filtered_live = self._filter_restaurants(
                    restaurants=live_results,
                    meal_type=meal_type,
                    near_lat=near_lat,
                    near_lon=near_lon,
                    max_cost=max_cost,
                    limit=limit,
                    popularity=popularity,
                )
                if filtered_live:
                    self._log_search_results(
                        "search_restaurants", "live_api", len(filtered_live)
                    )
                    return filtered_live

        stale_cached = self._load_cached_restaurants(
            session=session,
            destination=destination,
            fresh_only=False,
        )
        filtered_stale = self._filter_restaurants(
            restaurants=stale_cached,
            meal_type=meal_type,
            near_lat=near_lat,
            near_lon=near_lon,
            max_cost=max_cost,
            limit=limit,
            popularity=popularity,
        )
        if filtered_stale:
            self._log_search_results(
                "search_restaurants", "stale_cache", len(filtered_stale)
            )
            return filtered_stale

        seed_records = self._load_seed_restaurants(session, destination)
        filtered_seed = self._filter_restaurants(
            restaurants=seed_records,
            meal_type=meal_type,
            near_lat=near_lat,
            near_lon=near_lon,
            max_cost=max_cost,
            limit=limit,
            popularity=popularity,
        )
        self._log_search_results("search_restaurants", "seed", len(filtered_seed))
        return filtered_seed

    def search_hotels(
        self,
        session: Session,
        destination: str,
        nightly_budget: float | None = None,
        near_lat: float | None = None,
        near_lon: float | None = None,
        limit: int = 5,
    ) -> list[Hotel]:
        """
        Search hotels using cache-first retrieval with live fallback.

        Args:
            session: Active database session.
            destination: Destination name.
            nightly_budget: Optional nightly budget ceiling.
            near_lat: Optional latitude for proximity ranking to activity area.
            near_lon: Optional longitude for proximity ranking to activity area.
            limit: Maximum number of results.

        Returns:
            Matching hotels ranked by suitability.
        """

        logger.info(
            "search_hotels: destination='%s' nightly_budget=%s near=(%s,%s) limit=%d",
            destination,
            nightly_budget,
            near_lat,
            near_lon,
            limit,
        )
        popularity = self.xiaohongshu_client.search_popularity(session, destination)
        fresh_cached = self._load_cached_hotels(
            session=session,
            destination=destination,
            fresh_only=True,
        )
        filtered_fresh = self._filter_hotels(
            hotels=fresh_cached,
            nightly_budget=nightly_budget,
            near_lat=near_lat,
            near_lon=near_lon,
            limit=limit,
            popularity=popularity,
        )
        if filtered_fresh:
            self._log_search_results(
                "search_hotels", "fresh_cache", len(filtered_fresh)
            )
            return filtered_fresh

        live_results = self.places_client.search_hotels(
            destination=destination,
            limit=limit,
        )
        if live_results:
            self._cache_hotels(session, live_results)
            filtered_live = self._filter_hotels(
                hotels=live_results,
                nightly_budget=nightly_budget,
                near_lat=near_lat,
                near_lon=near_lon,
                limit=limit,
                popularity=popularity,
            )
            if filtered_live:
                self._log_search_results(
                    "search_hotels", "live_api", len(filtered_live)
                )
                return filtered_live

        stale_cached = self._load_cached_hotels(
            session=session,
            destination=destination,
            fresh_only=False,
        )
        filtered_stale = self._filter_hotels(
            hotels=stale_cached,
            nightly_budget=nightly_budget,
            near_lat=near_lat,
            near_lon=near_lon,
            limit=limit,
            popularity=popularity,
        )
        if filtered_stale:
            self._log_search_results(
                "search_hotels", "stale_cache", len(filtered_stale)
            )
            return filtered_stale

        seed_records = self._load_seed_hotels(session, destination)
        filtered_seed = self._filter_hotels(
            hotels=seed_records,
            nightly_budget=nightly_budget,
            near_lat=near_lat,
            near_lon=near_lon,
            limit=limit,
            popularity=popularity,
        )
        self._log_search_results("search_hotels", "seed", len(filtered_seed))
        return filtered_seed

    def list_accommodations(
        self,
        session: Session,
        destination: str,
        nightly_budget: float | None = None,
        limit: int = 5,
    ) -> list[Hotel]:
        """
        Retrieve accommodation options.

        Args:
            session: Active database session.
            destination: Destination name.
            nightly_budget: Optional budget ceiling per night.
            limit: Maximum number of options.

        Returns:
            Matching hotel options.
        """

        return self.search_hotels(
            session=session,
            destination=destination,
            nightly_budget=nightly_budget,
            limit=limit,
        )

    def prefetch_xhs_guided(
        self,
        session: Session,
        destination: str,
        interests: list[str] | None = None,
        nightly_budget: float | None = None,
    ) -> XhsDiscovery:
        """
        Warm caches using Xiaohongshu-discovered place hints.

        After finding places via Google Places, performs secondary XHS searches
        for each discovered place to collect richer note titles for AI planning.

        Args:
            session: Active database session.
            destination: Destination name.
            interests: Optional interest list reserved for future refinement.
            nightly_budget: Optional nightly budget reserved for future refinement.
        """

        _ = (interests, nightly_budget)
        logger.info("prefetch_xhs_guided: warming caches for '%s'", destination)
        discovery = self.xiaohongshu_client.discover_destination(session, destination)
        if not any(
            discovery.hints_by_category.get(category)
            for category in ("attraction", "hotel", "restaurant")
        ):
            logger.info(
                "prefetch_xhs_guided: cached %d attractions, %d hotels, %d restaurants",
                0,
                0,
                0,
            )
            return discovery

        attraction_results: list[Attraction] = []
        for hint_title, _ in discovery.hints_by_category.get("attraction", [])[:5]:
            attraction_results.extend(
                self.places_client.search_by_name(
                    destination=destination,
                    name=hint_title,
                    place_type="tourist_attraction",
                    limit=3,
                )
            )
        if attraction_results:
            unique_attractions = list(
                {item.id: item for item in attraction_results}.values()
            )
            self._cache_attractions(session, unique_attractions)
        else:
            unique_attractions = []

        hotel_results: list[Hotel] = []
        for hint_title, _ in discovery.hints_by_category.get("hotel", [])[:3]:
            hotel_results.extend(
                self.places_client.search_by_name(
                    destination=destination,
                    name=hint_title,
                    place_type="lodging",
                    limit=3,
                )
            )
        if hotel_results:
            unique_hotels = list({item.id: item for item in hotel_results}.values())
            self._cache_hotels(session, unique_hotels)
        else:
            unique_hotels = []

        restaurant_results: list[Restaurant] = []
        for hint_title, _ in discovery.hints_by_category.get("restaurant", [])[:5]:
            restaurant_results.extend(
                self.places_client.search_by_name(
                    destination=destination,
                    name=hint_title,
                    place_type="restaurant",
                    limit=3,
                )
            )
        if restaurant_results:
            unique_restaurants = list(
                {item.id: item for item in restaurant_results}.values()
            )
            self._cache_restaurants(session, unique_restaurants)
        else:
            unique_restaurants = []

        place_notes: dict[str, list[str]] = {}
        all_place_names: list[str] = []
        for attraction in unique_attractions[:5]:
            all_place_names.append(attraction.name)
        for hotel in unique_hotels[:3]:
            all_place_names.append(hotel.name)
        for restaurant in unique_restaurants[:5]:
            all_place_names.append(restaurant.name)
        for place_name in all_place_names:
            notes = self.xiaohongshu_client.search_place_notes(place_name, limit=5)
            if notes:
                place_notes[place_name] = notes
        discovery.place_notes = place_notes

        logger.info(
            "prefetch_xhs_guided: cached %d attractions, %d hotels, %d restaurants, "
            "%d places with XHS notes",
            len(unique_attractions),
            len(unique_hotels),
            len(unique_restaurants),
            len(place_notes),
        )
        return discovery

    def _load_cached_attractions(
        self,
        session: Session,
        destination: str,
        fresh_only: bool,
    ) -> list[Attraction]:
        """
        Load cached Google-sourced attractions for a destination.

        Args:
            session: Active database session.
            destination: Destination name.
            fresh_only: Whether to require a non-expired cache entry.

        Returns:
            Cached attractions.
        """

        statement = select(AttractionRecord).where(
            AttractionRecord.destination == destination,
            AttractionRecord.source == "google_places",
        )
        records = session.exec(statement).all()
        filtered_records = [
            record
            for record in records
            if not fresh_only or self._is_cache_fresh(record.fetched_at)
        ]
        return [record.to_model() for record in filtered_records]

    def _load_seed_attractions(
        self,
        session: Session,
        destination: str,
    ) -> list[Attraction]:
        """
        Load non-Google attraction records for local fallback.

        Args:
            session: Active database session.
            destination: Destination name.

        Returns:
            Local fallback attractions.
        """

        statement = select(AttractionRecord).where(
            AttractionRecord.destination == destination,
            AttractionRecord.source != "google_places",
        )
        records = session.exec(statement).all()
        return [record.to_model() for record in records]

    def _load_cached_restaurants(
        self,
        session: Session,
        destination: str,
        fresh_only: bool,
    ) -> list[Restaurant]:
        """
        Load cached Google-sourced restaurants for a destination.

        Args:
            session: Active database session.
            destination: Destination name.
            fresh_only: Whether to require a non-expired cache entry.

        Returns:
            Cached restaurants.
        """

        statement = select(RestaurantRecord).where(
            RestaurantRecord.destination == destination,
            RestaurantRecord.source == "google_places",
        )
        records = session.exec(statement).all()
        filtered_records = [
            record
            for record in records
            if not fresh_only or self._is_cache_fresh(record.fetched_at)
        ]
        return [record.to_model() for record in filtered_records]

    def _load_seed_restaurants(
        self,
        session: Session,
        destination: str,
    ) -> list[Restaurant]:
        """
        Load non-Google restaurant records for local fallback.

        Args:
            session: Active database session.
            destination: Destination name.

        Returns:
            Local fallback restaurants.
        """

        statement = select(RestaurantRecord).where(
            RestaurantRecord.destination == destination,
            RestaurantRecord.source != "google_places",
        )
        records = session.exec(statement).all()
        return [record.to_model() for record in records]

    def _load_cached_hotels(
        self,
        session: Session,
        destination: str,
        fresh_only: bool,
    ) -> list[Hotel]:
        """
        Load cached Google-sourced hotels for a destination.

        Args:
            session: Active database session.
            destination: Destination name.
            fresh_only: Whether to require a non-expired cache entry.

        Returns:
            Cached hotels.
        """

        statement = select(AccommodationRecord).where(
            AccommodationRecord.destination == destination,
            AccommodationRecord.source == "google_places",
        )
        records = session.exec(statement).all()
        filtered_records = [
            record
            for record in records
            if not fresh_only or self._is_cache_fresh(record.fetched_at)
        ]
        return [record.to_model() for record in filtered_records]

    def _load_seed_hotels(
        self,
        session: Session,
        destination: str,
    ) -> list[Hotel]:
        """
        Load non-Google hotels for local fallback.

        Args:
            session: Active database session.
            destination: Destination name.

        Returns:
            Local fallback hotels.
        """

        statement = select(AccommodationRecord).where(
            AccommodationRecord.destination == destination,
            AccommodationRecord.source != "google_places",
        )
        records = session.exec(statement).all()
        return [record.to_model() for record in records]

    def _filter_attractions(
        self,
        attractions: list[Attraction],
        categories: list[str] | None,
        max_cost: float | None,
        limit: int,
        popularity: dict[str, float] | None = None,
    ) -> list[Attraction]:
        """
        Apply local ranking and filters to attraction candidates.

        Uses category-aware round-robin interleaving to ensure diverse types
        appear in the final selection rather than over-representing one category.

        Args:
            attractions: Candidate attractions.
            categories: Optional desired categories or tags.
            max_cost: Optional cost ceiling per attraction.
            limit: Maximum number of results.
            popularity: Optional Xiaohongshu popularity mapping.

        Returns:
            Filtered and ranked attractions.
        """

        lowered_categories = {item.lower() for item in categories or []}
        filtered = attractions
        if lowered_categories:
            filtered = [
                attraction
                for attraction in filtered
                if attraction.category.lower() in lowered_categories
                or bool(
                    lowered_categories.intersection(
                        {tag.lower() for tag in attraction.tags}
                    )
                )
            ]
        if max_cost is not None:
            filtered = [
                attraction for attraction in filtered if attraction.cost <= max_cost
            ]
        filtered.sort(
            key=lambda item: (
                -(item.rating + self._compute_popularity_boost(item.name, popularity)),
                item.cost,
                item.name,
            )
        )
        return self._interleave_by_category(filtered, limit)

    def _interleave_by_category(
        self,
        attractions: list[Attraction],
        limit: int,
    ) -> list[Attraction]:
        """
        Round-robin interleave attractions by category to ensure diversity.

        Within each category, attractions retain their original ranking order.

        Args:
            attractions: Pre-sorted attraction candidates.
            limit: Maximum number of results.

        Returns:
            Diversified attraction list.
        """

        if len(attractions) <= limit:
            return attractions

        buckets: dict[str, list[Attraction]] = {}
        for attraction in attractions:
            category = attraction.category.lower()
            buckets.setdefault(category, []).append(attraction)

        if len(buckets) <= 1:
            return attractions[:limit]

        bucket_order = sorted(
            buckets.keys(),
            key=lambda cat: (-len(buckets[cat]), cat),
        )
        result: list[Attraction] = []
        seen_ids: set[str] = set()
        iterators = {cat: iter(items) for cat, items in buckets.items()}
        while len(result) < limit:
            added_this_round = False
            for cat in bucket_order:
                if len(result) >= limit:
                    break
                for attraction in iterators[cat]:
                    if attraction.id not in seen_ids:
                        result.append(attraction)
                        seen_ids.add(attraction.id)
                        added_this_round = True
                        break
            if not added_this_round:
                break
        return result

    def _filter_restaurants(
        self,
        restaurants: list[Restaurant],
        meal_type: str,
        near_lat: float | None,
        near_lon: float | None,
        max_cost: float | None,
        limit: int,
        popularity: dict[str, float] | None = None,
    ) -> list[Restaurant]:
        """
        Apply local ranking and filters to restaurant candidates.

        For lunch, proper meal restaurants are ranked higher than cafes and
        similar light-meal establishments.

        Args:
            restaurants: Candidate restaurants.
            meal_type: Meal slot name.
            near_lat: Optional latitude used for proximity ranking.
            near_lon: Optional longitude used for proximity ranking.
            max_cost: Optional per-person budget ceiling.
            limit: Maximum number of results.
            popularity: Optional Xiaohongshu popularity mapping.

        Returns:
            Filtered and ranked restaurants.
        """

        lowered_meal_type = meal_type.lower()
        filtered = [
            restaurant
            for restaurant in restaurants
            if lowered_meal_type in {item.lower() for item in restaurant.meal_types}
        ]
        if max_cost is not None:
            filtered = [
                restaurant
                for restaurant in filtered
                if restaurant.average_cost <= max_cost
            ]
        if near_lat is not None and near_lon is not None:
            filtered.sort(
                key=lambda item: (
                    self._haversine_meters(
                        near_lat,
                        near_lon,
                        item.latitude,
                        item.longitude,
                    ),
                    -item.rating,
                    item.average_cost,
                    item.name,
                )
            )
            nearby_restaurants = filtered[: max(limit * 2, limit)]
            nearby_restaurants.sort(
                key=lambda item: (
                    self._cafe_penalty(item) if lowered_meal_type == "lunch" else 0,
                    -(
                        item.rating
                        + self._compute_popularity_boost(item.name, popularity)
                    ),
                    item.average_cost,
                    item.name,
                )
            )
            return nearby_restaurants[:limit]
        filtered.sort(
            key=lambda item: (
                self._cafe_penalty(item) if lowered_meal_type == "lunch" else 0,
                -(item.rating + self._compute_popularity_boost(item.name, popularity)),
                item.average_cost,
                item.name,
            )
        )
        return filtered[:limit]

    def _cafe_penalty(self, restaurant: Restaurant) -> int:
        """
        Return a sort penalty for cafe-like restaurants to deprioritize them for lunch.

        Args:
            restaurant: Restaurant to evaluate.

        Returns:
            ``1`` for cafe-like establishments, ``0`` for proper meal restaurants.
        """

        cafe_indicators = {"cafe", "coffee", "bakery", "tea", "brunch"}
        name_lower = restaurant.name.lower()
        cuisine_lower = restaurant.cuisine.lower()
        tags_lower = {tag.lower() for tag in restaurant.tags}
        if cuisine_lower in cafe_indicators:
            return 1
        if tags_lower.intersection(cafe_indicators):
            return 1
        if any(indicator in name_lower for indicator in cafe_indicators):
            return 1
        return 0

    def _filter_hotels(
        self,
        hotels: list[Hotel],
        nightly_budget: float | None,
        limit: int,
        near_lat: float | None = None,
        near_lon: float | None = None,
        popularity: dict[str, float] | None = None,
    ) -> list[Hotel]:
        """
        Apply local ranking and filters to hotel candidates.

        When activity-area coordinates are provided, hotels closer to that
        centroid are ranked higher.

        Args:
            hotels: Candidate hotels.
            nightly_budget: Optional nightly budget ceiling.
            limit: Maximum number of results.
            near_lat: Optional latitude of activity area centroid.
            near_lon: Optional longitude of activity area centroid.
            popularity: Optional Xiaohongshu popularity mapping.

        Returns:
            Filtered and ranked hotels.
        """

        filtered = hotels
        if nightly_budget is not None and nightly_budget > 0:
            filtered = [
                hotel for hotel in filtered if hotel.price_per_night <= nightly_budget
            ]
        if near_lat is not None and near_lon is not None:
            filtered.sort(
                key=lambda item: (
                    self._haversine_meters(
                        near_lat,
                        near_lon,
                        item.latitude,
                        item.longitude,
                    ),
                    -(
                        item.star_rating
                        + self._compute_popularity_boost(item.name, popularity)
                    ),
                    item.price_per_night,
                    item.name,
                )
            )
            nearby = filtered[: max(limit * 2, limit)]
            nearby.sort(
                key=lambda item: (
                    -(
                        item.star_rating
                        + self._compute_popularity_boost(item.name, popularity)
                    ),
                    item.price_per_night,
                    item.name,
                )
            )
            return nearby[:limit]
        filtered.sort(
            key=lambda item: (
                -(
                    item.star_rating
                    + self._compute_popularity_boost(item.name, popularity)
                ),
                item.price_per_night,
                item.name,
            )
        )
        return filtered[:limit]

    def _compute_popularity_boost(
        self,
        name: str,
        popularity: dict[str, float] | None,
    ) -> float:
        """
        Compute a normalized ranking boost from Xiaohongshu note-title matches.

        Args:
            name: Attraction or restaurant name.
            popularity: Popularity map keyed by lowercase note titles.

        Returns:
            A normalized popularity boost between ``0.0`` and ``2.0``.
        """

        if not popularity:
            return 0.0
        name_lower = name.strip().lower()
        if not name_lower:
            return 0.0

        total_score = 0.0
        for title, score in popularity.items():
            if name_lower in title:
                total_score += score
        if total_score <= 0:
            return 0.0
        return min(2.0, math.log1p(total_score) / 10.0)

    def _cache_attractions(
        self,
        session: Session,
        attractions: list[Attraction],
    ) -> None:
        """
        Persist live attractions into the local cache.

        Args:
            session: Active database session.
            attractions: Fresh live attractions.
        """

        fetched_at = datetime.now(UTC)
        for attraction in attractions:
            record = AttractionRecord.from_model(attraction, fetched_at=fetched_at)
            session.merge(record)
        session.commit()

    def _cache_restaurants(
        self,
        session: Session,
        restaurants: list[Restaurant],
    ) -> None:
        """
        Persist live restaurants into the local cache.

        Args:
            session: Active database session.
            restaurants: Fresh live restaurants.
        """

        fetched_at = datetime.now(UTC)
        for restaurant in restaurants:
            merged_restaurant = restaurant
            existing_record = session.get(RestaurantRecord, restaurant.id)
            if existing_record is not None:
                existing_model = existing_record.to_model()
                merged_restaurant = restaurant.model_copy(
                    update={
                        "meal_types": sorted(
                            {
                                *existing_model.meal_types,
                                *restaurant.meal_types,
                            }
                        ),
                        "tags": sorted({*existing_model.tags, *restaurant.tags}),
                    }
                )
            record = RestaurantRecord.from_model(
                merged_restaurant,
                fetched_at=fetched_at,
            )
            session.merge(record)
        session.commit()

    def _cache_hotels(
        self,
        session: Session,
        hotels: list[Hotel],
    ) -> None:
        """
        Persist live hotels into the local cache.

        Args:
            session: Active database session.
            hotels: Fresh live hotels.
        """

        fetched_at = datetime.now(UTC)
        for hotel in hotels:
            session.merge(AccommodationRecord.from_model(hotel, fetched_at=fetched_at))
        session.commit()

    def _is_cache_fresh(self, fetched_at: datetime | None) -> bool:
        """
        Return whether a cache timestamp is still valid.

        Args:
            fetched_at: Cache timestamp.

        Returns:
            ``True`` when the cache entry is within the configured TTL.
        """

        if fetched_at is None:
            return False
        normalized = fetched_at
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=UTC)
        cutoff = datetime.now(UTC) - timedelta(
            hours=self.settings.places_cache_ttl_hours
        )
        return normalized >= cutoff

    def _haversine_meters(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
    ) -> float:
        """
        Calculate the Haversine distance in meters.

        Args:
            origin_lat: Origin latitude.
            origin_lon: Origin longitude.
            destination_lat: Destination latitude.
            destination_lon: Destination longitude.

        Returns:
            Straight-line distance in meters.
        """

        earth_radius_m = 6_371_000
        latitude_delta = math.radians(destination_lat - origin_lat)
        longitude_delta = math.radians(destination_lon - origin_lon)
        origin_lat_rad = math.radians(origin_lat)
        destination_lat_rad = math.radians(destination_lat)
        component = (
            math.sin(latitude_delta / 2) ** 2
            + math.cos(origin_lat_rad)
            * math.cos(destination_lat_rad)
            * math.sin(longitude_delta / 2) ** 2
        )
        return (
            2
            * earth_radius_m
            * math.atan2(math.sqrt(component), math.sqrt(1 - component))
        )

    def _log_search_results(
        self,
        search_name: str,
        source_label: str,
        count: int,
    ) -> None:
        """
        Log a cache/live source decision for a search operation.

        Args:
            search_name: Search operation name.
            source_label: Result source label.
            count: Number of returned results.
        """

        logger.info(
            "%s: source=%s returned %d results", search_name, source_label, count
        )
