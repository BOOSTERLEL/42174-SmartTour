"""
Multi-day itinerary generation with geographic clustering and feasibility checks.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Session

from smarttour.config import get_settings
from smarttour.models import (
    Attraction,
    DayPlan,
    Itinerary,
    PlanningSession,
    PlanningSessionRecord,
    TimeSlot,
    TravelPreferences,
)
from smarttour.services.data_retrieval import DataRetrievalService
from smarttour.services.maps_client import MapsClient
from smarttour.services.route_optimizer import RouteOptimizer


class ItineraryGenerator:
    """
    Build a feasible itinerary from preferences and database content.
    """

    def __init__(
        self,
        data_retrieval: DataRetrievalService | None = None,
        maps_client: MapsClient | None = None,
        route_optimizer: RouteOptimizer | None = None,
    ) -> None:
        """
        Initialize the itinerary generator.

        Args:
            data_retrieval: Optional attraction retrieval service.
            maps_client: Optional map service.
            route_optimizer: Optional route optimizer.
        """

        self.settings = get_settings()
        self.data_retrieval = data_retrieval or DataRetrievalService()
        self.maps_client = maps_client or MapsClient()
        self.route_optimizer = route_optimizer or RouteOptimizer()

    def generate(
        self,
        session: Session,
        preferences: TravelPreferences,
        user_input: str = "",
    ) -> PlanningSession:
        """
        Generate and persist a planning session.

        Args:
            session: Active database session.
            preferences: Structured travel preferences.
            user_input: Original user prompt.

        Returns:
            The persisted planning session.
        """

        max_cost = self._max_cost_per_attraction(preferences)
        limit = preferences.trip_days * self.settings.max_attractions_per_day
        attractions = self.data_retrieval.search_attractions(
            session=session,
            destination=preferences.destination,
            categories=preferences.interests,
            max_cost=max_cost,
            limit=limit,
        )
        if not attractions:
            attractions = self.data_retrieval.search_attractions(
                session=session,
                destination=preferences.destination,
                limit=limit,
            )
        if not attractions:
            raise ValueError(
                f"No attractions available for destination '{preferences.destination}'."
            )

        daily_groups = self._cluster_attractions_by_geography(
            attractions, preferences.trip_days
        )

        hotels = self.data_retrieval.list_accommodations(
            session=session,
            destination=preferences.destination,
            nightly_budget=self._nightly_budget(preferences),
            limit=1,
        )

        day_plans: list[DayPlan] = []
        warnings: list[str] = []
        total_cost = 0.0
        for day_number, bucket in enumerate(daily_groups, start=1):
            if not bucket:
                continue
            graph = self.maps_client.build_route_graph(
                session, bucket, preferences.travel_mode
            )
            route = self.route_optimizer.optimize(
                bucket, graph, preferences.travel_mode
            )
            ordered_bucket = self._ordered_attractions(route.attraction_order, bucket)
            day_plan = self._build_day_plan(
                day_number, ordered_bucket, route, preferences
            )
            if day_plan.warnings:
                warnings.extend(day_plan.warnings)
            total_cost += day_plan.estimated_cost
            day_plans.append(day_plan)

        itinerary = Itinerary(
            id=uuid4().hex,
            destination=preferences.destination,
            created_at=datetime.now(UTC),
            preferences=preferences,
            accommodation=hotels[0] if hotels else None,
            days=day_plans,
            total_estimated_cost=round(total_cost, 2),
            warnings=warnings,
        )
        planning_session = PlanningSession(
            id=itinerary.id,
            created_at=itinerary.created_at,
            user_input=user_input,
            preferences=preferences,
            itinerary=itinerary,
            version=1,
        )
        session.add(PlanningSessionRecord.from_model(planning_session))
        session.commit()
        return planning_session

    def regenerate(
        self,
        session: Session,
        previous_session_id: str,
        preferences: TravelPreferences,
        user_input: str = "",
    ) -> PlanningSession:
        """
        Regenerate an itinerary linked to a previous session with incremented version.

        Args:
            session: Active database session.
            previous_session_id: The session ID of the previous generation.
            preferences: Updated travel preferences.
            user_input: Original user prompt.

        Returns:
            A new planning session with incremented version.
        """

        previous_version = 0
        previous_record = session.get(PlanningSessionRecord, previous_session_id)
        if previous_record is not None:
            previous_version = previous_record.version

        max_cost = self._max_cost_per_attraction(preferences)
        limit = preferences.trip_days * self.settings.max_attractions_per_day
        attractions = self.data_retrieval.search_attractions(
            session=session,
            destination=preferences.destination,
            categories=preferences.interests,
            max_cost=max_cost,
            limit=limit,
        )
        if not attractions:
            attractions = self.data_retrieval.search_attractions(
                session=session,
                destination=preferences.destination,
                limit=limit,
            )
        if not attractions:
            raise ValueError(
                f"No attractions available for destination '{preferences.destination}'."
            )

        daily_groups = self._cluster_attractions_by_geography(
            attractions, preferences.trip_days
        )

        hotels = self.data_retrieval.list_accommodations(
            session=session,
            destination=preferences.destination,
            nightly_budget=self._nightly_budget(preferences),
            limit=1,
        )

        day_plans: list[DayPlan] = []
        warnings: list[str] = []
        total_cost = 0.0
        for day_number, bucket in enumerate(daily_groups, start=1):
            if not bucket:
                continue
            graph = self.maps_client.build_route_graph(
                session, bucket, preferences.travel_mode
            )
            route = self.route_optimizer.optimize(
                bucket, graph, preferences.travel_mode
            )
            ordered_bucket = self._ordered_attractions(route.attraction_order, bucket)
            day_plan = self._build_day_plan(
                day_number, ordered_bucket, route, preferences
            )
            if day_plan.warnings:
                warnings.extend(day_plan.warnings)
            total_cost += day_plan.estimated_cost
            day_plans.append(day_plan)

        itinerary = Itinerary(
            id=uuid4().hex,
            destination=preferences.destination,
            created_at=datetime.now(UTC),
            preferences=preferences,
            accommodation=hotels[0] if hotels else None,
            days=day_plans,
            total_estimated_cost=round(total_cost, 2),
            warnings=warnings,
        )
        new_version = previous_version + 1
        planning_session = PlanningSession(
            id=itinerary.id,
            created_at=itinerary.created_at,
            user_input=user_input,
            preferences=preferences,
            itinerary=itinerary,
            version=new_version,
        )
        session.add(PlanningSessionRecord.from_model(planning_session))
        session.commit()
        return planning_session

    def get_session(self, session: Session, session_id: str) -> PlanningSession | None:
        """
        Retrieve a saved planning session.

        Args:
            session: Active database session.
            session_id: Session identifier.

        Returns:
            The saved planning session or ``None``.
        """

        record = session.get(PlanningSessionRecord, session_id)
        if record is None:
            return None
        return record.to_model()

    def _max_cost_per_attraction(self, preferences: TravelPreferences) -> float | None:
        """
        Estimate a reasonable attraction-level budget cap.

        Args:
            preferences: Structured travel preferences.

        Returns:
            Estimated maximum attraction cost or ``None``.
        """

        if preferences.budget_total <= 0:
            return None
        denominator = max(
            1, preferences.trip_days * self.settings.max_attractions_per_day
        )
        return max(20.0, preferences.budget_total * 0.45 / denominator)

    def _nightly_budget(self, preferences: TravelPreferences) -> float | None:
        """
        Estimate accommodation budget per night.

        Args:
            preferences: Structured travel preferences.

        Returns:
            Estimated nightly budget or ``None``.
        """

        if preferences.budget_total <= 0:
            return None
        return max(80.0, preferences.budget_total * 0.35 / preferences.trip_days)

    def _cluster_attractions_by_geography(
        self,
        attractions: list[Attraction],
        trip_days: int,
    ) -> list[list[Attraction]]:
        """
        Cluster attractions into daily groups based on geographic proximity.

        Uses a greedy nearest-centroid approach to keep each day's attractions
        geographically close, minimizing intra-day travel distance.

        Args:
            attractions: Ranked attraction list.
            trip_days: Trip duration.

        Returns:
            A list of daily attraction buckets.
        """

        max_per_day = self.settings.max_attractions_per_day
        if trip_days <= 1:
            return [attractions[:max_per_day]]
        if len(attractions) <= trip_days:
            groups: list[list[Attraction]] = [[a] for a in attractions]
            while len(groups) < trip_days:
                groups.append([])
            return groups[:trip_days]

        centroids = self._pick_initial_centroids(attractions, trip_days)
        groups = [[] for _ in range(trip_days)]  # type: ignore[var-annotated]

        for attraction in attractions:
            best_group = 0
            best_distance = float("inf")
            for group_index, centroid in enumerate(centroids):
                if len(groups[group_index]) >= max_per_day:
                    continue
                distance = self._euclidean_geo_distance(
                    attraction.latitude,
                    attraction.longitude,
                    centroid[0],
                    centroid[1],
                )
                if distance < best_distance:
                    best_distance = distance
                    best_group = group_index
            groups[best_group].append(attraction)

        for group_index in range(trip_days):
            if not groups[group_index]:
                continue
            lats = [a.latitude for a in groups[group_index]]
            lons = [a.longitude for a in groups[group_index]]
            centroids[group_index] = (sum(lats) / len(lats), sum(lons) / len(lons))

        return groups

    def _pick_initial_centroids(
        self, attractions: list[Attraction], k: int
    ) -> list[tuple[float, float]]:
        """
        Pick k initial centroids spread across the attraction set.

        Uses a maximin strategy: first centroid is the first attraction,
        each subsequent centroid is the attraction farthest from existing centroids.

        Args:
            attractions: List of attractions.
            k: Number of centroids (= trip_days).

        Returns:
            List of (latitude, longitude) tuples.
        """

        centroids: list[tuple[float, float]] = [
            (attractions[0].latitude, attractions[0].longitude)
        ]
        for _ in range(1, min(k, len(attractions))):
            best_attraction = None
            best_min_dist = -1.0
            for attraction in attractions:
                min_dist = min(
                    self._euclidean_geo_distance(
                        attraction.latitude, attraction.longitude, c[0], c[1]
                    )
                    for c in centroids
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_attraction = attraction
            if best_attraction is not None:
                centroids.append((best_attraction.latitude, best_attraction.longitude))
        while len(centroids) < k:
            centroids.append(centroids[-1])
        return centroids

    def _euclidean_geo_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Fast approximate distance for clustering (not for routing).

        Args:
            lat1: First latitude.
            lon1: First longitude.
            lat2: Second latitude.
            lon2: Second longitude.

        Returns:
            Approximate distance in degrees.
        """

        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    def _ordered_attractions(
        self,
        attraction_order: list[str],
        bucket: list[Attraction],
    ) -> list[Attraction]:
        """
        Reorder attractions using route output.

        Args:
            attraction_order: Ordered attraction identifiers.
            bucket: Original daily attraction list.

        Returns:
            Attractions in route order.
        """

        mapping = {attraction.id: attraction for attraction in bucket}
        return [
            mapping[attraction_id]
            for attraction_id in attraction_order
            if attraction_id in mapping
        ]

    def _build_day_plan(
        self,
        day_number: int,
        attractions: list[Attraction],
        route,
        preferences: TravelPreferences,
    ) -> DayPlan:
        """
        Build a day plan with time slots, opening hours validation, and feasibility checks.

        Args:
            day_number: Day number in the trip.
            attractions: Ordered attractions for the day.
            route: Optimized route details.
            preferences: Structured travel preferences.

        Returns:
            A daily plan.
        """

        current_minutes = preferences.preferred_start_hour * 60
        closing_minutes = preferences.preferred_end_hour * 60
        slots: list[TimeSlot] = []
        warnings: list[str] = []
        segment_lookup = {
            (segment.origin_id, segment.destination_id): segment
            for segment in route.segments
        }
        total_cost = 0.0
        previous_attraction: Attraction | None = None

        for attraction in attractions:
            transport = None
            if previous_attraction is not None:
                transport = segment_lookup.get((previous_attraction.id, attraction.id))
                if transport is not None:
                    current_minutes += max(5, round(transport.duration_s / 60))

            opening_warning = self._check_opening_hours(
                attraction, current_minutes, day_number
            )
            if opening_warning:
                warnings.append(opening_warning)

            start_minutes = current_minutes
            end_minutes = start_minutes + attraction.visit_duration
            slots.append(
                TimeSlot(
                    start_time=self._format_minutes(start_minutes),
                    end_time=self._format_minutes(end_minutes),
                    title=attraction.name,
                    description=attraction.description,
                    attraction=attraction,
                    transport_from_previous=transport,
                    cost=attraction.cost,
                )
            )
            current_minutes = end_minutes
            total_cost += attraction.cost
            previous_attraction = attraction

        if current_minutes > closing_minutes:
            warnings.append(
                f"Day {day_number} ends after the preferred time window "
                f"({self._format_minutes(current_minutes)})."
            )

        return DayPlan(
            day_number=day_number,
            label=f"Day {day_number}",
            slots=slots,
            route=route,
            estimated_cost=round(total_cost, 2),
            warnings=warnings,
        )

    def _check_opening_hours(
        self,
        attraction: Attraction,
        arrival_minutes: int,
        day_number: int,
    ) -> str | None:
        """
        Validate that the arrival time falls within the attraction's opening hours.

        Args:
            attraction: The attraction to check.
            arrival_minutes: Planned arrival time as minutes from midnight.
            day_number: The day number in the trip.

        Returns:
            A warning string if the attraction may be closed, otherwise ``None``.
        """

        if not attraction.opening_hours:
            return None

        day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        hours_value: str | None = None
        for day_name in day_names:
            if day_name in attraction.opening_hours:
                hours_value = attraction.opening_hours[day_name]
                break
        if hours_value is None:
            first_key = next(iter(attraction.opening_hours))
            hours_value = attraction.opening_hours[first_key]

        if not hours_value or hours_value.lower() in ("closed", "n/a"):
            return (
                f"Day {day_number}: {attraction.name} may be closed "
                f"(listed as '{hours_value}')."
            )

        parts = hours_value.replace(" ", "").split("-")
        if len(parts) != 2:
            return None

        try:
            open_minutes = self._parse_time_to_minutes(parts[0])
            close_minutes = self._parse_time_to_minutes(parts[1])
        except ValueError:
            return None

        if arrival_minutes < open_minutes:
            return (
                f"Day {day_number}: {attraction.name} opens at "
                f"{self._format_minutes(open_minutes)} but arrival is scheduled "
                f"at {self._format_minutes(arrival_minutes)}."
            )
        if arrival_minutes >= close_minutes:
            return (
                f"Day {day_number}: {attraction.name} closes at "
                f"{self._format_minutes(close_minutes)} but arrival is scheduled "
                f"at {self._format_minutes(arrival_minutes)}."
            )

        return None

    def _parse_time_to_minutes(self, time_str: str) -> int:
        """
        Parse a time string like '09:00' or '9:30' into minutes from midnight.

        Args:
            time_str: Time string.

        Returns:
            Minutes from midnight.

        Raises:
            ValueError: If the time string cannot be parsed.
        """

        parts = time_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid time format: {time_str}")
        return int(parts[0]) * 60 + int(parts[1])

    def _format_minutes(self, total_minutes: int) -> str:
        """
        Format integer minutes from midnight into HH:MM.

        Args:
            total_minutes: Minutes from midnight.

        Returns:
            A formatted time string.
        """

        hours, minutes = divmod(total_minutes, 60)
        return f"{hours:02d}:{minutes:02d}"
