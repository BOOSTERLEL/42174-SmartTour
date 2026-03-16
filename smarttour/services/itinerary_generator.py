"""
Multi-day itinerary generation with geographic clustering and feasibility checks.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Session

from smarttour.config import get_settings
from smarttour.models import (
    Attraction,
    DayPlan,
    Hotel,
    Itinerary,
    ItineraryPlan,
    PlanningSession,
    PlanningSessionRecord,
    Restaurant,
    Route,
    TimeSlot,
    TransportSegment,
    TravelPreferences,
)
from smarttour.services.data_retrieval import DataRetrievalService
from smarttour.services.itinerary_planner import ItineraryPlanner
from smarttour.services.maps_client import MapsClient
from smarttour.services.route_optimizer import RouteOptimizer

logger = logging.getLogger(__name__)


class ItineraryGenerator:
    """
    Build a feasible itinerary from preferences and database content.
    """

    def __init__(
        self,
        data_retrieval: DataRetrievalService | None = None,
        maps_client: MapsClient | None = None,
        route_optimizer: RouteOptimizer | None = None,
        planner: ItineraryPlanner | None = None,
    ) -> None:
        """
        Initialize the itinerary generator.

        Args:
            data_retrieval: Optional attraction retrieval service.
            maps_client: Optional map service.
            route_optimizer: Optional route optimizer.
            planner: Optional AI itinerary planner.
        """

        self.settings = get_settings()
        self.data_retrieval = data_retrieval or DataRetrievalService()
        self.maps_client = maps_client or MapsClient()
        self.route_optimizer = route_optimizer or RouteOptimizer()
        self.planner = planner or ItineraryPlanner()

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
        itinerary = self._build_itinerary(session, preferences)
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

        itinerary = self._build_itinerary(session, preferences)
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

    def _build_itinerary(
        self,
        session: Session,
        preferences: TravelPreferences,
    ) -> Itinerary:
        """
        Construct an itinerary with attractions, meals, and routing details.

        Args:
            session: Active database session.
            preferences: Structured travel preferences.

        Returns:
            A populated itinerary model.
        """

        logger.info(
            "Building itinerary: destination='%s' days=%d budget=%.0f pace=%s",
            preferences.destination,
            preferences.trip_days,
            preferences.budget_total,
            preferences.pace,
        )
        discovery = self.data_retrieval.prefetch_xhs_guided(
            session=session,
            destination=preferences.destination,
            interests=preferences.interests,
            nightly_budget=self._nightly_budget(preferences),
        )
        popularity = discovery.popularity
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

        centroid_lat = sum(a.latitude for a in attractions) / len(attractions)
        centroid_lon = sum(a.longitude for a in attractions) / len(attractions)

        hotels = self.data_retrieval.search_hotels(
            session=session,
            destination=preferences.destination,
            nightly_budget=self._nightly_budget(preferences),
            near_lat=centroid_lat,
            near_lon=centroid_lon,
            limit=1,
        )
        hotel = hotels[0] if hotels else None
        if hotel is None:
            logger.warning(
                "No hotel found for destination='%s' nightly_budget=%s",
                preferences.destination,
                self._nightly_budget(preferences),
            )
            hotels = self.data_retrieval.search_hotels(
                session=session,
                destination=preferences.destination,
                nightly_budget=None,
                near_lat=centroid_lat,
                near_lon=centroid_lon,
                limit=1,
            )
            hotel = hotels[0] if hotels else None
        plan = self.planner.plan(
            attractions=attractions,
            hotel=hotel,
            preferences=preferences,
            popularity=popularity,
            place_notes=discovery.place_notes,
        )
        ai_plan_reasoning = ""
        if plan is not None and plan.day_themes:
            daily_groups = self._apply_ai_plan(
                attractions=attractions,
                plan=plan,
                trip_days=preferences.trip_days,
            )
            ai_plan_reasoning = plan.reasoning
            grouping_method = "ai_plan"
        else:
            daily_groups = self._cluster_attractions_by_geography(
                attractions, preferences.trip_days
            )
            grouping_method = "geographic_clustering"
        logger.info("Attraction grouping: method=%s", grouping_method)

        day_plans: list[DayPlan] = []
        warnings: list[str] = []
        total_cost = 0.0
        trip_used_restaurant_ids: set[str] = set()
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
            route = self.maps_client.populate_route_polylines(
                session=session,
                route=route,
                locations=ordered_bucket,
            )
            day_plan = self._build_day_plan(
                session=session,
                day_number=day_number,
                attractions=ordered_bucket,
                route=route,
                preferences=preferences,
                trip_used_restaurant_ids=trip_used_restaurant_ids,
                hotel=hotel,
            )
            self._validate_day_constraints(day_plan)
            warnings.extend(day_plan.warnings)
            total_cost += day_plan.estimated_cost
            day_plans.append(day_plan)
            logger.info(
                "Day %d: %d attractions, route=%.0fm, cost=%.2f",
                day_number,
                len(ordered_bucket),
                route.total_distance_m,
                day_plan.estimated_cost,
            )

        itinerary = Itinerary(
            id=uuid4().hex,
            destination=preferences.destination,
            created_at=datetime.now(UTC),
            preferences=preferences,
            accommodation=hotel,
            days=day_plans,
            total_estimated_cost=round(total_cost, 2),
            warnings=warnings,
            ai_plan_reasoning=ai_plan_reasoning,
        )
        review = self.planner.review(itinerary)
        if review is not None:
            itinerary.ai_review = review
            itinerary.warnings.extend(review.revised_warnings)
        logger.info(
            "Itinerary complete: id=%s days=%d total_cost=%.2f warnings=%d",
            itinerary.id,
            len(itinerary.days),
            itinerary.total_estimated_cost,
            len(itinerary.warnings),
        )
        return itinerary

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

    def _apply_ai_plan(
        self,
        attractions: list[Attraction],
        plan: ItineraryPlan,
        trip_days: int,
    ) -> list[list[Attraction]]:
        """
        Convert an AI itinerary plan into daily attraction buckets.

        Args:
            attractions: Available attractions.
            plan: AI-generated itinerary plan.
            trip_days: Number of trip days.

        Returns:
            Daily attraction groups, or a geographic fallback when invalid.
        """

        if trip_days <= 0:
            return self._cluster_attractions_by_geography(attractions, 1)

        attraction_lookup = {attraction.id: attraction for attraction in attractions}
        groups: list[list[Attraction]] = [[] for _ in range(trip_days)]
        assigned_ids: set[str] = set()
        invalid_grouping = False

        for day_theme in sorted(plan.day_themes, key=lambda item: item.day_number):
            day_index = day_theme.day_number - 1
            if day_index < 0 or day_index >= trip_days:
                invalid_grouping = True
                break
            for attraction_id in day_theme.attraction_ids:
                attraction = attraction_lookup.get(attraction_id)
                if attraction is None or attraction_id in assigned_ids:
                    invalid_grouping = True
                    break
                groups[day_index].append(attraction)
                assigned_ids.add(attraction_id)
            if invalid_grouping:
                break

        if invalid_grouping:
            return self._cluster_attractions_by_geography(attractions, trip_days)

        max_per_day = self.settings.max_attractions_per_day
        overflow: list[Attraction] = []
        for group in groups:
            if len(group) > max_per_day:
                overflow.extend(group[max_per_day:])
                del group[max_per_day:]

        for attraction in overflow:
            target_index = min(
                range(len(groups)),
                key=lambda index: (len(groups[index]), index),
            )
            if len(groups[target_index]) < max_per_day:
                groups[target_index].append(attraction)
                assigned_ids.add(attraction.id)

        for attraction in attractions:
            if attraction.id in assigned_ids:
                continue
            target_index = min(
                range(len(groups)),
                key=lambda index: (len(groups[index]), index),
            )
            if len(groups[target_index]) < max_per_day:
                groups[target_index].append(attraction)
                assigned_ids.add(attraction.id)

        flattened_ids = [attraction.id for group in groups for attraction in group]
        if len(groups) != trip_days or len(set(flattened_ids)) != len(flattened_ids):
            return self._cluster_attractions_by_geography(attractions, trip_days)
        return groups

    def _cluster_attractions_by_geography(
        self,
        attractions: list[Attraction],
        trip_days: int,
    ) -> list[list[Attraction]]:
        """
        Cluster attractions into daily groups using an angular sweep.

        After initial clustering, validates that each group fits within the
        configured daily range limit and splits oversized clusters.

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

        mean_lat = sum(attraction.latitude for attraction in attractions) / len(
            attractions
        )
        mean_lon = sum(attraction.longitude for attraction in attractions) / len(
            attractions
        )
        sorted_by_angle = sorted(
            attractions,
            key=lambda attraction: math.atan2(
                attraction.longitude - mean_lon,
                attraction.latitude - mean_lat,
            ),
        )

        chunk_size = min(max_per_day, math.ceil(len(sorted_by_angle) / trip_days))
        groups = [
            sorted_by_angle[index : index + chunk_size]
            for index in range(0, len(sorted_by_angle), chunk_size)
        ]
        while len(groups) < trip_days:
            groups.append([])

        max_range_m = self.settings.max_daily_range_km * 1000
        refined: list[list[Attraction]] = []
        overflow: list[Attraction] = []
        for group in groups:
            if len(group) <= 1:
                refined.append(group)
                continue
            radius = self._cluster_radius_m(group)
            if radius <= max_range_m:
                refined.append(group)
                continue
            centroid_lat = sum(a.latitude for a in group) / len(group)
            centroid_lon = sum(a.longitude for a in group) / len(group)
            group.sort(
                key=lambda a: self._haversine_m(
                    centroid_lat, centroid_lon, a.latitude, a.longitude
                )
            )
            kept: list[Attraction] = []
            for attraction in group:
                candidate = [*kept, attraction]
                if self._cluster_radius_m(candidate) <= max_range_m or len(kept) < 2:
                    kept.append(attraction)
                else:
                    overflow.append(attraction)
            refined.append(kept)

        for attraction in overflow:
            best_index = min(
                range(len(refined)),
                key=lambda i: (len(refined[i]), i),
            )
            if len(refined[best_index]) < max_per_day:
                refined[best_index].append(attraction)

        return refined

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
        session: Session,
        day_number: int,
        attractions: list[Attraction],
        route: Route,
        preferences: TravelPreferences,
        trip_used_restaurant_ids: set[str],
        hotel: Hotel | None = None,
    ) -> DayPlan:
        """
        Build a day plan with meals, attractions, and navigation details.

        Args:
            session: Active database session.
            day_number: Day number in the trip.
            attractions: Ordered attractions for the day.
            route: Optimized route details.
            preferences: Structured travel preferences.
            trip_used_restaurant_ids: Restaurants already used on previous days.
            hotel: Selected hotel used as the daily anchor.

        Returns:
            A daily plan.
        """

        attraction_start_minutes = preferences.preferred_start_hour * 60
        closing_minutes = preferences.preferred_end_hour * 60
        meal_duration = self.settings.meal_duration_minutes
        hotel_transfer_minutes = 15 if hotel is not None else 0
        slots: list[TimeSlot] = []
        warnings: list[str] = []
        segment_lookup = {
            (segment.origin_id, segment.destination_id): segment
            for segment in route.segments
        }
        total_cost = 0.0
        used_restaurant_ids = set(trip_used_restaurant_ids)
        selected_restaurant_ids: set[str] = set()
        current_minutes = max(
            6 * 60,
            attraction_start_minutes - meal_duration - hotel_transfer_minutes,
        )
        previous_stop: Attraction | Restaurant | Hotel | None = None
        meal_budget = self._max_meal_cost_per_person(preferences)
        lunch_window_minutes = self.settings.lunch_window_start * 60
        dinner_window_minutes = self.settings.dinner_window_start * 60
        lunch_insertion_index = self._find_lunch_insertion_index(
            attractions=attractions,
            route=route,
            start_minutes=max(
                current_minutes + hotel_transfer_minutes,
                attraction_start_minutes,
            ),
        )

        if hotel is not None:
            current_minutes, previous_stop = self._append_hotel_slot(
                session=session,
                slot_type="hotel_checkout",
                current_minutes=current_minutes,
                previous_stop=previous_stop,
                hotel=hotel,
                preferences=preferences,
                slots=slots,
                cost=0.0,
            )

        breakfast_minutes, previous_stop, breakfast_cost, _ = self._append_meal_slot(
            session=session,
            day_number=day_number,
            meal_type="breakfast",
            current_minutes=current_minutes,
            previous_stop=previous_stop,
            anchor=hotel or attractions[0],
            preferences=preferences,
            used_restaurant_ids=used_restaurant_ids,
            selected_restaurant_ids=selected_restaurant_ids,
            meal_budget=meal_budget,
            slots=slots,
            warnings=warnings,
        )
        current_minutes = max(breakfast_minutes, attraction_start_minutes)
        total_cost += breakfast_cost

        lunch_attempted = False
        for index, attraction in enumerate(attractions):
            if not lunch_attempted and index == lunch_insertion_index:
                lunch_start = max(current_minutes, lunch_window_minutes)
                lunch_anchor = previous_stop or attraction
                (
                    current_minutes,
                    previous_stop,
                    lunch_cost,
                    _,
                ) = self._append_meal_slot(
                    session=session,
                    day_number=day_number,
                    meal_type="lunch",
                    current_minutes=lunch_start,
                    previous_stop=previous_stop,
                    anchor=lunch_anchor,
                    preferences=preferences,
                    used_restaurant_ids=used_restaurant_ids,
                    selected_restaurant_ids=selected_restaurant_ids,
                    meal_budget=meal_budget,
                    slots=slots,
                    warnings=warnings,
                )
                total_cost += lunch_cost
                lunch_attempted = True

            current_minutes, previous_stop, attraction_cost = (
                self._append_attraction_slot(
                    session=session,
                    day_number=day_number,
                    attraction=attraction,
                    current_minutes=current_minutes,
                    previous_stop=previous_stop,
                    segment_lookup=segment_lookup,
                    preferences=preferences,
                    slots=slots,
                    warnings=warnings,
                )
            )
            total_cost += attraction_cost

        if not lunch_attempted and attractions:
            lunch_start = max(current_minutes, lunch_window_minutes)
            (
                current_minutes,
                previous_stop,
                lunch_cost,
                _,
            ) = self._append_meal_slot(
                session=session,
                day_number=day_number,
                meal_type="lunch",
                current_minutes=lunch_start,
                previous_stop=previous_stop,
                anchor=previous_stop or attractions[-1],
                preferences=preferences,
                used_restaurant_ids=used_restaurant_ids,
                selected_restaurant_ids=selected_restaurant_ids,
                meal_budget=meal_budget,
                slots=slots,
                warnings=warnings,
            )
            total_cost += lunch_cost

        if lunch_insertion_index <= 0:
            warnings.append(
                f"Day {day_number}: lunch is scheduled before any attraction visits."
            )
        elif lunch_insertion_index >= len(attractions):
            warnings.append(
                f"Day {day_number}: lunch is scheduled after all attraction visits."
            )

        dinner_start = max(current_minutes, dinner_window_minutes)
        if attractions:
            (
                current_minutes,
                previous_stop,
                dinner_cost,
                _,
            ) = self._append_meal_slot(
                session=session,
                day_number=day_number,
                meal_type="dinner",
                current_minutes=dinner_start,
                previous_stop=previous_stop,
                anchor=hotel or previous_stop or attractions[-1],
                preferences=preferences,
                used_restaurant_ids=used_restaurant_ids,
                selected_restaurant_ids=selected_restaurant_ids,
                meal_budget=meal_budget,
                slots=slots,
                warnings=warnings,
            )
            total_cost += dinner_cost

        if hotel is not None:
            current_minutes, previous_stop = self._append_hotel_slot(
                session=session,
                slot_type="hotel_checkin",
                current_minutes=current_minutes,
                previous_stop=previous_stop,
                hotel=hotel,
                preferences=preferences,
                slots=slots,
                cost=hotel.price_per_night,
            )
            total_cost += hotel.price_per_night

        trip_used_restaurant_ids.update(selected_restaurant_ids)

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

    def _validate_day_constraints(self, day_plan: DayPlan) -> None:
        """
        Check daily range and walking time constraints, appending warnings.

        Args:
            day_plan: The day plan to validate.
        """

        max_range_m = self.settings.max_daily_range_km * 1000
        max_walking = self.settings.max_daily_walking_minutes

        locations: list[tuple[float, float]] = []
        total_walking_minutes = 0
        for slot in day_plan.slots:
            if slot.attraction is not None:
                locations.append((slot.attraction.latitude, slot.attraction.longitude))
            elif slot.restaurant is not None:
                locations.append((slot.restaurant.latitude, slot.restaurant.longitude))
            elif slot.hotel is not None:
                locations.append((slot.hotel.latitude, slot.hotel.longitude))
            if (
                slot.transport_from_previous is not None
                and slot.transport_from_previous.travel_mode == "walking"
            ):
                total_walking_minutes += max(
                    1, round(slot.transport_from_previous.duration_s / 60)
                )

        if len(locations) >= 2:
            max_dist = 0.0
            for i, loc_a in enumerate(locations):
                for loc_b in locations[i + 1 :]:
                    dist = self._haversine_m(loc_a[0], loc_a[1], loc_b[0], loc_b[1])
                    if dist > max_dist:
                        max_dist = dist
            if max_dist > max_range_m:
                day_plan.warnings.append(
                    f"Day {day_plan.day_number}: activity range is "
                    f"{max_dist / 1000:.1f}km, exceeding the "
                    f"{self.settings.max_daily_range_km:.0f}km limit."
                )

        if total_walking_minutes > max_walking:
            day_plan.warnings.append(
                f"Day {day_plan.day_number}: total walking time is "
                f"{total_walking_minutes} minutes, exceeding the "
                f"{max_walking}-minute limit."
            )

    def _find_lunch_insertion_index(
        self,
        attractions: list[Attraction],
        route: Route,
        start_minutes: int,
    ) -> int:
        """
        Find the best lunch insertion point based on time window and route gaps.

        Args:
            attractions: Ordered attractions for the day.
            route: Optimized route.
            start_minutes: Planned start time for attraction visits.

        Returns:
            The attraction index before which lunch should be inserted. A value
            equal to ``len(attractions)`` means lunch should happen after the
            final attraction.
        """

        if not attractions:
            return 0

        lunch_earliest = self.settings.lunch_window_start * 60
        lunch_latest = lunch_earliest + 150
        cursor = start_minutes
        best_index = max(1, len(attractions) // 2)
        best_duration = -1

        for index, attraction in enumerate(attractions):
            if index > 0 and index - 1 < len(route.segments):
                travel_minutes = max(
                    5,
                    round(route.segments[index - 1].duration_s / 60),
                )
                cursor += travel_minutes
                if lunch_earliest <= cursor <= lunch_latest:
                    segment_duration = route.segments[index - 1].duration_s
                    if segment_duration > best_duration:
                        best_duration = segment_duration
                        best_index = index

            cursor += attraction.visit_duration

        if best_duration >= 0:
            return best_index
        if cursor < lunch_earliest:
            return len(attractions)
        if start_minutes >= lunch_earliest:
            return 0
        return min(best_index, len(attractions))

    def _append_attraction_slot(
        self,
        session: Session,
        day_number: int,
        attraction: Attraction,
        current_minutes: int,
        previous_stop: Attraction | Restaurant | Hotel | None,
        segment_lookup: dict[tuple[str, str], TransportSegment],
        preferences: TravelPreferences,
        slots: list[TimeSlot],
        warnings: list[str],
    ) -> tuple[int, Attraction, float]:
        """
        Append an attraction visit to the timeline.

        Args:
            session: Active database session.
            day_number: Day number in the trip.
            attraction: Attraction to schedule.
            current_minutes: Current time cursor.
            previous_stop: Previous stop in the day.
            segment_lookup: Precomputed attraction-to-attraction segments.
            preferences: Structured travel preferences.
            slots: Mutable day slots list.
            warnings: Mutable warnings list.

        Returns:
            Updated time cursor, the attraction as previous stop, and added cost.
        """

        transport = None
        if previous_stop is not None:
            transport = segment_lookup.get((previous_stop.id, attraction.id))
            if transport is None:
                transport = self.maps_client.get_segment(
                    session,
                    previous_stop,
                    attraction,
                    preferences.travel_mode,
                )
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
                slot_type="attraction",
                start_time=self._format_minutes(start_minutes),
                end_time=self._format_minutes(end_minutes),
                title=attraction.name,
                description=attraction.description,
                attraction=attraction,
                transport_from_previous=transport,
                cost=attraction.cost,
            )
        )
        return end_minutes, attraction, attraction.cost

    def _append_meal_slot(
        self,
        session: Session,
        day_number: int,
        meal_type: str,
        current_minutes: int,
        previous_stop: Attraction | Restaurant | Hotel | None,
        anchor: Attraction | Restaurant | Hotel,
        preferences: TravelPreferences,
        used_restaurant_ids: set[str],
        selected_restaurant_ids: set[str],
        meal_budget: float | None,
        slots: list[TimeSlot],
        warnings: list[str],
    ) -> tuple[int, Attraction | Restaurant | Hotel | None, float, bool]:
        """
        Append a meal stop when a suitable restaurant is available.

        Args:
            session: Active database session.
            day_number: Day number in the trip.
            meal_type: Meal slot name.
            current_minutes: Current time cursor.
            previous_stop: Previous stop in the day.
            anchor: Location used to search for nearby restaurants.
            preferences: Structured travel preferences.
            used_restaurant_ids: Restaurants already used in this day.
            selected_restaurant_ids: Restaurants newly selected for the current day.
            meal_budget: Optional per-person meal budget.
            slots: Mutable day slots list.
            warnings: Mutable warnings list.

        Returns:
            Updated time cursor, new previous stop, added cost, and whether a meal
            slot was scheduled.
        """

        restaurant = self._select_restaurant(
            session=session,
            destination=preferences.destination,
            meal_type=meal_type,
            anchor=anchor,
            used_restaurant_ids=used_restaurant_ids,
            meal_budget=meal_budget,
        )
        if restaurant is None:
            warnings.append(
                f"Day {day_number}: no {meal_type} restaurant matched the current route."
            )
            return current_minutes, previous_stop, 0.0, False

        transport = None
        if previous_stop is not None:
            transport = self.maps_client.get_segment(
                session,
                previous_stop,
                restaurant,
                preferences.travel_mode,
            )
            current_minutes += max(5, round(transport.duration_s / 60))

        start_minutes = current_minutes
        end_minutes = start_minutes + restaurant.visit_duration
        slot_cost = round(restaurant.average_cost * preferences.travelers, 2)
        slots.append(
            TimeSlot(
                slot_type=meal_type,
                start_time=self._format_minutes(start_minutes),
                end_time=self._format_minutes(end_minutes),
                title=restaurant.name,
                description=restaurant.description,
                restaurant=restaurant,
                transport_from_previous=transport,
                cost=slot_cost,
            )
        )
        used_restaurant_ids.add(restaurant.id)
        selected_restaurant_ids.add(restaurant.id)
        return end_minutes, restaurant, slot_cost, True

    def _append_hotel_slot(
        self,
        session: Session,
        slot_type: str,
        current_minutes: int,
        previous_stop: Attraction | Restaurant | Hotel | None,
        hotel: Hotel,
        preferences: TravelPreferences,
        slots: list[TimeSlot],
        cost: float,
    ) -> tuple[int, Hotel]:
        """
        Append a hotel check-in or check-out slot to the day timeline.

        Args:
            session: Active database session.
            slot_type: Hotel slot type.
            current_minutes: Current time cursor.
            previous_stop: Previous stop in the day.
            hotel: Selected hotel.
            preferences: Structured travel preferences.
            slots: Mutable day slots list.
            cost: Slot cost to display.

        Returns:
            Updated time cursor and the hotel as previous stop.
        """

        transport = None
        if previous_stop is not None and previous_stop.id != hotel.id:
            transport = self.maps_client.get_segment(
                session,
                previous_stop,
                hotel,
                preferences.travel_mode,
            )
            current_minutes += max(5, round(transport.duration_s / 60))

        start_minutes = current_minutes
        end_minutes = start_minutes + 15
        title = hotel.name
        description = (
            f"Check out from {hotel.name} and begin the day."
            if slot_type == "hotel_checkout"
            else f"Return to {hotel.name} for the night."
        )
        slots.append(
            TimeSlot(
                slot_type=slot_type,
                start_time=self._format_minutes(start_minutes),
                end_time=self._format_minutes(end_minutes),
                title=title,
                description=description,
                hotel=hotel,
                transport_from_previous=transport,
                cost=cost,
            )
        )
        return end_minutes, hotel

    def _select_restaurant(
        self,
        session: Session,
        destination: str,
        meal_type: str,
        anchor: Attraction | Restaurant | Hotel,
        used_restaurant_ids: set[str],
        meal_budget: float | None,
    ) -> Restaurant | None:
        """
        Select a nearby restaurant for a specific meal slot.

        Args:
            session: Active database session.
            destination: Destination name.
            meal_type: Meal slot name.
            anchor: Location used for proximity ranking.
            used_restaurant_ids: Restaurants already used in this day.
            meal_budget: Optional per-person meal budget.

        Returns:
            The best matching restaurant or ``None``.
        """

        budgets: list[float | None] = (
            [meal_budget] if meal_budget is not None else [None]
        )
        if meal_budget is not None:
            budgets.append(None)
        for budget in budgets:
            candidates = self.data_retrieval.search_restaurants(
                session=session,
                destination=destination,
                meal_type=meal_type,
                near_lat=anchor.latitude,
                near_lon=anchor.longitude,
                max_cost=budget,
                limit=6,
            )
            for candidate in candidates:
                if candidate.id not in used_restaurant_ids:
                    return candidate
        return None

    def _max_meal_cost_per_person(
        self,
        preferences: TravelPreferences,
    ) -> float | None:
        """
        Estimate a per-person meal budget.

        Args:
            preferences: Structured travel preferences.

        Returns:
            Estimated per-person meal budget or ``None``.
        """

        if preferences.budget_total <= 0:
            return None
        denominator = max(1, preferences.trip_days * 3 * preferences.travelers)
        return max(15.0, preferences.budget_total * 0.2 / denominator)

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

    def _cluster_radius_m(self, attractions: list[Attraction]) -> float:
        """
        Compute the maximum pairwise distance within a cluster.

        Args:
            attractions: Attractions in the cluster.

        Returns:
            Maximum pairwise distance in meters.
        """

        if len(attractions) <= 1:
            return 0.0
        max_dist = 0.0
        for i, a in enumerate(attractions):
            for b in attractions[i + 1 :]:
                dist = self._haversine_m(
                    a.latitude, a.longitude, b.latitude, b.longitude
                )
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Haversine distance in meters between two geographic points.

        Args:
            lat1: Origin latitude.
            lon1: Origin longitude.
            lat2: Destination latitude.
            lon2: Destination longitude.

        Returns:
            Distance in meters.
        """

        earth_radius_m = 6_371_000
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        return 2 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))
