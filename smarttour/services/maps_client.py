"""
Travel-distance calculations with Google Maps API and Haversine fallback.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlmodel import Session, select

from smarttour.config import get_settings
from smarttour.models import DistanceCacheRecord, Route, TransportSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RouteGraph:
    """
    In-memory travel graph for one set of attractions.
    """

    duration_matrix: list[list[int]]
    segment_lookup: dict[tuple[int, int], TransportSegment]


class RouteLocation(Protocol):
    """
    Protocol for any stop that can be routed.
    """

    id: str
    name: str
    latitude: float
    longitude: float


class RouteUnavailableError(RuntimeError):
    """
    Raised when Google returns no route for a requested mode.
    """


MODE_MAPPING: dict[str, str] = {
    "walking": "walking",
    "transit": "transit",
    "driving": "driving",
}


class MapsClient:
    """
    Compute distance and travel time estimates.

    Uses Google Maps APIs when configured and falls back to Haversine estimates.
    """

    _MAX_MATRIX_ELEMENTS: int = 100

    haversine_speeds_mps: dict[str, float] = {
        "walking": 1.4,
        "transit": 6.5,
        "driving": 11.0,
    }

    def __init__(self) -> None:
        """
        Initialize the maps client with optional Google Maps API access.
        """

        settings = get_settings()
        self.api_key = settings.google_maps_api_key
        self._gmaps = None
        if self.api_key:
            try:
                import googlemaps

                self._gmaps = googlemaps.Client(key=self.api_key)
            except Exception:
                logger.exception("Failed to initialize Google Maps client")

    @property
    def google_maps_enabled(self) -> bool:
        """
        Return whether Google Maps API access is available.

        Returns:
            ``True`` when the Google Maps client is initialized.
        """

        return self._gmaps is not None

    def build_route_graph(
        self,
        session: Session,
        attractions: Sequence[RouteLocation],
        travel_mode: str,
    ) -> RouteGraph:
        """
        Build a route graph for a list of attractions.

        Args:
            session: Active database session.
            attractions: Attractions for one route.
            travel_mode: Transport mode.

        Returns:
            Duration matrix and detailed segment lookup.
        """

        logger.info(
            "Distance matrix: %d origins × %d destinations, mode=%s",
            len(attractions),
            len(attractions),
            travel_mode,
        )
        if travel_mode == "walking_transit":
            return self._build_mixed_route_graph(session, attractions)
        return self._build_single_mode_route_graph(
            session=session,
            attractions=attractions,
            travel_mode=travel_mode,
            allow_estimated_fallback=True,
        )

    def get_segment(
        self,
        session: Session,
        origin: RouteLocation,
        destination: RouteLocation,
        travel_mode: str,
    ) -> TransportSegment:
        """
        Return a cached, API-fetched, or Haversine-computed transport segment.

        Args:
            session: Active database session.
            origin: Start location.
            destination: End location.
            travel_mode: Requested transport mode.

        Returns:
            Transport segment data.
        """

        if origin.id == destination.id:
            return TransportSegment(
                origin_id=origin.id,
                destination_id=destination.id,
                travel_mode="walking",
                distance_m=0,
                duration_s=0,
                navigation_hint=f"Already at {destination.name}.",
            )

        segment: TransportSegment | None
        if travel_mode == "walking_transit":
            segment = self._select_best_mixed_segment(session, origin, destination)
        else:
            segment = self._get_or_create_segment(
                session=session,
                origin=origin,
                destination=destination,
                travel_mode=travel_mode,
                allow_estimated_fallback=True,
            )
        assert segment is not None
        return self._ensure_polyline(session, origin, destination, segment)

    def populate_route_polylines(
        self,
        session: Session,
        route: Route,
        locations: Sequence[RouteLocation],
    ) -> Route:
        """
        Populate polylines for final route segments only.

        Args:
            session: Active database session.
            route: Optimized route.
            locations: Locations involved in the route.

        Returns:
            Route with polyline-enriched segments.
        """

        location_lookup = {location.id: location for location in locations}
        updated_segments: list[TransportSegment] = []
        for segment in route.segments:
            origin = location_lookup.get(segment.origin_id)
            destination = location_lookup.get(segment.destination_id)
            if origin is None or destination is None:
                updated_segments.append(segment)
                continue
            updated_segments.append(
                self._ensure_polyline(session, origin, destination, segment)
            )
        return route.model_copy(update={"segments": updated_segments})

    def _build_mixed_route_graph(
        self,
        session: Session,
        attractions: Sequence[RouteLocation],
    ) -> RouteGraph:
        """
        Build a graph that chooses the faster of walking and transit per leg.

        Args:
            session: Active database session.
            attractions: Candidate route locations.

        Returns:
            A route graph with per-leg best modes.
        """

        walking_graph = self._build_single_mode_route_graph(
            session=session,
            attractions=attractions,
            travel_mode="walking",
            allow_estimated_fallback=True,
        )
        transit_graph = self._build_single_mode_route_graph(
            session=session,
            attractions=attractions,
            travel_mode="transit",
            allow_estimated_fallback=not self.google_maps_enabled,
        )

        duration_matrix: list[list[int]] = []
        segment_lookup: dict[tuple[int, int], TransportSegment] = {}
        for origin_index in range(len(attractions)):
            row: list[int] = []
            for destination_index in range(len(attractions)):
                walking_segment = walking_graph.segment_lookup[
                    (origin_index, destination_index)
                ]
                transit_segment = transit_graph.segment_lookup.get(
                    (origin_index, destination_index)
                )
                best_segment = walking_segment
                if transit_segment is not None and (
                    walking_segment.duration_s <= 0
                    or transit_segment.duration_s < walking_segment.duration_s
                ):
                    best_segment = transit_segment
                row.append(best_segment.duration_s)
                segment_lookup[(origin_index, destination_index)] = best_segment
            duration_matrix.append(row)

        return RouteGraph(
            duration_matrix=duration_matrix,
            segment_lookup=segment_lookup,
        )

    def _build_single_mode_route_graph(
        self,
        session: Session,
        attractions: Sequence[RouteLocation],
        travel_mode: str,
        allow_estimated_fallback: bool,
    ) -> RouteGraph:
        """
        Build a graph for one concrete mode.

        Args:
            session: Active database session.
            attractions: Candidate route locations.
            travel_mode: Concrete transport mode.
            allow_estimated_fallback: Whether missing API routes can fall back
                to Haversine estimation.

        Returns:
            Duration matrix and detailed segment lookup.
        """

        duration_matrix: list[list[int]] = []
        segment_lookup: dict[tuple[int, int], TransportSegment] = {}
        uncached_pairs: list[tuple[int, int]] = []

        for origin_index, origin in enumerate(attractions):
            row: list[int] = []
            for destination_index, destination in enumerate(attractions):
                if origin.id == destination.id:
                    segment = TransportSegment(
                        origin_id=origin.id,
                        destination_id=destination.id,
                        travel_mode=travel_mode,
                        distance_m=0,
                        duration_s=0,
                        navigation_hint=f"Already at {destination.name}.",
                    )
                    row.append(0)
                    segment_lookup[(origin_index, destination_index)] = segment
                    continue

                cached = self._get_cached(
                    session,
                    origin.id,
                    destination.id,
                    travel_mode,
                )
                if cached is not None:
                    cached_with_hint = self._apply_navigation_hint(
                        cached,
                        origin,
                        destination,
                    )
                    row.append(cached_with_hint.duration_s)
                    segment_lookup[(origin_index, destination_index)] = cached_with_hint
                else:
                    row.append(-1)
                    uncached_pairs.append((origin_index, destination_index))
            duration_matrix.append(row)

        if uncached_pairs and self.google_maps_enabled:
            self._fill_from_google_maps(
                session=session,
                attractions=attractions,
                travel_mode=travel_mode,
                duration_matrix=duration_matrix,
                segment_lookup=segment_lookup,
                uncached_pairs=uncached_pairs,
            )

        if allow_estimated_fallback or not self.google_maps_enabled:
            for origin_index, destination_index in uncached_pairs:
                if duration_matrix[origin_index][destination_index] != -1:
                    continue
                origin = attractions[origin_index]
                destination = attractions[destination_index]
                segment = self._haversine_segment(origin, destination, travel_mode)
                segment = self._apply_navigation_hint(segment, origin, destination)
                duration_matrix[origin_index][destination_index] = segment.duration_s
                segment_lookup[(origin_index, destination_index)] = segment
                session.merge(DistanceCacheRecord.from_model(segment))

        if uncached_pairs or session.new or session.dirty:
            session.commit()

        return RouteGraph(
            duration_matrix=duration_matrix,
            segment_lookup=segment_lookup,
        )

    def _select_best_mixed_segment(
        self,
        session: Session,
        origin: RouteLocation,
        destination: RouteLocation,
    ) -> TransportSegment:
        """
        Compare walking and transit for one final segment.

        Args:
            session: Active database session.
            origin: Start location.
            destination: End location.

        Returns:
            The faster available segment.
        """

        walking_segment = self._get_or_create_segment(
            session=session,
            origin=origin,
            destination=destination,
            travel_mode="walking",
            allow_estimated_fallback=True,
        )
        assert walking_segment is not None

        transit_segment = self._get_or_create_segment(
            session=session,
            origin=origin,
            destination=destination,
            travel_mode="transit",
            allow_estimated_fallback=not self.google_maps_enabled,
        )
        if (
            transit_segment is not None
            and transit_segment.duration_s < walking_segment.duration_s
        ):
            return transit_segment
        return walking_segment

    def _get_or_create_segment(
        self,
        session: Session,
        origin: RouteLocation,
        destination: RouteLocation,
        travel_mode: str,
        allow_estimated_fallback: bool,
    ) -> TransportSegment | None:
        """
        Get or build a concrete-mode segment.

        Args:
            session: Active database session.
            origin: Start location.
            destination: End location.
            travel_mode: Concrete transport mode.
            allow_estimated_fallback: Whether Haversine fallback is allowed.

        Returns:
            A transport segment or ``None`` when no route is available.
        """

        cached = self._get_cached(session, origin.id, destination.id, travel_mode)
        if cached is not None:
            return self._apply_navigation_hint(cached, origin, destination)

        if self.google_maps_enabled:
            try:
                segment = self._fetch_from_google_maps(origin, destination, travel_mode)
                segment = self._apply_navigation_hint(segment, origin, destination)
                session.merge(DistanceCacheRecord.from_model(segment))
                session.commit()
                return segment
            except RouteUnavailableError:
                if not allow_estimated_fallback:
                    return None
            except Exception:
                logger.exception(
                    "Google Maps API failed for %s -> %s via %s",
                    origin.name,
                    destination.name,
                    travel_mode,
                )
                if not allow_estimated_fallback:
                    return None

        if not allow_estimated_fallback and self.google_maps_enabled:
            return None

        segment = self._haversine_segment(origin, destination, travel_mode)
        segment = self._apply_navigation_hint(segment, origin, destination)
        session.merge(DistanceCacheRecord.from_model(segment))
        session.commit()
        return segment

    def _get_cached(
        self,
        session: Session,
        origin_id: str,
        dest_id: str,
        travel_mode: str,
    ) -> TransportSegment | None:
        """
        Look up a cached distance record.

        Args:
            session: Active database session.
            origin_id: Origin ID.
            dest_id: Destination ID.
            travel_mode: Concrete transport mode.

        Returns:
            Cached segment or ``None``.
        """

        statement = select(DistanceCacheRecord).where(
            DistanceCacheRecord.origin_id == origin_id,
            DistanceCacheRecord.dest_id == dest_id,
            DistanceCacheRecord.travel_mode == travel_mode,
        )
        cached = session.exec(statement).first()
        if cached is not None:
            return cached.to_model()
        return None

    def _fill_from_google_maps(
        self,
        session: Session,
        attractions: Sequence[RouteLocation],
        travel_mode: str,
        duration_matrix: list[list[int]],
        segment_lookup: dict[tuple[int, int], TransportSegment],
        uncached_pairs: list[tuple[int, int]],
    ) -> None:
        """
        Batch-fill uncached pairs using Google Maps Distance Matrix API.

        The API enforces a per-request element limit (origins x destinations
        must not exceed ``_MAX_MATRIX_ELEMENTS``).  This method partitions the
        request into compliant batches automatically.

        Args:
            session: Active database session.
            attractions: List of route locations.
            travel_mode: Concrete transport mode.
            duration_matrix: Mutable duration matrix to fill.
            segment_lookup: Mutable segment lookup to fill.
            uncached_pairs: List of (origin_index, dest_index) pairs needing data.
        """

        origin_indices = sorted({pair[0] for pair in uncached_pairs})
        dest_indices = sorted({pair[1] for pair in uncached_pairs})

        filled_pairs: set[tuple[int, int]] = set()
        for origin_batch, dest_batch in self._chunk_matrix_indices(
            origin_indices, dest_indices
        ):
            self._fill_matrix_batch(
                session=session,
                attractions=attractions,
                travel_mode=travel_mode,
                duration_matrix=duration_matrix,
                segment_lookup=segment_lookup,
                uncached_pairs=uncached_pairs,
                origin_batch=origin_batch,
                dest_batch=dest_batch,
                filled_pairs=filled_pairs,
            )

        remaining_pairs = [pair for pair in uncached_pairs if pair not in filled_pairs]
        uncached_pairs.clear()
        uncached_pairs.extend(remaining_pairs)

    def _chunk_matrix_indices(
        self,
        origin_indices: list[int],
        dest_indices: list[int],
    ) -> list[tuple[list[int], list[int]]]:
        """
        Partition origin and destination indices into batches that respect
        the Distance Matrix API element limit.

        Args:
            origin_indices: Sorted unique origin indices.
            dest_indices: Sorted unique destination indices.

        Returns:
            List of (origin_batch, dest_batch) tuples.
        """

        max_dest_per_batch = min(len(dest_indices), self._MAX_MATRIX_ELEMENTS)
        batches: list[tuple[list[int], list[int]]] = []
        for d_start in range(0, len(dest_indices), max_dest_per_batch):
            d_batch = dest_indices[d_start : d_start + max_dest_per_batch]
            max_origins = max(1, self._MAX_MATRIX_ELEMENTS // max(1, len(d_batch)))
            for o_start in range(0, len(origin_indices), max_origins):
                o_batch = origin_indices[o_start : o_start + max_origins]
                batches.append((o_batch, d_batch))
        return batches

    def _fill_matrix_batch(
        self,
        session: Session,
        attractions: Sequence[RouteLocation],
        travel_mode: str,
        duration_matrix: list[list[int]],
        segment_lookup: dict[tuple[int, int], TransportSegment],
        uncached_pairs: list[tuple[int, int]],
        origin_batch: list[int],
        dest_batch: list[int],
        filled_pairs: set[tuple[int, int]],
    ) -> None:
        """
        Execute one Distance Matrix API call for a batch of origins and destinations.

        Args:
            session: Active database session.
            attractions: List of route locations.
            travel_mode: Concrete transport mode.
            duration_matrix: Mutable duration matrix to fill.
            segment_lookup: Mutable segment lookup to fill.
            uncached_pairs: Full list of uncached pairs.
            origin_batch: Origin indices for this batch.
            dest_batch: Destination indices for this batch.
            filled_pairs: Accumulator for successfully filled pairs.
        """

        origins = [
            f"{attractions[index].latitude},{attractions[index].longitude}"
            for index in origin_batch
        ]
        destinations = [
            f"{attractions[index].latitude},{attractions[index].longitude}"
            for index in dest_batch
        ]

        google_mode = MODE_MAPPING.get(travel_mode, "walking")
        distance_matrix_kwargs: dict[str, object] = {
            "origins": origins,
            "destinations": destinations,
            "mode": google_mode,
        }
        if google_mode == "transit":
            distance_matrix_kwargs["departure_time"] = datetime.now(UTC)

        assert self._gmaps is not None
        try:
            result = self._gmaps.distance_matrix(**distance_matrix_kwargs)
        except Exception:
            logger.exception(
                "Google Maps Distance Matrix API call failed for mode %s",
                travel_mode,
            )
            return

        origin_idx_map = {idx: pos for pos, idx in enumerate(origin_batch)}
        dest_idx_map = {idx: pos for pos, idx in enumerate(dest_batch)}

        for origin_index, destination_index in uncached_pairs:
            if (
                origin_index not in origin_idx_map
                or destination_index not in dest_idx_map
            ):
                continue

            row_pos = origin_idx_map[origin_index]
            col_pos = dest_idx_map[destination_index]
            try:
                element = result["rows"][row_pos]["elements"][col_pos]
            except (KeyError, IndexError):
                logger.warning(
                    "Missing Distance Matrix element for pair (%d, %d)",
                    origin_index,
                    destination_index,
                )
                continue

            if element.get("status") != "OK":
                continue

            origin = attractions[origin_index]
            destination = attractions[destination_index]
            segment = TransportSegment(
                origin_id=origin.id,
                destination_id=destination.id,
                travel_mode=travel_mode,
                distance_m=int(element["distance"]["value"]),
                duration_s=int(element["duration"]["value"]),
                polyline=None,
            )
            segment = self._apply_navigation_hint(segment, origin, destination)
            duration_matrix[origin_index][destination_index] = segment.duration_s
            segment_lookup[(origin_index, destination_index)] = segment
            session.merge(DistanceCacheRecord.from_model(segment))
            filled_pairs.add((origin_index, destination_index))

    def _fetch_from_google_maps(
        self,
        origin: RouteLocation,
        destination: RouteLocation,
        travel_mode: str,
    ) -> TransportSegment:
        """
        Fetch a single origin-destination pair from Google Maps.

        Args:
            origin: Start location.
            destination: End location.
            travel_mode: Concrete transport mode.

        Returns:
            A transport segment with real distance and duration data.

        Raises:
            RouteUnavailableError: When the API returns no usable route.
        """

        google_mode = MODE_MAPPING.get(travel_mode, "walking")
        distance_matrix_kwargs: dict[str, object] = {
            "origins": [f"{origin.latitude},{origin.longitude}"],
            "destinations": [f"{destination.latitude},{destination.longitude}"],
            "mode": google_mode,
        }
        if google_mode == "transit":
            distance_matrix_kwargs["departure_time"] = datetime.now(UTC)

        assert self._gmaps is not None
        logger.debug(
            "Directions API: %s -> %s, mode=%s",
            origin.name,
            destination.name,
            travel_mode,
        )
        result = self._gmaps.distance_matrix(**distance_matrix_kwargs)
        element = result["rows"][0]["elements"][0]
        if element.get("status") != "OK":
            raise RouteUnavailableError(
                f"Google Maps returned status '{element.get('status')}' "
                f"for {origin.name} -> {destination.name}"
            )
        return TransportSegment(
            origin_id=origin.id,
            destination_id=destination.id,
            travel_mode=travel_mode,
            distance_m=int(element["distance"]["value"]),
            duration_s=int(element["duration"]["value"]),
            polyline=None,
        )

    def _ensure_polyline(
        self,
        session: Session,
        origin: RouteLocation,
        destination: RouteLocation,
        segment: TransportSegment,
    ) -> TransportSegment:
        """
        Populate a segment polyline when Google Directions is available.

        Args:
            session: Active database session.
            origin: Start location.
            destination: End location.
            segment: Existing segment data.

        Returns:
            The segment with polyline when available.
        """

        if segment.polyline is not None or not self.google_maps_enabled:
            return self._apply_navigation_hint(segment, origin, destination)

        polyline = self._fetch_directions_polyline(
            origin=origin,
            destination=destination,
            travel_mode=segment.travel_mode,
        )
        if polyline is None:
            return self._apply_navigation_hint(segment, origin, destination)

        updated_segment = segment.model_copy(update={"polyline": polyline})
        updated_segment = self._apply_navigation_hint(
            updated_segment,
            origin,
            destination,
        )
        session.merge(DistanceCacheRecord.from_model(updated_segment))
        session.commit()
        return updated_segment

    def _fetch_directions_polyline(
        self,
        origin: RouteLocation,
        destination: RouteLocation,
        travel_mode: str,
    ) -> str | None:
        """
        Fetch an overview polyline for a final route segment.

        Args:
            origin: Start location.
            destination: End location.
            travel_mode: Concrete transport mode.

        Returns:
            Encoded overview polyline or ``None`` when unavailable.
        """

        google_mode = MODE_MAPPING.get(travel_mode)
        if google_mode is None or not self.google_maps_enabled:
            return None

        directions_kwargs: dict[str, object] = {
            "origin": f"{origin.latitude},{origin.longitude}",
            "destination": f"{destination.latitude},{destination.longitude}",
            "mode": google_mode,
            "units": "metric",
        }
        if google_mode == "transit":
            directions_kwargs["departure_time"] = datetime.now(UTC)

        assert self._gmaps is not None
        try:
            logger.debug(
                "Directions API: %s -> %s, mode=%s",
                origin.name,
                destination.name,
                travel_mode,
            )
            routes = self._gmaps.directions(**directions_kwargs)
        except Exception:
            logger.exception(
                "Google Maps Directions API failed for %s -> %s via %s",
                origin.name,
                destination.name,
                travel_mode,
            )
            return None

        if not routes:
            return None
        overview_polyline = routes[0].get("overview_polyline", {})
        if not isinstance(overview_polyline, dict):
            return None
        points = overview_polyline.get("points")
        if isinstance(points, str) and points:
            return points
        return None

    def _haversine_segment(
        self,
        origin: RouteLocation,
        destination: RouteLocation,
        travel_mode: str,
    ) -> TransportSegment:
        """
        Compute a transport segment using Haversine distance estimation.

        Args:
            origin: Start location.
            destination: End location.
            travel_mode: Concrete transport mode.

        Returns:
            An estimated transport segment.
        """

        distance_m = int(
            self._haversine_meters(
                origin.latitude,
                origin.longitude,
                destination.latitude,
                destination.longitude,
            )
        )
        speed = self.haversine_speeds_mps.get(
            travel_mode,
            self.haversine_speeds_mps["walking"],
        )
        duration_s = max(300, int(distance_m / speed))
        return TransportSegment(
            origin_id=origin.id,
            destination_id=destination.id,
            travel_mode=travel_mode,
            distance_m=distance_m,
            duration_s=duration_s,
            polyline=None,
        )

    def _apply_navigation_hint(
        self,
        segment: TransportSegment,
        origin: RouteLocation,
        destination: RouteLocation,
    ) -> TransportSegment:
        """
        Return a segment enriched with a deterministic navigation hint.

        Args:
            segment: Raw transport segment.
            origin: Start point.
            destination: End point.

        Returns:
            A transport segment with a user-facing navigation hint.
        """

        navigation_hint = self._build_navigation_hint(
            origin=origin,
            destination=destination,
            travel_mode=segment.travel_mode,
            distance_m=segment.distance_m,
            duration_s=segment.duration_s,
        )
        return segment.model_copy(update={"navigation_hint": navigation_hint})

    def _build_navigation_hint(
        self,
        origin: RouteLocation,
        destination: RouteLocation,
        travel_mode: str,
        distance_m: int,
        duration_s: int,
    ) -> str:
        """
        Build a concise navigation summary for a segment.

        Args:
            origin: Start point.
            destination: End point.
            travel_mode: Selected travel mode.
            distance_m: Segment distance in meters.
            duration_s: Segment duration in seconds.

        Returns:
            A navigation hint for display in the itinerary.
        """

        if distance_m <= 0 or duration_s <= 0:
            return f"Already at {destination.name}."
        direction = self._bearing_direction(
            origin.latitude,
            origin.longitude,
            destination.latitude,
            destination.longitude,
        )
        duration_minutes = max(1, round(duration_s / 60))
        formatted_distance = self._format_distance(distance_m)
        if travel_mode == "transit":
            return (
                f"Take public transport for about {duration_minutes} min "
                f"({formatted_distance}) {direction} to {destination.name}."
            )
        if travel_mode == "driving":
            return (
                f"Drive about {duration_minutes} min ({formatted_distance}) "
                f"{direction} to {destination.name}."
            )
        return (
            f"Walk about {duration_minutes} min ({formatted_distance}) "
            f"{direction} to {destination.name}."
        )

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

    def _bearing_direction(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
    ) -> str:
        """
        Convert two coordinates into a coarse compass direction.

        Args:
            origin_lat: Origin latitude.
            origin_lon: Origin longitude.
            destination_lat: Destination latitude.
            destination_lon: Destination longitude.

        Returns:
            A short compass direction.
        """

        latitude_1 = math.radians(origin_lat)
        latitude_2 = math.radians(destination_lat)
        longitude_delta = math.radians(destination_lon - origin_lon)
        y_component = math.sin(longitude_delta) * math.cos(latitude_2)
        x_component = math.cos(latitude_1) * math.sin(latitude_2) - math.sin(
            latitude_1
        ) * math.cos(latitude_2) * math.cos(longitude_delta)
        bearing = (math.degrees(math.atan2(y_component, x_component)) + 360) % 360
        directions = [
            "north",
            "north-east",
            "east",
            "south-east",
            "south",
            "south-west",
            "west",
            "north-west",
        ]
        index = round(bearing / 45) % len(directions)
        return directions[index]

    def _format_distance(self, distance_m: int) -> str:
        """
        Format a distance for itinerary presentation.

        Args:
            distance_m: Distance in meters.

        Returns:
            A readable distance string.
        """

        if distance_m < 1000:
            return f"{distance_m} m"
        return f"{distance_m / 1000:.1f} km"
