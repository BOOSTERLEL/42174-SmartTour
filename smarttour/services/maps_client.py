"""
Travel-distance calculations with Google Maps API and Haversine fallback.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from sqlmodel import Session, select

from smarttour.config import get_settings
from smarttour.models import Attraction, DistanceCacheRecord, TransportSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RouteGraph:
    """
    In-memory travel graph for one set of attractions.
    """

    duration_matrix: list[list[int]]
    segment_lookup: dict[tuple[int, int], TransportSegment]


MODE_MAPPING: dict[str, str] = {
    "walking": "walking",
    "transit": "transit",
    "driving": "driving",
}


class MapsClient:
    """
    Compute distance and travel time estimates.

    Uses Google Maps Distance Matrix API when configured,
    falls back to Haversine straight-line estimates.
    """

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
        attractions: list[Attraction],
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
                    )
                    row.append(0)
                    segment_lookup[(origin_index, destination_index)] = segment
                    continue

                cached = self._get_cached(
                    session, origin.id, destination.id, travel_mode
                )
                if cached is not None:
                    row.append(cached.duration_s)
                    segment_lookup[(origin_index, destination_index)] = cached
                else:
                    row.append(-1)
                    uncached_pairs.append((origin_index, destination_index))
            duration_matrix.append(row)

        if uncached_pairs and self.google_maps_enabled:
            self._fill_from_google_maps(
                session,
                attractions,
                travel_mode,
                duration_matrix,
                segment_lookup,
                uncached_pairs,
            )

        for origin_index, destination_index in uncached_pairs:
            if duration_matrix[origin_index][destination_index] == -1:
                origin = attractions[origin_index]
                destination = attractions[destination_index]
                segment = self._haversine_segment(origin, destination, travel_mode)
                duration_matrix[origin_index][destination_index] = segment.duration_s
                segment_lookup[(origin_index, destination_index)] = segment
                session.add(DistanceCacheRecord.from_model(segment))

        if uncached_pairs:
            session.commit()

        return RouteGraph(
            duration_matrix=duration_matrix, segment_lookup=segment_lookup
        )

    def get_segment(
        self,
        session: Session,
        origin: Attraction,
        destination: Attraction,
        travel_mode: str,
    ) -> TransportSegment:
        """
        Return a cached, API-fetched, or Haversine-computed transport segment.

        Args:
            session: Active database session.
            origin: Start attraction.
            destination: End attraction.
            travel_mode: Transport mode.

        Returns:
            Transport segment data.
        """

        if origin.id == destination.id:
            return TransportSegment(
                origin_id=origin.id,
                destination_id=destination.id,
                travel_mode=travel_mode,
                distance_m=0,
                duration_s=0,
            )

        cached = self._get_cached(session, origin.id, destination.id, travel_mode)
        if cached is not None:
            return cached

        if self.google_maps_enabled:
            try:
                segment = self._fetch_from_google_maps(origin, destination, travel_mode)
                session.add(DistanceCacheRecord.from_model(segment))
                session.commit()
                return segment
            except Exception:
                logger.exception(
                    "Google Maps API failed for %s -> %s", origin.name, destination.name
                )

        segment = self._haversine_segment(origin, destination, travel_mode)
        session.add(DistanceCacheRecord.from_model(segment))
        session.commit()
        return segment

    def _get_cached(
        self, session: Session, origin_id: str, dest_id: str, travel_mode: str
    ) -> TransportSegment | None:
        """
        Look up a cached distance record.

        Args:
            session: Active database session.
            origin_id: Origin attraction ID.
            dest_id: Destination attraction ID.
            travel_mode: Transport mode.

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
        attractions: list[Attraction],
        travel_mode: str,
        duration_matrix: list[list[int]],
        segment_lookup: dict[tuple[int, int], TransportSegment],
        uncached_pairs: list[tuple[int, int]],
    ) -> None:
        """
        Batch-fill uncached pairs using Google Maps Distance Matrix API.

        Args:
            session: Active database session.
            attractions: List of attractions.
            travel_mode: Transport mode.
            duration_matrix: Mutable duration matrix to fill.
            segment_lookup: Mutable segment lookup to fill.
            uncached_pairs: List of (origin_index, dest_index) pairs needing data.
        """

        origin_indices = sorted({pair[0] for pair in uncached_pairs})
        dest_indices = sorted({pair[1] for pair in uncached_pairs})

        origins = [
            f"{attractions[i].latitude},{attractions[i].longitude}"
            for i in origin_indices
        ]
        destinations = [
            f"{attractions[i].latitude},{attractions[i].longitude}"
            for i in dest_indices
        ]

        google_mode = MODE_MAPPING.get(travel_mode, "walking")

        assert self._gmaps is not None
        try:
            result = self._gmaps.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode=google_mode,
            )
        except Exception:
            logger.exception("Google Maps Distance Matrix API call failed")
            return

        origin_idx_map = {idx: pos for pos, idx in enumerate(origin_indices)}
        dest_idx_map = {idx: pos for pos, idx in enumerate(dest_indices)}

        filled = set()
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
                if element["status"] != "OK":
                    continue
                distance_m = element["distance"]["value"]
                duration_s = element["duration"]["value"]
                origin = attractions[origin_index]
                destination = attractions[destination_index]
                segment = TransportSegment(
                    origin_id=origin.id,
                    destination_id=destination.id,
                    travel_mode=travel_mode,
                    distance_m=distance_m,
                    duration_s=duration_s,
                    polyline=None,
                )
                duration_matrix[origin_index][destination_index] = duration_s
                segment_lookup[(origin_index, destination_index)] = segment
                session.add(DistanceCacheRecord.from_model(segment))
                filled.add((origin_index, destination_index))
            except (KeyError, IndexError):
                logger.warning(
                    "Missing Distance Matrix element for pair (%d, %d)",
                    origin_index,
                    destination_index,
                )

        remaining = [pair for pair in uncached_pairs if pair not in filled]
        uncached_pairs.clear()
        uncached_pairs.extend(remaining)

    def _fetch_from_google_maps(
        self, origin: Attraction, destination: Attraction, travel_mode: str
    ) -> TransportSegment:
        """
        Fetch a single origin-destination pair from Google Maps.

        Args:
            origin: Start attraction.
            destination: End attraction.
            travel_mode: Transport mode.

        Returns:
            A transport segment with real distance and duration data.

        Raises:
            RuntimeError: When the API returns no usable result.
        """

        google_mode = MODE_MAPPING.get(travel_mode, "walking")
        assert self._gmaps is not None
        result = self._gmaps.distance_matrix(
            origins=[f"{origin.latitude},{origin.longitude}"],
            destinations=[f"{destination.latitude},{destination.longitude}"],
            mode=google_mode,
        )
        element = result["rows"][0]["elements"][0]
        if element["status"] != "OK":
            raise RuntimeError(
                f"Google Maps returned status '{element['status']}' "
                f"for {origin.name} -> {destination.name}"
            )
        return TransportSegment(
            origin_id=origin.id,
            destination_id=destination.id,
            travel_mode=travel_mode,
            distance_m=element["distance"]["value"],
            duration_s=element["duration"]["value"],
            polyline=None,
        )

    def _haversine_segment(
        self, origin: Attraction, destination: Attraction, travel_mode: str
    ) -> TransportSegment:
        """
        Compute a transport segment using Haversine distance estimation.

        Args:
            origin: Start attraction.
            destination: End attraction.
            travel_mode: Transport mode.

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
            travel_mode, self.haversine_speeds_mps["walking"]
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
