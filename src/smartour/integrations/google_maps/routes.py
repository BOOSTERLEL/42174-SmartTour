"""Google Routes API client."""

from typing import Any

from smartour.integrations.google_maps.client import GoogleMapsHttpClient
from smartour.integrations.google_maps.field_masks import (
    ROUTE_MATRIX_SUMMARY_FIELD_MASK,
    ROUTES_SUMMARY_FIELD_MASK,
)

COMPUTE_ROUTE_MATRIX_URL = (
    "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
)
COMPUTE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"


class GoogleRoutesClient:
    """
    Client for Google Routes API requests.
    """

    def __init__(self, base_client: GoogleMapsHttpClient) -> None:
        """
        Initialize the Routes client.

        Args:
            base_client: The shared Google Maps HTTP client.
        """
        self.base_client = base_client

    async def compute_routes(
        self,
        origin_latitude: float,
        origin_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
        travel_mode: str = "DRIVE",
        routing_preference: str | None = "TRAFFIC_AWARE",
        field_mask: str = ROUTES_SUMMARY_FIELD_MASK,
    ) -> dict[str, Any]:
        """
        Compute route details between two coordinates.

        Args:
            origin_latitude: The origin latitude.
            origin_longitude: The origin longitude.
            destination_latitude: The destination latitude.
            destination_longitude: The destination longitude.
            travel_mode: The Google Routes travel mode.
            routing_preference: The optional routing preference.
            field_mask: The Google Routes response field mask.

        Returns:
            The Compute Routes response payload.
        """
        body: dict[str, Any] = {
            "origin": _lat_lng_waypoint(origin_latitude, origin_longitude),
            "destination": _lat_lng_waypoint(
                destination_latitude, destination_longitude
            ),
            "travelMode": travel_mode,
            "computeAlternativeRoutes": False,
            "units": "METRIC",
        }
        if routing_preference:
            body["routingPreference"] = routing_preference
        return await self.base_client.post_json(
            "routes", COMPUTE_ROUTES_URL, body, field_mask
        )

    async def compute_route_matrix(
        self,
        origins: list[tuple[float, float]],
        destinations: list[tuple[float, float]],
        travel_mode: str = "DRIVE",
        routing_preference: str | None = "TRAFFIC_AWARE",
        field_mask: str = ROUTE_MATRIX_SUMMARY_FIELD_MASK,
    ) -> dict[str, Any]:
        """
        Compute route matrix data for multiple origins and destinations.

        Args:
            origins: The origin latitude and longitude pairs.
            destinations: The destination latitude and longitude pairs.
            travel_mode: The Google Routes travel mode.
            routing_preference: The optional routing preference.
            field_mask: The Google Routes response field mask.

        Returns:
            The Compute Route Matrix response payload.
        """
        body: dict[str, Any] = {
            "origins": [
                {"waypoint": _lat_lng_waypoint(latitude, longitude)}
                for latitude, longitude in origins
            ],
            "destinations": [
                {"waypoint": _lat_lng_waypoint(latitude, longitude)}
                for latitude, longitude in destinations
            ],
            "travelMode": travel_mode,
        }
        if routing_preference:
            body["routingPreference"] = routing_preference
        return await self.base_client.post_json(
            "routes", COMPUTE_ROUTE_MATRIX_URL, body, field_mask
        )


def _lat_lng_waypoint(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Build a Google Routes waypoint from latitude and longitude.

    Args:
        latitude: The waypoint latitude.
        longitude: The waypoint longitude.

    Returns:
        The Google Routes waypoint object.
    """
    return {"location": {"latLng": {"latitude": latitude, "longitude": longitude}}}
