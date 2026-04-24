"""Google Time Zone API client."""

from typing import Any

from smartour.integrations.google_maps.client import (
    GoogleMapsApiError,
    GoogleMapsHttpClient,
)

TIME_ZONE_URL = "https://maps.googleapis.com/maps/api/timezone/json"


class GoogleTimeZoneClient:
    """
    Client for Google Time Zone API requests.
    """

    def __init__(self, base_client: GoogleMapsHttpClient) -> None:
        """
        Initialize the Time Zone client.

        Args:
            base_client: The shared Google Maps HTTP client.
        """
        self.base_client = base_client

    async def get_time_zone(
        self, latitude: float, longitude: float, timestamp: int
    ) -> dict[str, Any]:
        """
        Fetch time zone data for a coordinate and timestamp.

        Args:
            latitude: The target latitude.
            longitude: The target longitude.
            timestamp: The Unix timestamp for the desired time.

        Returns:
            The Time Zone API response payload.
        """
        params = {"location": f"{latitude},{longitude}", "timestamp": timestamp}
        payload = await self.base_client.get_json("timezone", TIME_ZONE_URL, params)
        status = payload.get("status")
        if status != "OK":
            message = str(
                payload.get("errorMessage") or status or "Time Zone request failed"
            )
            raise GoogleMapsApiError("timezone", message)
        return payload
