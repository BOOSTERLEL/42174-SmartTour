"""Google Geocoding API client."""

from typing import Any

from smartour.integrations.google_maps.client import (
    GoogleMapsApiError,
    GoogleMapsHttpClient,
)

GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"


class GoogleGeocodingClient:
    """
    Client for Google Geocoding API requests.
    """

    def __init__(self, base_client: GoogleMapsHttpClient) -> None:
        """
        Initialize the Geocoding client.

        Args:
            base_client: The shared Google Maps HTTP client.
        """
        self.base_client = base_client

    async def geocode(
        self,
        address: str,
        language: str | None = None,
        region: str | None = None,
    ) -> dict[str, Any]:
        """
        Convert an address or place text into geocoding results.

        Args:
            address: The address or supported place query to geocode.
            language: The optional response language.
            region: The optional region bias.

        Returns:
            The Geocoding API response payload.
        """
        params: dict[str, Any] = {"address": address}
        if language:
            params["language"] = language
        if region:
            params["region"] = region
        payload = await self.base_client.get_json("geocoding", GEOCODING_URL, params)
        status = payload.get("status")
        if status != "OK":
            message = str(
                payload.get("error_message") or status or "Geocoding request failed"
            )
            raise GoogleMapsApiError("geocoding", message)
        return payload
