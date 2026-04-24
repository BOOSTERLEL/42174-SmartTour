"""Shared Google Maps Platform HTTP client primitives."""

from dataclasses import dataclass
from typing import Any

import httpx

from smartour.core.errors import ExternalServiceError


class GoogleMapsApiError(ExternalServiceError):
    """
    Error raised when a Google Maps Platform API request fails.
    """


class GoogleMapsHttpClient:
    """
    Low-level async HTTP client for Google Maps Platform requests.
    """

    def __init__(self, api_key: str, http_client: httpx.AsyncClient) -> None:
        """
        Initialize the low-level Google Maps HTTP client.

        Args:
            api_key: The Google Maps API key.
            http_client: The async HTTP client used for transport.
        """
        self.api_key = api_key
        self.http_client = http_client

    async def get_json(
        self, service: str, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Send a Google Maps GET request and return JSON.

        Args:
            service: The logical Google Maps service name.
            url: The full Google Maps endpoint URL.
            params: The query parameters for the request.

        Returns:
            The decoded JSON response.
        """
        request_params = dict(params or {})
        request_params["key"] = self.api_key
        try:
            response = await self.http_client.get(url, params=request_params)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise self._api_error_from_response(service, error.response) from error
        except httpx.HTTPError as error:
            raise GoogleMapsApiError(service, str(error)) from error
        return self._json_response(service, response)

    async def post_json(
        self,
        service: str,
        url: str,
        body: dict[str, Any],
        field_mask: str,
    ) -> dict[str, Any]:
        """
        Send a Google Maps POST request and return JSON.

        Args:
            service: The logical Google Maps service name.
            url: The full Google Maps endpoint URL.
            body: The JSON request body.
            field_mask: The required Google response field mask.

        Returns:
            The decoded JSON response.
        """
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }
        try:
            response = await self.http_client.post(url, json=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise self._api_error_from_response(service, error.response) from error
        except httpx.HTTPError as error:
            raise GoogleMapsApiError(service, str(error)) from error
        return self._json_response(service, response)

    def _api_error_from_response(
        self, service: str, response: httpx.Response
    ) -> GoogleMapsApiError:
        """
        Convert an HTTP error response into a sanitized Google Maps error.

        Args:
            service: The logical Google Maps service name.
            response: The failed HTTP response.

        Returns:
            A normalized Google Maps API error.
        """
        message = response.text
        try:
            payload = response.json()
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = error["message"]
            elif isinstance(payload.get("error_message"), str):
                message = payload["error_message"]
        except ValueError:
            pass
        return GoogleMapsApiError(service, message, response.status_code)

    def _json_response(self, service: str, response: httpx.Response) -> dict[str, Any]:
        """
        Decode a successful HTTP response as JSON.

        Args:
            service: The logical Google Maps service name.
            response: The successful HTTP response.

        Returns:
            The decoded JSON object.
        """
        try:
            payload = response.json()
        except ValueError as error:
            raise GoogleMapsApiError(
                service, "Google Maps returned invalid JSON", response.status_code
            ) from error
        if not isinstance(payload, dict):
            raise GoogleMapsApiError(
                service,
                "Google Maps returned a non-object JSON payload",
                response.status_code,
            )
        return payload


@dataclass(frozen=True, slots=True)
class GoogleMapsClient:
    """
    Aggregated Google Maps API clients.
    """

    places: Any
    routes: Any
    geocoding: Any
    timezone: Any


def create_google_maps_client(
    api_key: str, http_client: httpx.AsyncClient
) -> GoogleMapsClient:
    """
    Create an aggregated Google Maps client group.

    Args:
        api_key: The Google Maps API key.
        http_client: The async HTTP client used for transport.

    Returns:
        The aggregated Google Maps client group.
    """
    from smartour.integrations.google_maps.geocoding import GoogleGeocodingClient
    from smartour.integrations.google_maps.places import GooglePlacesClient
    from smartour.integrations.google_maps.routes import GoogleRoutesClient
    from smartour.integrations.google_maps.timezone import GoogleTimeZoneClient

    base_client = GoogleMapsHttpClient(api_key, http_client)
    return GoogleMapsClient(
        places=GooglePlacesClient(base_client),
        routes=GoogleRoutesClient(base_client),
        geocoding=GoogleGeocodingClient(base_client),
        timezone=GoogleTimeZoneClient(base_client),
    )
