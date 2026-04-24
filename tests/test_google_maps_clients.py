"""Tests for Google Maps client request construction and response handling."""

import json
from typing import Any

import httpx
import pytest

from smartour.integrations.google_maps.client import (
    GoogleMapsApiError,
    GoogleMapsHttpClient,
)
from smartour.integrations.google_maps.field_masks import (
    PLACES_DISCOVERY_FIELD_MASK,
    ROUTES_SUMMARY_FIELD_MASK,
)
from smartour.integrations.google_maps.geocoding import GoogleGeocodingClient
from smartour.integrations.google_maps.places import GooglePlacesClient
from smartour.integrations.google_maps.routes import GoogleRoutesClient


@pytest.mark.asyncio
async def test_places_text_search_sends_required_headers() -> None:
    """
    Verify that Places Text Search sends the required API key and field mask headers.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        """
        Handle the mocked Places request.

        Args:
            request: The outgoing HTTP request.

        Returns:
            A mocked successful Places response.
        """
        assert str(request.url) == "https://places.googleapis.com/v1/places:searchText"
        assert request.headers["X-Goog-Api-Key"] == "test-key"
        assert request.headers["X-Goog-FieldMask"] == PLACES_DISCOVERY_FIELD_MASK
        payload = _json_body(request)
        assert payload["textQuery"] == "coffee in Sydney"
        assert payload["pageSize"] == 1
        return httpx.Response(200, json={"places": [{"id": "place-1"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GooglePlacesClient(GoogleMapsHttpClient("test-key", http_client))
        payload = await client.search_text("coffee in Sydney", page_size=1)

    assert payload["places"][0]["id"] == "place-1"


@pytest.mark.asyncio
async def test_routes_compute_routes_sends_field_mask() -> None:
    """
    Verify that Compute Routes sends a field mask and coordinate body.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        """
        Handle the mocked Routes request.

        Args:
            request: The outgoing HTTP request.

        Returns:
            A mocked successful Routes response.
        """
        assert (
            str(request.url)
            == "https://routes.googleapis.com/directions/v2:computeRoutes"
        )
        assert request.headers["X-Goog-Api-Key"] == "test-key"
        assert request.headers["X-Goog-FieldMask"] == ROUTES_SUMMARY_FIELD_MASK
        payload = _json_body(request)
        assert payload["origin"]["location"]["latLng"]["latitude"] == -33.8567844
        assert payload["destination"]["location"]["latLng"]["longitude"] == 151.20689
        return httpx.Response(200, json={"routes": [{"distanceMeters": 1000}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GoogleRoutesClient(GoogleMapsHttpClient("test-key", http_client))
        payload = await client.compute_routes(
            -33.8567844, 151.213108, -33.87365, 151.20689
        )

    assert payload["routes"][0]["distanceMeters"] == 1000


@pytest.mark.asyncio
async def test_geocoding_status_error_becomes_google_maps_error() -> None:
    """
    Verify that Geocoding API status failures become normalized integration errors.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        """
        Handle the mocked Geocoding request.

        Args:
            request: The outgoing HTTP request.

        Returns:
            A mocked failed Geocoding response.
        """
        assert request.url.params["key"] == "test-key"
        assert request.url.params["address"] == "Sydney"
        return httpx.Response(
            200, json={"status": "REQUEST_DENIED", "error_message": "API disabled"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GoogleGeocodingClient(GoogleMapsHttpClient("test-key", http_client))
        with pytest.raises(GoogleMapsApiError, match="API disabled"):
            await client.geocode("Sydney")


def _json_body(request: httpx.Request) -> dict[str, Any]:
    """
    Decode a mocked request JSON body.

    Args:
        request: The outgoing HTTP request.

    Returns:
        The decoded JSON body.
    """
    payload = json.loads(request.content.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload
