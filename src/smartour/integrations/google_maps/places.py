"""Google Places API client."""

from typing import Any

from smartour.integrations.google_maps.client import GoogleMapsHttpClient
from smartour.integrations.google_maps.field_masks import (
    PLACES_DETAILS_FIELD_MASK,
    PLACES_DISCOVERY_FIELD_MASK,
)

PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"
TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


class GooglePlacesClient:
    """
    Client for Google Places API New requests.
    """

    def __init__(self, base_client: GoogleMapsHttpClient) -> None:
        """
        Initialize the Places client.

        Args:
            base_client: The shared Google Maps HTTP client.
        """
        self.base_client = base_client

    async def search_text(
        self,
        text_query: str,
        page_size: int = 5,
        field_mask: str = PLACES_DISCOVERY_FIELD_MASK,
        language_code: str | None = None,
        region_code: str | None = None,
        location_bias: dict[str, Any] | None = None,
        included_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Search places with a text query.

        Args:
            text_query: The query used to discover matching places.
            page_size: The maximum number of places to return.
            field_mask: The Google Places response field mask.
            language_code: The optional language code.
            region_code: The optional region code.
            location_bias: The optional location bias object.
            included_type: The optional Google place type filter.

        Returns:
            The Places Text Search response payload.
        """
        body: dict[str, Any] = {"textQuery": text_query, "pageSize": page_size}
        if language_code:
            body["languageCode"] = language_code
        if region_code:
            body["regionCode"] = region_code
        if location_bias:
            body["locationBias"] = location_bias
        if included_type:
            body["includedType"] = included_type
        return await self.base_client.post_json(
            "places", TEXT_SEARCH_URL, body, field_mask
        )

    async def get_place_details(
        self,
        place_id: str,
        field_mask: str = PLACES_DETAILS_FIELD_MASK,
        language_code: str | None = None,
        region_code: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve details for a Google place ID.

        Args:
            place_id: The Google place ID.
            field_mask: The Google Places response field mask.
            language_code: The optional language code.
            region_code: The optional region code.

        Returns:
            The Place Details response payload.
        """
        params: dict[str, Any] = {}
        if language_code:
            params["languageCode"] = language_code
        if region_code:
            params["regionCode"] = region_code
        url = PLACE_DETAILS_URL.format(place_id=place_id)
        return await self.base_client.get_json(
            "places", url, params | {"fields": field_mask}
        )
