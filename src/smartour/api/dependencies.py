"""Shared FastAPI dependencies for the Smartour API."""

from collections.abc import AsyncIterator
from functools import lru_cache

import httpx

from smartour.core.config import Settings
from smartour.integrations.google_maps.client import (
    GoogleMapsClient,
    create_google_maps_client,
)


@lru_cache
def get_settings() -> Settings:
    """
    Load application settings once per process.

    Returns:
        The validated application settings.
    """
    return Settings()


async def get_google_maps_client() -> AsyncIterator[GoogleMapsClient]:
    """
    Create a request-scoped Google Maps API client.

    Yields:
        A Google Maps client group backed by an async HTTP client.
    """
    settings = get_settings()
    timeout = httpx.Timeout(settings.google_maps_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        yield create_google_maps_client(settings.google_maps_api_key, http_client)
