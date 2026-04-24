"""Google Maps diagnostic API routes."""

from typing import Annotated

import httpx
from fastapi import APIRouter, Query

from smartour.api.dependencies import get_settings
from smartour.integrations.google_maps.client import create_google_maps_client
from smartour.integrations.google_maps.probe import (
    GoogleMapsProbeResponse,
    run_google_maps_probe,
)

router = APIRouter(prefix="/google-maps", tags=["google-maps"])


@router.get("/probe", response_model=GoogleMapsProbeResponse)
async def probe_google_maps(
    live: Annotated[bool, Query()] = False,
) -> GoogleMapsProbeResponse:
    """
    Probe Google Maps API availability.

    Args:
        live: Whether to call Google Maps APIs. When false, no external calls are made.

    Returns:
        A sanitized probe result that never includes API keys.
    """
    if not live:
        return GoogleMapsProbeResponse(
            live=False,
            results=[],
            note="Set live=true to run low-cost Google Maps API probes.",
        )
    settings = get_settings()
    timeout = httpx.Timeout(settings.google_maps_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        google_maps_client = create_google_maps_client(
            settings.google_maps_api_key, http_client
        )
        return await run_google_maps_probe(google_maps_client)
