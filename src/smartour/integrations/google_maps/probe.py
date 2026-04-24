"""Google Maps Platform availability probe."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel

from smartour.core.config import Settings
from smartour.core.errors import ExternalServiceError
from smartour.integrations.google_maps.client import (
    GoogleMapsClient,
    create_google_maps_client,
)


class GoogleMapsProbeResult(BaseModel):
    """
    Result for one Google Maps service probe.
    """

    service: str
    ok: bool
    detail: str


class GoogleMapsProbeResponse(BaseModel):
    """
    Sanitized Google Maps availability probe response.
    """

    live: bool
    results: list[GoogleMapsProbeResult]
    note: str | None = None


async def run_google_maps_probe(
    google_maps_client: GoogleMapsClient,
) -> GoogleMapsProbeResponse:
    """
    Run low-cost live probes against the Google Maps APIs used by Smartour.

    Args:
        google_maps_client: The aggregated Google Maps client group.

    Returns:
        The sanitized probe response.
    """
    checks: list[tuple[str, Callable[[], Awaitable[str]]]] = [
        ("places_text_search_new", lambda: _probe_places(google_maps_client)),
        ("routes_compute_routes", lambda: _probe_routes(google_maps_client)),
        ("geocoding_v3", lambda: _probe_geocoding(google_maps_client)),
        ("timezone", lambda: _probe_timezone(google_maps_client)),
    ]
    results = [await _run_probe_check(service, check) for service, check in checks]
    return GoogleMapsProbeResponse(live=True, results=results)


async def _run_probe_check(
    service: str, check: Callable[[], Awaitable[str]]
) -> GoogleMapsProbeResult:
    """
    Run a single probe and normalize its result.

    Args:
        service: The probe service name.
        check: The async check to run.

    Returns:
        A sanitized probe result.
    """
    try:
        detail = await check()
    except ExternalServiceError as error:
        detail = str(error)
        return GoogleMapsProbeResult(service=service, ok=False, detail=detail)
    except Exception as error:
        detail = error.__class__.__name__
        return GoogleMapsProbeResult(service=service, ok=False, detail=detail)
    return GoogleMapsProbeResult(service=service, ok=True, detail=detail)


async def _probe_places(google_maps_client: GoogleMapsClient) -> str:
    """
    Probe Places Text Search New.

    Args:
        google_maps_client: The aggregated Google Maps client group.

    Returns:
        A short sanitized result detail.
    """
    payload = await google_maps_client.places.search_text(
        "coffee near Circular Quay Sydney", page_size=1
    )
    return f"returned_places={len(payload.get('places', []))}"


async def _probe_routes(google_maps_client: GoogleMapsClient) -> str:
    """
    Probe Routes Compute Routes.

    Args:
        google_maps_client: The aggregated Google Maps client group.

    Returns:
        A short sanitized result detail.
    """
    payload = await google_maps_client.routes.compute_routes(
        origin_latitude=-33.8567844,
        origin_longitude=151.213108,
        destination_latitude=-33.87365,
        destination_longitude=151.20689,
    )
    return f"returned_routes={len(payload.get('routes', []))}"


async def _probe_geocoding(google_maps_client: GoogleMapsClient) -> str:
    """
    Probe Geocoding API.

    Args:
        google_maps_client: The aggregated Google Maps client group.

    Returns:
        A short sanitized result detail.
    """
    payload = await google_maps_client.geocoding.geocode("Sydney NSW Australia")
    returned_results = len(payload.get("results", []))
    return f"status={payload.get('status')}; returned_results={returned_results}"


async def _probe_timezone(google_maps_client: GoogleMapsClient) -> str:
    """
    Probe Time Zone API.

    Args:
        google_maps_client: The aggregated Google Maps client group.

    Returns:
        A short sanitized result detail.
    """
    timestamp = int(datetime.now(tz=UTC).timestamp())
    payload = await google_maps_client.timezone.get_time_zone(
        -33.8688, 151.2093, timestamp
    )
    return f"status={payload.get('status')}; time_zone_id={payload.get('timeZoneId')}"


async def async_main() -> int:
    """
    Run the Google Maps availability probe as an async CLI command.

    Returns:
        The process exit code.
    """
    settings = Settings()
    timeout = httpx.Timeout(settings.google_maps_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        google_maps_client = create_google_maps_client(
            settings.google_maps_api_key, http_client
        )
        response = await run_google_maps_probe(google_maps_client)
    print(json.dumps(response.model_dump(), indent=2))
    if all(result.ok for result in response.results):
        return 0
    return 1


def main() -> None:
    """
    Execute the Google Maps availability probe CLI.
    """
    raise SystemExit(asyncio.run(async_main()))
