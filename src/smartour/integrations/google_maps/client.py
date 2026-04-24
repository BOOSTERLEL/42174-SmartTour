"""Shared Google Maps Platform HTTP client primitives."""

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
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

    def __init__(
        self,
        api_key: str,
        http_client: httpx.AsyncClient,
        api_store: Any | None = None,
        default_cache_ttl_seconds: int = 0,
        routes_cache_ttl_seconds: int = 0,
    ) -> None:
        """
        Initialize the low-level Google Maps HTTP client.

        Args:
            api_key: The Google Maps API key.
            http_client: The async HTTP client used for transport.
            api_store: Optional cache and metrics store.
            default_cache_ttl_seconds: Default cache TTL for idempotent requests.
            routes_cache_ttl_seconds: Cache TTL for Routes API requests.
        """
        self.api_key = api_key
        self.http_client = http_client
        self.api_store = api_store
        self.default_cache_ttl_seconds = default_cache_ttl_seconds
        self.routes_cache_ttl_seconds = routes_cache_ttl_seconds

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
        request_hash = _request_hash("GET", url, request_params, None, None)
        cache_key = _cache_key(service, request_hash)
        cached_payload = await self._cached_payload(cache_key, service)
        if cached_payload is not None:
            await self._record_metric(service, url, True, 200, 0.0)
            return cached_payload
        request_params["key"] = self.api_key
        start_time = perf_counter()
        try:
            response = await self.http_client.get(url, params=request_params)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            await self._record_metric(
                service,
                url,
                False,
                error.response.status_code,
                _duration_ms(start_time),
                error.response.text,
            )
            raise self._api_error_from_response(service, error.response) from error
        except httpx.HTTPError as error:
            await self._record_metric(
                service, url, False, None, _duration_ms(start_time), str(error)
            )
            raise GoogleMapsApiError(service, str(error)) from error
        payload = self._json_response(service, response)
        await self._record_metric(
            service, url, False, response.status_code, _duration_ms(start_time)
        )
        await self._save_cached_payload(
            cache_key, service, url, None, request_hash, payload
        )
        return payload

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
        request_hash = _request_hash("POST", url, None, body, field_mask)
        cache_key = _cache_key(service, request_hash)
        cached_payload = await self._cached_payload(cache_key, service)
        if cached_payload is not None:
            await self._record_metric(service, url, True, 200, 0.0)
            return cached_payload
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }
        start_time = perf_counter()
        try:
            response = await self.http_client.post(url, json=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            await self._record_metric(
                service,
                url,
                False,
                error.response.status_code,
                _duration_ms(start_time),
                error.response.text,
            )
            raise self._api_error_from_response(service, error.response) from error
        except httpx.HTTPError as error:
            await self._record_metric(
                service, url, False, None, _duration_ms(start_time), str(error)
            )
            raise GoogleMapsApiError(service, str(error)) from error
        payload = self._json_response(service, response)
        await self._record_metric(
            service, url, False, response.status_code, _duration_ms(start_time)
        )
        await self._save_cached_payload(
            cache_key, service, url, field_mask, request_hash, payload
        )
        return payload

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

    async def _cached_payload(
        self, cache_key: str, service: str
    ) -> dict[str, Any] | None:
        """
        Return a cached payload when caching is enabled.

        Args:
            cache_key: The normalized cache key.
            service: The logical Google Maps service name.

        Returns:
            The cached payload when available.
        """
        if self.api_store is None or self._cache_ttl(service) <= 0:
            return None
        return await self.api_store.get_cached_response(cache_key)

    async def _save_cached_payload(
        self,
        cache_key: str,
        service: str,
        endpoint: str,
        field_mask: str | None,
        request_hash: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Save a response payload when caching is enabled.

        Args:
            cache_key: The normalized cache key.
            service: The logical Google Maps service name.
            endpoint: The requested endpoint.
            field_mask: The requested field mask.
            request_hash: The normalized request hash.
            payload: The response payload.
        """
        ttl_seconds = self._cache_ttl(service)
        if self.api_store is None or ttl_seconds <= 0:
            return
        await self.api_store.save_cached_response(
            cache_key,
            service,
            endpoint,
            field_mask,
            request_hash,
            payload,
            ttl_seconds,
        )

    async def _record_metric(
        self,
        service: str,
        endpoint: str,
        cache_hit: bool,
        status_code: int | None,
        duration_ms: float,
        error_message: str | None = None,
    ) -> None:
        """
        Record a Google API request metric when metrics are enabled.

        Args:
            service: The logical Google Maps service name.
            endpoint: The requested endpoint.
            cache_hit: Whether the response came from cache.
            status_code: The HTTP status code when available.
            duration_ms: The request duration in milliseconds.
            error_message: The sanitized error message when available.
        """
        if self.api_store is None:
            return
        await self.api_store.record_request_metric(
            service, endpoint, cache_hit, status_code, duration_ms, error_message
        )

    def _cache_ttl(self, service: str) -> int:
        """
        Return the cache TTL for a Google API service.

        Args:
            service: The logical Google Maps service name.

        Returns:
            The cache TTL in seconds.
        """
        if service == "routes":
            return self.routes_cache_ttl_seconds
        return self.default_cache_ttl_seconds


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
    api_key: str,
    http_client: httpx.AsyncClient,
    api_store: Any | None = None,
    default_cache_ttl_seconds: int = 0,
    routes_cache_ttl_seconds: int = 0,
) -> GoogleMapsClient:
    """
    Create an aggregated Google Maps client group.

    Args:
        api_key: The Google Maps API key.
        http_client: The async HTTP client used for transport.
        api_store: Optional cache and metrics store.
        default_cache_ttl_seconds: Default cache TTL for idempotent requests.
        routes_cache_ttl_seconds: Cache TTL for Routes API requests.

    Returns:
        The aggregated Google Maps client group.
    """
    from smartour.integrations.google_maps.geocoding import GoogleGeocodingClient
    from smartour.integrations.google_maps.places import GooglePlacesClient
    from smartour.integrations.google_maps.routes import GoogleRoutesClient
    from smartour.integrations.google_maps.timezone import GoogleTimeZoneClient

    base_client = GoogleMapsHttpClient(
        api_key,
        http_client,
        api_store=api_store,
        default_cache_ttl_seconds=default_cache_ttl_seconds,
        routes_cache_ttl_seconds=routes_cache_ttl_seconds,
    )
    return GoogleMapsClient(
        places=GooglePlacesClient(base_client),
        routes=GoogleRoutesClient(base_client),
        geocoding=GoogleGeocodingClient(base_client),
        timezone=GoogleTimeZoneClient(base_client),
    )


def _request_hash(
    method: str,
    url: str,
    params: dict[str, Any] | None,
    body: dict[str, Any] | None,
    field_mask: str | None,
) -> str:
    """
    Build a stable hash for an idempotent Google API request.

    Args:
        method: The HTTP method.
        url: The requested endpoint URL.
        params: The request query parameters.
        body: The request JSON body.
        field_mask: The requested field mask.

    Returns:
        A SHA-256 request hash.
    """
    normalized_request = {
        "body": body or {},
        "field_mask": field_mask,
        "method": method,
        "params": params or {},
        "url": url,
    }
    request_json = json.dumps(normalized_request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(request_json.encode("utf-8")).hexdigest()


def _cache_key(service: str, request_hash: str) -> str:
    """
    Build a namespaced cache key.

    Args:
        service: The logical Google Maps service name.
        request_hash: The normalized request hash.

    Returns:
        The cache key.
    """
    return f"{service}:{request_hash}"


def _duration_ms(start_time: float) -> float:
    """
    Return elapsed milliseconds from a perf counter start value.

    Args:
        start_time: The perf counter start value.

    Returns:
        The elapsed duration in milliseconds.
    """
    return (perf_counter() - start_time) * 1000.0
