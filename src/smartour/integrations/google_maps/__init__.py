"""Google Maps Platform integration package."""

from smartour.integrations.google_maps.client import (
    GoogleMapsClient,
    create_google_maps_client,
)

__all__ = ["GoogleMapsClient", "create_google_maps_client"]
