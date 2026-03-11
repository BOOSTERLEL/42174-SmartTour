"""
HTTP client for the FastAPI backend.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from smarttour.config import get_settings
from smarttour.models import (
    Attraction,
    AttractionSearchResponse,
    GuidanceRequest,
    GuidanceResponse,
    ItineraryGenerateRequest,
    PlanningSession,
    PreferenceParseRequest,
    PreferenceParseResponse,
    PreferenceRefineRequest,
    RouteOptimizationRequest,
    RouteOptimizationResponse,
)


class APIClient:
    """
    Synchronous HTTP client used by the Streamlit frontend.
    """

    def __init__(self, base_url: str | None = None) -> None:
        """
        Initialize the API client.

        Args:
            base_url: Optional backend base URL override.
        """

        self.base_url = (base_url or get_settings().backend_url).rstrip("/")

    def parse_preferences(
        self, payload: PreferenceParseRequest
    ) -> PreferenceParseResponse:
        """
        Request preference parsing.

        Args:
            payload: Parse request payload.

        Returns:
            Parsed preferences.
        """

        data = self._post("/api/preferences/parse", payload.model_dump(mode="json"))
        return PreferenceParseResponse.model_validate(data)

    def refine_preferences(
        self, payload: PreferenceRefineRequest
    ) -> PreferenceParseResponse:
        """
        Request preference refinement.

        Args:
            payload: Refinement payload.

        Returns:
            Refined preferences.
        """

        data = self._post("/api/preferences/refine", payload.model_dump(mode="json"))
        return PreferenceParseResponse.model_validate(data)

    def generate_itinerary(self, payload: ItineraryGenerateRequest) -> PlanningSession:
        """
        Generate an itinerary.

        Args:
            payload: Generation payload.

        Returns:
            Generated planning session.
        """

        data = self._post("/api/itinerary/generate", payload.model_dump(mode="json"))
        return PlanningSession.model_validate(data)

    def get_itinerary(self, session_id: str) -> PlanningSession:
        """
        Fetch a saved itinerary.

        Args:
            session_id: Saved session identifier.

        Returns:
            The saved planning session.
        """

        data = self._get(f"/api/itinerary/{session_id}")
        return PlanningSession.model_validate(data)

    def search_attractions(
        self,
        destination: str,
        categories: list[str] | None = None,
        max_cost: float | None = None,
        limit: int = 12,
    ) -> AttractionSearchResponse:
        """
        Search attractions.

        Args:
            destination: Destination name.
            categories: Optional categories.
            max_cost: Optional price ceiling.
            limit: Maximum result count.

        Returns:
            Search response payload.
        """

        params: dict[
            str,
            str | int | float | bool | None | Sequence[str | int | float | bool | None],
        ] = {"destination": destination, "limit": limit}
        if categories:
            params["categories"] = categories
        if max_cost is not None:
            params["max_cost"] = max_cost
        data = self._get("/api/attractions/search", params=params)
        return AttractionSearchResponse.model_validate(data)

    def get_attraction(self, attraction_id: str) -> Attraction:
        """
        Fetch a single attraction.

        Args:
            attraction_id: Attraction identifier.

        Returns:
            The matching attraction.
        """

        data = self._get(f"/api/attractions/{attraction_id}")
        return Attraction.model_validate(data)

    def optimize_route(
        self, payload: RouteOptimizationRequest
    ) -> RouteOptimizationResponse:
        """
        Optimize one route.

        Args:
            payload: Route optimization payload.

        Returns:
            Route optimization result.
        """

        data = self._post("/api/routes/optimize", payload.model_dump(mode="json"))
        return RouteOptimizationResponse.model_validate(data)

    def explain_attraction(self, payload: GuidanceRequest) -> GuidanceResponse:
        """
        Generate attraction guidance.

        Args:
            payload: Guidance request payload.

        Returns:
            Guidance response.
        """

        data = self._post("/api/guidance/explain", payload.model_dump(mode="json"))
        return GuidanceResponse.model_validate(data)

    def _get(
        self,
        path: str,
        params: dict[
            str,
            str | int | float | bool | None | Sequence[str | int | float | bool | None],
        ]
        | None = None,
    ) -> dict[str, object]:
        """
        Execute a GET request.

        Args:
            path: API path.
            params: Optional query parameters.

        Returns:
            Parsed JSON payload.
        """

        with httpx.Client(timeout=20.0) as client:
            response = client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            return response.json()

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """
        Execute a POST request.

        Args:
            path: API path.
            payload: JSON body.

        Returns:
            Parsed JSON payload.
        """

        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            return response.json()
