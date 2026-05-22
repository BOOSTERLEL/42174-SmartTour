"""Application service for Google Maps cost monitoring."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


class GoogleMapsCostBreakdown(BaseModel):
    """
    Cost and usage summary for one Google Maps service endpoint.
    """

    service: str
    endpoint: str
    total_requests: int
    estimated_billable_requests: int
    cache_hits: int
    error_requests: int
    average_duration_ms: float
    estimated_cost_usd: float


class GoogleMapsCostSummary(BaseModel):
    """
    Cost and usage summary for Google Maps requests.
    """

    job_id: str | None = None
    window_hours: int = Field(ge=1)
    currency: str = "USD"
    generated_at: datetime
    total_requests: int
    estimated_billable_requests: int
    cache_hits: int
    error_requests: int
    average_duration_ms: float
    estimated_cost_usd: float
    services: list[GoogleMapsCostBreakdown]


class CostMonitoringService:
    """
    Builds cost-monitoring summaries from backend Google API metrics.
    """

    def __init__(
        self,
        google_api_store: Any,
        unit_costs_usd: Mapping[str, float],
    ) -> None:
        """
        Initialize the cost monitoring service.

        Args:
            google_api_store: Store used to query Google API request metrics.
            unit_costs_usd: Estimated per-request cost by Google service name.
        """
        self.google_api_store = google_api_store
        self.unit_costs_usd = dict(unit_costs_usd)

    async def get_google_maps_summary(
        self, job_id: str | None = None, window_hours: int = 24
    ) -> GoogleMapsCostSummary:
        """
        Return a Google Maps usage and estimated cost summary.

        Args:
            job_id: Optional itinerary job filter.
            window_hours: Lookback window in hours.

        Returns:
            A Google Maps cost summary.
        """
        generated_at = datetime.now(tz=UTC)
        since = generated_at - timedelta(hours=window_hours)
        rows = await self.google_api_store.summarize_request_metrics(since, job_id)
        services = [self._breakdown_from_row(row) for row in rows]
        total_requests = sum(service.total_requests for service in services)
        total_duration_ms = sum(
            float(row.get("total_duration_ms") or 0.0) for row in rows
        )
        average_duration_ms = (
            round(total_duration_ms / total_requests, 3)
            if total_requests > 0
            else 0.0
        )
        return GoogleMapsCostSummary(
            job_id=job_id,
            window_hours=window_hours,
            generated_at=generated_at,
            total_requests=total_requests,
            estimated_billable_requests=sum(
                service.estimated_billable_requests for service in services
            ),
            cache_hits=sum(service.cache_hits for service in services),
            error_requests=sum(service.error_requests for service in services),
            average_duration_ms=average_duration_ms,
            estimated_cost_usd=round(
                sum(service.estimated_cost_usd for service in services), 6
            ),
            services=services,
        )

    def _breakdown_from_row(self, row: Mapping[str, Any]) -> GoogleMapsCostBreakdown:
        """
        Convert one grouped metric row into a service breakdown.

        Args:
            row: A grouped metric row returned by the Google API store.

        Returns:
            A service endpoint cost breakdown.
        """
        service = str(row["service"])
        estimated_billable_requests = int(row["estimated_billable_requests"] or 0)
        return GoogleMapsCostBreakdown(
            service=service,
            endpoint=str(row["endpoint"]),
            total_requests=int(row["total_requests"] or 0),
            estimated_billable_requests=estimated_billable_requests,
            cache_hits=int(row["cache_hits"] or 0),
            error_requests=int(row["error_requests"] or 0),
            average_duration_ms=round(float(row["average_duration_ms"] or 0.0), 3),
            estimated_cost_usd=round(
                estimated_billable_requests * self.unit_costs_usd.get(service, 0.0),
                6,
            ),
        )
