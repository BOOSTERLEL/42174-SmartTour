"""Tests for Google Maps cost monitoring."""

from pathlib import Path

import pytest

from smartour.application.cost_monitoring_service import CostMonitoringService
from smartour.infrastructure.database import SQLiteDatabase
from smartour.infrastructure.google_api_store import SQLiteGoogleApiStore


@pytest.mark.asyncio
async def test_cost_monitoring_summarizes_job_metrics(tmp_path: Path) -> None:
    """
    Verify that Google Maps cost summaries are grouped and filtered by job.
    """
    api_store = SQLiteGoogleApiStore(SQLiteDatabase(str(tmp_path / "smartour.sqlite3")))
    await api_store.record_request_metric(
        "places",
        "https://places.googleapis.com/v1/places:searchText",
        cache_hit=False,
        status_code=200,
        duration_ms=100.0,
        job_id="job_1",
    )
    await api_store.record_request_metric(
        "places",
        "https://places.googleapis.com/v1/places:searchText",
        cache_hit=True,
        status_code=200,
        duration_ms=0.0,
        job_id="job_1",
    )
    await api_store.record_request_metric(
        "routes",
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        cache_hit=False,
        status_code=500,
        duration_ms=30.0,
        job_id="job_1",
        error_message="quota exceeded",
    )
    await api_store.record_request_metric(
        "places",
        "https://places.googleapis.com/v1/places:searchText",
        cache_hit=False,
        status_code=200,
        duration_ms=80.0,
        job_id="job_2",
    )
    service = CostMonitoringService(
        api_store,
        unit_costs_usd={
            "places": 0.01,
            "routes": 0.02,
        },
    )

    summary = await service.get_google_maps_summary("job_1")

    assert summary.job_id == "job_1"
    assert summary.total_requests == 3
    assert summary.estimated_billable_requests == 2
    assert summary.cache_hits == 1
    assert summary.error_requests == 1
    assert summary.estimated_cost_usd == 0.03
    assert summary.average_duration_ms == 43.333
    service_counts = [
        (service.service, service.total_requests) for service in summary.services
    ]
    assert service_counts == [
        ("places", 2),
        ("routes", 1),
    ]
    assert summary.services[0].estimated_billable_requests == 1
    assert summary.services[0].cache_hits == 1
    assert summary.services[1].error_requests == 1


@pytest.mark.asyncio
async def test_cost_monitoring_returns_zero_summary_without_metrics(
    tmp_path: Path,
) -> None:
    """
    Verify that empty metric windows produce a zero summary.
    """
    api_store = SQLiteGoogleApiStore(SQLiteDatabase(str(tmp_path / "smartour.sqlite3")))
    service = CostMonitoringService(api_store, unit_costs_usd={})

    summary = await service.get_google_maps_summary("job_missing")

    assert summary.total_requests == 0
    assert summary.estimated_billable_requests == 0
    assert summary.cache_hits == 0
    assert summary.error_requests == 0
    assert summary.average_duration_ms == 0.0
    assert summary.estimated_cost_usd == 0.0
    assert summary.services == []
