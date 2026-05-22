"""Tests for admin dashboard statistics and access control."""

from pathlib import Path

import pytest
from fastapi import HTTPException, status

from smartour.api.dependencies import require_admin_user, require_current_user
from smartour.api.routes.admin import get_admin_stats
from smartour.api.routes.costs import get_google_maps_cost_summary
from smartour.application.admin_service import AdminService
from smartour.application.cost_monitoring_service import CostMonitoringService
from smartour.domain.conversation import Conversation
from smartour.domain.itinerary import Itinerary
from smartour.domain.itinerary_job import ItineraryJob
from smartour.domain.share import ItineraryShareLink
from smartour.domain.user import User, normalize_username
from smartour.infrastructure.database import SQLiteDatabase
from smartour.infrastructure.google_api_store import SQLiteGoogleApiStore
from smartour.infrastructure.repositories.conversations import (
    SQLiteConversationRepository,
)
from smartour.infrastructure.repositories.itineraries import SQLiteItineraryRepository
from smartour.infrastructure.repositories.itinerary_jobs import (
    SQLiteItineraryJobRepository,
)
from smartour.infrastructure.repositories.shares import SQLiteItineraryShareRepository
from smartour.infrastructure.repositories.users import SQLiteUserRepository


@pytest.mark.asyncio
async def test_admin_service_returns_stats_and_cost_summary(tmp_path: Path) -> None:
    """
    Verify that admin stats include backend counts and Google Maps costs.
    """
    database = SQLiteDatabase(str(tmp_path / "smartour.sqlite3"))
    api_store = SQLiteGoogleApiStore(database)
    cost_service = CostMonitoringService(api_store, unit_costs_usd={"places": 0.01})
    admin_service = AdminService(database, cost_service)
    await _seed_admin_stats(database, api_store)

    stats = await admin_service.get_stats(window_hours=24)

    assert stats.record_counts.users == 2
    assert stats.record_counts.conversations == 1
    assert stats.record_counts.itineraries == 1
    assert stats.record_counts.itinerary_jobs == 1
    assert stats.record_counts.share_links == 1
    assert [(item.status, item.count) for item in stats.job_status_counts] == [
        ("succeeded", 1)
    ]
    assert stats.google_maps_cost_summary.total_requests == 1
    assert stats.google_maps_cost_summary.estimated_cost_usd == 0.01


@pytest.mark.asyncio
async def test_admin_stats_route_requires_admin_user(tmp_path: Path) -> None:
    """
    Verify that the admin stats route returns data for admin users.
    """
    database = SQLiteDatabase(str(tmp_path / "smartour.sqlite3"))
    api_store = SQLiteGoogleApiStore(database)
    admin_service = AdminService(
        database,
        CostMonitoringService(api_store, unit_costs_usd={}),
    )
    await _seed_admin_stats(database, api_store)

    stats = await get_admin_stats(admin_service, _user("admin", is_admin=True), 24)

    assert stats.record_counts.users == 2


@pytest.mark.asyncio
async def test_google_maps_cost_route_requires_admin_user(tmp_path: Path) -> None:
    """
    Verify that the cost route can be called by an admin user.
    """
    api_store = SQLiteGoogleApiStore(SQLiteDatabase(str(tmp_path / "smartour.sqlite3")))
    cost_service = CostMonitoringService(api_store, unit_costs_usd={})

    summary = await get_google_maps_cost_summary(
        cost_service,
        _user("admin", is_admin=True),
        window_hours=24,
    )

    assert summary.total_requests == 0


@pytest.mark.asyncio
async def test_admin_dependencies_reject_unauthorized_users() -> None:
    """
    Verify unauthenticated and non-admin users cannot access admin resources.
    """
    with pytest.raises(HTTPException) as unauthenticated_error:
        await require_current_user(None)
    with pytest.raises(HTTPException) as forbidden_error:
        await require_admin_user(_user("traveler", is_admin=False))

    assert unauthenticated_error.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert forbidden_error.value.status_code == status.HTTP_403_FORBIDDEN


async def _seed_admin_stats(
    database: SQLiteDatabase,
    api_store: SQLiteGoogleApiStore,
) -> None:
    """
    Seed records used by admin statistics tests.

    Args:
        database: The SQLite database.
        api_store: The Google API metrics store.
    """
    user_repository = SQLiteUserRepository(database)
    conversation_repository = SQLiteConversationRepository(database)
    itinerary_repository = SQLiteItineraryRepository(database)
    job_repository = SQLiteItineraryJobRepository(database)
    share_repository = SQLiteItineraryShareRepository(database)
    admin_user = _user("admin", is_admin=True)
    traveler_user = _user("traveler", is_admin=False)
    conversation = Conversation(user_id=traveler_user.id)
    itinerary = Itinerary(
        conversation_id=conversation.id,
        user_id=traveler_user.id,
        title="Sydney Plan",
        destination_name="Sydney",
        guide_markdown="# Sydney Plan",
    )
    job = ItineraryJob(conversation_id=conversation.id)
    job.mark_succeeded(itinerary.id)
    share_link = ItineraryShareLink(itinerary_id=itinerary.id, user_id=traveler_user.id)

    await user_repository.save_user(admin_user)
    await user_repository.save_user(traveler_user)
    await conversation_repository.save(conversation)
    await itinerary_repository.save(itinerary)
    await job_repository.save(job)
    await share_repository.save(share_link)
    await api_store.record_request_metric(
        "places",
        "https://places.googleapis.com/v1/places:searchText",
        cache_hit=False,
        status_code=200,
        duration_ms=50.0,
    )


def _user(username: str, is_admin: bool) -> User:
    """
    Create a user fixture.

    Args:
        username: The username.
        is_admin: Whether the user is an admin.

    Returns:
        The user fixture.
    """
    return User(
        username=username,
        normalized_username=normalize_username(username),
        password_hash="hash",
        password_salt="salt",
        is_admin=is_admin,
    )
