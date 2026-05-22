"""Shared FastAPI dependencies for the Smartour API."""

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Request, status

from smartour.application.admin_service import AdminService
from smartour.application.auth_service import SESSION_COOKIE_NAME, AuthService
from smartour.application.conversation_service import ConversationService
from smartour.application.cost_monitoring_service import CostMonitoringService
from smartour.application.itinerary_job_service import ItineraryJobService
from smartour.application.planning_service import PlanningService
from smartour.application.report_service import ReportService
from smartour.application.requirement_extractor import (
    RequirementExtractor,
    RuleBasedRequirementExtractor,
)
from smartour.application.share_service import ShareService
from smartour.application.user_dashboard_service import UserDashboardService
from smartour.core.config import Settings
from smartour.domain.user import User
from smartour.infrastructure.database import SQLiteDatabase
from smartour.infrastructure.google_api_store import SQLiteGoogleApiStore
from smartour.infrastructure.rate_limit import SimpleRateLimiter, SQLiteRateLimitStore
from smartour.infrastructure.repositories.conversations import (
    SQLiteConversationRepository,
)
from smartour.infrastructure.repositories.itineraries import SQLiteItineraryRepository
from smartour.infrastructure.repositories.itinerary_jobs import (
    SQLiteItineraryJobRepository,
)
from smartour.infrastructure.repositories.shares import SQLiteItineraryShareRepository
from smartour.infrastructure.repositories.users import SQLiteUserRepository
from smartour.integrations.google_maps.client import (
    GoogleMapsClient,
    create_google_maps_client,
)
from smartour.integrations.requirement_model import RequirementModelExtractor


@lru_cache
def get_settings() -> Settings:
    """
    Load application settings once per process.

    Returns:
        The validated application settings.
    """
    return Settings()


@lru_cache
def get_database() -> SQLiteDatabase:
    """
    Create the process-local SQLite database handle.

    Returns:
        The SQLite database handle.
    """
    return SQLiteDatabase(get_settings().sqlite_path)


@lru_cache
def get_conversation_repository() -> SQLiteConversationRepository:
    """
    Create the process-local conversation repository.

    Returns:
        The SQLite conversation repository.
    """
    return SQLiteConversationRepository(get_database())


@lru_cache
def get_itinerary_repository() -> SQLiteItineraryRepository:
    """
    Create the process-local itinerary repository.

    Returns:
        The SQLite itinerary repository.
    """
    return SQLiteItineraryRepository(get_database())


@lru_cache
def get_itinerary_job_repository() -> SQLiteItineraryJobRepository:
    """
    Create the process-local itinerary job repository.

    Returns:
        The SQLite itinerary job repository.
    """
    return SQLiteItineraryJobRepository(get_database())


@lru_cache
def get_itinerary_share_repository() -> SQLiteItineraryShareRepository:
    """
    Create the process-local itinerary share-link repository.

    Returns:
        The SQLite itinerary share-link repository.
    """
    return SQLiteItineraryShareRepository(get_database())


@lru_cache
def get_user_repository() -> SQLiteUserRepository:
    """
    Create the process-local user repository.

    Returns:
        The SQLite user repository.
    """
    return SQLiteUserRepository(get_database())


@lru_cache
def get_auth_service() -> AuthService:
    """
    Create the local authentication service.

    Returns:
        The authentication service.
    """
    settings = get_settings()
    admin_usernames = [
        username.strip()
        for username in settings.admin_usernames.split(",")
        if username.strip()
    ]
    return AuthService(
        user_repository=get_user_repository(),
        admin_usernames=admin_usernames,
        session_ttl_seconds=settings.session_ttl_seconds,
    )


@lru_cache
def get_google_api_store() -> SQLiteGoogleApiStore:
    """
    Create the Google API cache and metrics store.

    Returns:
        The SQLite-backed Google API store.
    """
    return SQLiteGoogleApiStore(get_database())


@lru_cache
def get_cost_monitoring_service() -> CostMonitoringService:
    """
    Create the Google Maps cost monitoring service.

    Returns:
        The cost monitoring service.
    """
    settings = get_settings()
    return CostMonitoringService(
        google_api_store=get_google_api_store(),
        unit_costs_usd={
            "places": settings.google_maps_places_unit_cost_usd,
            "routes": settings.google_maps_routes_unit_cost_usd,
            "geocoding": settings.google_maps_geocoding_unit_cost_usd,
            "timezone": settings.google_maps_timezone_unit_cost_usd,
        },
    )


@lru_cache
def get_admin_service() -> AdminService:
    """
    Create the admin dashboard service.

    Returns:
        The admin dashboard service.
    """
    return AdminService(
        database=get_database(),
        cost_monitoring_service=get_cost_monitoring_service(),
    )


@lru_cache
def get_conversation_rate_limiter() -> SimpleRateLimiter:
    """
    Create the conversation-scoped itinerary generation rate limiter.

    Returns:
        The SQLite-backed conversation rate limiter.
    """
    settings = get_settings()
    return SimpleRateLimiter(
        store=SQLiteRateLimitStore(get_database()),
        max_events=settings.itinerary_job_conversation_rate_limit_count,
        window_seconds=settings.itinerary_job_rate_limit_window_seconds,
    )


@lru_cache
def get_ip_rate_limiter() -> SimpleRateLimiter:
    """
    Create the client IP-scoped itinerary generation rate limiter.

    Returns:
        The SQLite-backed client IP rate limiter.
    """
    settings = get_settings()
    return SimpleRateLimiter(
        store=SQLiteRateLimitStore(get_database()),
        max_events=settings.itinerary_job_ip_rate_limit_count,
        window_seconds=settings.itinerary_job_rate_limit_window_seconds,
    )


@lru_cache
def get_requirement_extractor() -> RequirementExtractor:
    """
    Create the requirement extractor.

    Returns:
        The configured requirement extractor.
    """
    settings = get_settings()
    if settings.requirement_model_path:
        return RequirementModelExtractor(
            model_path=settings.requirement_model_path,
            confidence_threshold=settings.requirement_model_confidence_threshold,
        )
    if settings.requirement_model_development_fallback_enabled:
        return RuleBasedRequirementExtractor()
    raise RuntimeError(
        "REQUIREMENT_MODEL_PATH is required when requirement model development "
        "fallback is disabled"
    )


@lru_cache
def get_conversation_service() -> ConversationService:
    """
    Create the conversation service.

    Returns:
        The conversation service.
    """
    return ConversationService(
        conversation_repository=get_conversation_repository(),
        requirement_extractor=get_requirement_extractor(),
    )


@lru_cache
def get_planning_service() -> PlanningService:
    """
    Create the planning service.

    Returns:
        The planning service.
    """
    return PlanningService(
        conversation_repository=get_conversation_repository(),
        itinerary_repository=get_itinerary_repository(),
    )


@lru_cache
def get_report_service() -> ReportService:
    """
    Create the itinerary report service.

    Returns:
        The report service.
    """
    return ReportService(itinerary_repository=get_itinerary_repository())


@lru_cache
def get_share_service() -> ShareService:
    """
    Create the itinerary share-link service.

    Returns:
        The share-link service.
    """
    return ShareService(
        itinerary_repository=get_itinerary_repository(),
        share_repository=get_itinerary_share_repository(),
        report_service=get_report_service(),
    )


@lru_cache
def get_user_dashboard_service() -> UserDashboardService:
    """
    Create the current-user dashboard service.

    Returns:
        The user dashboard service.
    """
    return UserDashboardService(
        itinerary_repository=get_itinerary_repository(),
        share_repository=get_itinerary_share_repository(),
    )


@lru_cache
def get_itinerary_job_service() -> ItineraryJobService:
    """
    Create the itinerary job service.

    Returns:
        The itinerary job service.
    """
    return ItineraryJobService(
        conversation_repository=get_conversation_repository(),
        job_repository=get_itinerary_job_repository(),
        planning_service=get_planning_service(),
        conversation_rate_limiter=get_conversation_rate_limiter(),
        ip_rate_limiter=get_ip_rate_limiter(),
    )


async def get_current_user(
    request: Request,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> User | None:
    """
    Return the current user when a valid session cookie is present.

    Args:
        request: The inbound HTTP request.
        auth_service: The authentication service.

    Returns:
        The authenticated user when present.
    """
    return await auth_service.get_user_for_session(
        request.cookies.get(SESSION_COOKIE_NAME)
    )


async def require_current_user(
    current_user: Annotated[User | None, Depends(get_current_user)],
) -> User:
    """
    Require an authenticated user.

    Args:
        current_user: The optional current user.

    Returns:
        The authenticated user.

    Raises:
        HTTPException: Raised when the request is unauthenticated.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return current_user


async def require_admin_user(
    current_user: Annotated[User, Depends(require_current_user)],
) -> User:
    """
    Require an authenticated admin user.

    Args:
        current_user: The authenticated user.

    Returns:
        The authenticated admin user.

    Raises:
        HTTPException: Raised when the request is not an admin.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def get_google_maps_client() -> AsyncIterator[GoogleMapsClient]:
    """
    Create a request-scoped Google Maps API client.

    Yields:
        A Google Maps client group backed by an async HTTP client.
    """
    settings = get_settings()
    timeout = httpx.Timeout(settings.google_maps_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        yield create_google_maps_client(
            settings.google_maps_api_key,
            http_client,
            api_store=get_google_api_store(),
            default_cache_ttl_seconds=settings.google_maps_cache_ttl_seconds,
            routes_cache_ttl_seconds=settings.google_maps_routes_cache_ttl_seconds,
        )
