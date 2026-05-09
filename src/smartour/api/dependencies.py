"""Shared FastAPI dependencies for the Smartour API."""

from collections.abc import AsyncIterator
from functools import lru_cache

import httpx

from smartour.application.conversation_service import ConversationService
from smartour.application.itinerary_job_service import ItineraryJobService
from smartour.application.planning_service import PlanningService
from smartour.application.requirement_extractor import (
    RequirementExtractor,
    RuleBasedRequirementExtractor,
)
from smartour.core.config import Settings
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
def get_google_api_store() -> SQLiteGoogleApiStore:
    """
    Create the Google API cache and metrics store.

    Returns:
        The SQLite-backed Google API store.
    """
    return SQLiteGoogleApiStore(get_database())


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
