"""Shared FastAPI dependencies for the Smartour API."""

from collections.abc import AsyncIterator
from functools import lru_cache

import httpx

from smartour.application.conversation_service import ConversationService
from smartour.application.requirement_extractor import (
    RequirementExtractor,
    RuleBasedRequirementExtractor,
)
from smartour.core.config import Settings
from smartour.infrastructure.repositories.conversations import (
    InMemoryConversationRepository,
)
from smartour.integrations.google_maps.client import (
    GoogleMapsClient,
    create_google_maps_client,
)
from smartour.integrations.openai.requirement_extractor import (
    HybridRequirementExtractor,
    OpenAIRequirementExtractor,
)


@lru_cache
def get_settings() -> Settings:
    """
    Load application settings once per process.

    Returns:
        The validated application settings.
    """
    return Settings()


@lru_cache
def get_conversation_repository() -> InMemoryConversationRepository:
    """
    Create the process-local conversation repository.

    Returns:
        The in-memory conversation repository.
    """
    return InMemoryConversationRepository()


@lru_cache
def get_requirement_extractor() -> RequirementExtractor:
    """
    Create the requirement extractor.

    Returns:
        The configured requirement extractor.
    """
    fallback_extractor = RuleBasedRequirementExtractor()
    settings = get_settings()
    if not settings.has_openai_config():
        return fallback_extractor
    return HybridRequirementExtractor(
        primary_extractor=OpenAIRequirementExtractor(
            api_key=settings.openai_api_key or "",
            model=settings.openai_api_model or "",
            base_url=settings.openai_api_baseurl,
        ),
        fallback_extractor=fallback_extractor,
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


async def get_google_maps_client() -> AsyncIterator[GoogleMapsClient]:
    """
    Create a request-scoped Google Maps API client.

    Yields:
        A Google Maps client group backed by an async HTTP client.
    """
    settings = get_settings()
    timeout = httpx.Timeout(settings.google_maps_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        yield create_google_maps_client(settings.google_maps_api_key, http_client)
