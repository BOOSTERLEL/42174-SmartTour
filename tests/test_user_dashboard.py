"""Tests for user-owned plan dashboard data."""

import pytest
from fastapi import HTTPException, status

from smartour.api.dependencies import require_current_user
from smartour.application.conversation_service import ConversationService
from smartour.application.requirement_extractor import RuleBasedRequirementExtractor
from smartour.application.user_dashboard_service import UserDashboardService
from smartour.domain.itinerary import Itinerary
from smartour.domain.share import ItineraryShareLink
from smartour.infrastructure.repositories.conversations import (
    InMemoryConversationRepository,
)
from smartour.infrastructure.repositories.itineraries import InMemoryItineraryRepository
from smartour.infrastructure.repositories.shares import InMemoryItineraryShareRepository


@pytest.mark.asyncio
async def test_user_dashboard_lists_only_owned_itineraries_and_share_links() -> None:
    """
    Verify that dashboard data is scoped to the current user's ownership.
    """
    itinerary_repository = InMemoryItineraryRepository()
    share_repository = InMemoryItineraryShareRepository()
    owned_itinerary = _itinerary("itin_1", "usr_1", "Sydney Plan")
    other_itinerary = _itinerary("itin_2", "usr_2", "Tokyo Plan")
    anonymous_itinerary = _itinerary("itin_3", None, "Anonymous Plan")
    owned_share_link = ItineraryShareLink(
        itinerary_id=owned_itinerary.id,
        user_id="usr_1",
    )
    other_share_link = ItineraryShareLink(
        itinerary_id=other_itinerary.id,
        user_id="usr_2",
    )
    await itinerary_repository.save(owned_itinerary)
    await itinerary_repository.save(other_itinerary)
    await itinerary_repository.save(anonymous_itinerary)
    await share_repository.save(owned_share_link)
    await share_repository.save(other_share_link)
    service = UserDashboardService(itinerary_repository, share_repository)

    dashboard = await service.get_dashboard("usr_1")

    assert [item.itinerary_id for item in dashboard.created_itineraries] == [
        owned_itinerary.id
    ]
    assert dashboard.created_itineraries[0].title == "Sydney Plan"
    assert dashboard.created_itineraries[0].open_path == (
        f"/itineraries/{owned_itinerary.id}"
    )
    assert [share.token for share in dashboard.share_links] == [owned_share_link.token]
    assert dashboard.share_links[0].itinerary_title == "Sydney Plan"
    assert dashboard.share_links[0].share_path == owned_share_link.share_path


@pytest.mark.asyncio
async def test_conversation_service_sets_optional_owner() -> None:
    """
    Verify that newly created conversations can be owned by a user.
    """
    conversation_repository = InMemoryConversationRepository()
    service = ConversationService(
        conversation_repository=conversation_repository,
        requirement_extractor=RuleBasedRequirementExtractor(),
    )

    conversation = await service.create_conversation(user_id="usr_1")

    assert conversation.user_id == "usr_1"
    saved_conversation = await conversation_repository.get(conversation.id)
    assert saved_conversation is not None
    assert saved_conversation.user_id == "usr_1"


@pytest.mark.asyncio
async def test_require_current_user_rejects_anonymous_access() -> None:
    """
    Verify that protected user dashboard dependencies require authentication.
    """
    with pytest.raises(HTTPException) as error_info:
        await require_current_user(None)

    assert error_info.value.status_code == status.HTTP_401_UNAUTHORIZED


def _itinerary(itinerary_id: str, user_id: str | None, title: str) -> Itinerary:
    """
    Create an itinerary fixture.

    Args:
        itinerary_id: The itinerary ID.
        user_id: The owner user ID.
        title: The itinerary title.

    Returns:
        The itinerary fixture.
    """
    return Itinerary(
        id=itinerary_id,
        conversation_id="conv_1",
        user_id=user_id,
        title=title,
        destination_name=title.replace(" Plan", ""),
        guide_markdown=f"# {title}",
    )
