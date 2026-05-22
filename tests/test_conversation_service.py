"""Tests for conversation requirement collection."""

import pytest

from smartour.application.conversation_service import ConversationService
from smartour.application.requirement_extractor import RuleBasedRequirementExtractor
from smartour.domain.conversation import ConversationState
from smartour.infrastructure.repositories.conversations import (
    InMemoryConversationRepository,
)


@pytest.mark.asyncio
async def test_conversation_collects_required_slots_from_initial_message() -> None:
    """
    Verify that the conversation service collects enough required slots.
    """
    service = _conversation_service()

    conversation = await service.create_conversation(
        "I want to go to Tokyo for 4 days with 2 people, moderate budget, "
        "relaxed pace, food and museums, stay near Shinjuku, use transit."
    )

    assert conversation.state == ConversationState.CONFIRMING_REQUIREMENTS
    assert conversation.requirement.destination == "Tokyo"
    assert conversation.requirement.trip_length_days == 4
    assert conversation.requirement.travelers.adults == 2
    assert conversation.requirement.budget_level == "medium"
    assert conversation.requirement.travel_pace == "relaxed"
    assert "food" in conversation.requirement.interests
    assert "museums" in conversation.requirement.interests
    assert conversation.requirement.hotel_area == "Shinjuku"
    assert conversation.requirement.transportation_mode == "transit"
    assert conversation.requirement.missing_required_slots() == []


@pytest.mark.asyncio
async def test_conversation_remains_collecting_when_slots_are_missing() -> None:
    """
    Verify that missing required slots keep the conversation in collection state.
    """
    service = _conversation_service()

    conversation = await service.create_conversation(
        "I want to visit Sydney for 3 days."
    )

    assert conversation.state == ConversationState.COLLECTING_REQUIREMENTS
    assert "travelers" in conversation.requirement.missing_required_slots()
    assert conversation.latest_assistant_message() is not None


@pytest.mark.asyncio
async def test_missing_slot_reply_includes_user_facing_options() -> None:
    """
    Verify missing requirement prompts include readable examples and options.
    """
    service = _conversation_service()

    conversation = await service.create_conversation(
        "I want to visit Sydney for 3 days with 2 people, medium budget, "
        "balanced pace, food and nature."
    )

    assistant_message = conversation.latest_assistant_message()

    assert assistant_message is not None
    assert "hotel_area" not in assistant_message
    assert "transportation_mode" not in assistant_message
    assert "preferred hotel area" in assistant_message
    assert "city center" in assistant_message
    assert "near public transit" in assistant_message
    assert "primary transportation mode" in assistant_message
    assert "transit, walking, or drive" in assistant_message


@pytest.mark.asyncio
async def test_short_hotel_area_reply_fills_missing_hotel_area() -> None:
    """
    Verify a short reply can answer the requested hotel area slot.
    """
    service = _conversation_service()
    conversation = await service.create_conversation(
        "We are 2 adults visiting Sydney for 3 days. We want a moderate budget, "
        "relaxed pace, museums, food, and transit."
    )

    updated_conversation = await service.handle_user_message(
        conversation.id, "city center"
    )

    assert updated_conversation is not None
    assert updated_conversation.state == ConversationState.CONFIRMING_REQUIREMENTS
    assert updated_conversation.requirement.hotel_area == "city center"
    assert updated_conversation.requirement.missing_required_slots() == []


@pytest.mark.asyncio
async def test_contextual_hotel_area_accepts_prompted_option_phrasing() -> None:
    """
    Verify explicit hotel-area wording fills the missing hotel area slot.
    """
    service = _conversation_service()
    conversation = await service.create_conversation(
        "We are 2 adults visiting Sydney for 3 days. We want a moderate budget, "
        "relaxed pace, museums, food, and transit."
    )

    updated_conversation = await service.handle_user_message(
        conversation.id, "preferred hotel area is near public transit"
    )

    assert updated_conversation is not None
    assert updated_conversation.state == ConversationState.CONFIRMING_REQUIREMENTS
    assert updated_conversation.requirement.hotel_area == "near public transit"
    assert updated_conversation.requirement.missing_required_slots() == []


@pytest.mark.asyncio
async def test_initial_message_extracts_contextual_hotel_area_phrase() -> None:
    """
    Verify hotel-area phrasing in a full initial message is collected.
    """
    service = _conversation_service()

    conversation = await service.create_conversation(
        "We are 2 adults visiting Sydney for 3 days. We want a moderate budget, "
        "relaxed pace, museums, harbour views, food, a CBD hotel area, and transit."
    )

    assert conversation.state == ConversationState.CONFIRMING_REQUIREMENTS
    assert conversation.requirement.hotel_area == "CBD"
    assert conversation.requirement.missing_required_slots() == []


@pytest.mark.asyncio
async def test_confirm_requirements_moves_conversation_to_planning() -> None:
    """
    Verify that confirmation moves a complete conversation into planning state.
    """
    service = _conversation_service()
    conversation = await service.create_conversation(
        "I want to visit Sydney for 3 days with 2 people, medium budget, "
        "balanced pace, food and nature, stay near station, use transit."
    )

    confirmed_conversation = await service.confirm_requirements(conversation.id)

    assert confirmed_conversation is not None
    assert confirmed_conversation.state == ConversationState.PLANNING
    assert (
        confirmed_conversation.latest_assistant_message()
        == "Requirements confirmed. Itinerary generation can start."
    )


def _conversation_service() -> ConversationService:
    """
    Create a conversation service for tests.

    Returns:
        A conversation service with in-memory dependencies.
    """
    return ConversationService(
        conversation_repository=InMemoryConversationRepository(),
        requirement_extractor=RuleBasedRequirementExtractor(),
    )
