"""Tests for itinerary generation job service."""

from typing import cast

import pytest

from smartour.application.itinerary_job_service import ItineraryJobService
from smartour.application.planning_service import PlanningService
from smartour.core.errors import PlanningInputError
from smartour.domain.conversation import Conversation, ConversationState
from smartour.domain.itinerary import Itinerary
from smartour.domain.itinerary_job import ItineraryJobStatus
from smartour.domain.requirement import Travelers, TravelRequirement
from smartour.infrastructure.repositories.conversations import (
    InMemoryConversationRepository,
)
from smartour.infrastructure.repositories.itinerary_jobs import (
    InMemoryItineraryJobRepository,
)
from smartour.integrations.google_maps.client import GoogleMapsClient


class FakePlanningService:
    """
    Fake planning service for job orchestration tests.
    """

    def __init__(
        self,
        itinerary: Itinerary | None = None,
        error: Exception | None = None,
    ) -> None:
        """
        Initialize the fake planning service.

        Args:
            itinerary: The itinerary returned by generation.
            error: The error raised by generation.
        """
        self.itinerary = itinerary
        self.error = error
        self.calls: list[str] = []

    async def generate_for_conversation(
        self, conversation_id: str, google_maps_client: GoogleMapsClient
    ) -> Itinerary | None:
        """
        Return a fake itinerary or raise a configured error.

        Args:
            conversation_id: The source conversation ID.
            google_maps_client: The Google Maps client group.

        Returns:
            The configured itinerary.
        """
        self.calls.append(conversation_id)
        if self.error is not None:
            raise self.error
        return self.itinerary


def test_create_job_sets_conversation_planning_state() -> None:
    """
    Verify that queued jobs move conversations into planning state.
    """
    conversation_repository = InMemoryConversationRepository()
    job_repository = InMemoryItineraryJobRepository()
    conversation = Conversation(requirement=_complete_requirement())
    conversation_repository.save(conversation)
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, FakePlanningService()),
    )

    job = service.create_job(conversation.id)

    assert job is not None
    assert job.status == ItineraryJobStatus.QUEUED
    saved_job = job_repository.get(job.id)
    saved_conversation = conversation_repository.get(conversation.id)
    assert saved_job is not None
    assert saved_conversation is not None
    assert saved_conversation.state == ConversationState.PLANNING
    assert saved_conversation.latest_assistant_message() == (
        "I am generating your itinerary now."
    )


@pytest.mark.asyncio
async def test_run_job_marks_job_succeeded() -> None:
    """
    Verify that successful generation stores itinerary ID and review state.
    """
    conversation_repository = InMemoryConversationRepository()
    job_repository = InMemoryItineraryJobRepository()
    conversation = Conversation(requirement=_complete_requirement())
    conversation_repository.save(conversation)
    itinerary = Itinerary(
        conversation_id=conversation.id,
        title="Tokyo Travel Guide",
        destination_name="Tokyo",
        guide_markdown="# Tokyo Travel Guide",
    )
    fake_planning_service = FakePlanningService(itinerary=itinerary)
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, fake_planning_service),
    )
    job = service.create_job(conversation.id)
    assert job is not None

    completed_job = await service.run_job(job.id, cast(GoogleMapsClient, object()))

    assert completed_job is not None
    assert completed_job.status == ItineraryJobStatus.SUCCEEDED
    assert completed_job.itinerary_id == itinerary.id
    assert fake_planning_service.calls == [conversation.id]
    saved_conversation = conversation_repository.get(conversation.id)
    assert saved_conversation is not None
    assert saved_conversation.state == ConversationState.READY_FOR_REVIEW
    assert saved_conversation.latest_assistant_message() == (
        "Your itinerary is ready for review."
    )


@pytest.mark.asyncio
async def test_run_job_marks_job_failed() -> None:
    """
    Verify that planning errors persist failed job and conversation state.
    """
    conversation_repository = InMemoryConversationRepository()
    job_repository = InMemoryItineraryJobRepository()
    conversation = Conversation(requirement=_complete_requirement())
    conversation_repository.save(conversation)
    fake_planning_service = FakePlanningService(
        error=PlanningInputError("No attraction candidates were found")
    )
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, fake_planning_service),
    )
    job = service.create_job(conversation.id)
    assert job is not None

    completed_job = await service.run_job(job.id, cast(GoogleMapsClient, object()))

    assert completed_job is not None
    assert completed_job.status == ItineraryJobStatus.FAILED
    assert completed_job.error_message == "No attraction candidates were found"
    saved_conversation = conversation_repository.get(conversation.id)
    assert saved_conversation is not None
    assert saved_conversation.state == ConversationState.FAILED


def test_create_job_rejects_incomplete_requirements() -> None:
    """
    Verify that jobs cannot start until required slots are complete.
    """
    conversation_repository = InMemoryConversationRepository()
    job_repository = InMemoryItineraryJobRepository()
    conversation = Conversation()
    conversation_repository.save(conversation)
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, FakePlanningService()),
    )

    with pytest.raises(PlanningInputError):
        service.create_job(conversation.id)


def test_create_job_returns_none_for_missing_conversation() -> None:
    """
    Verify that missing conversations are not queued.
    """
    service = ItineraryJobService(
        conversation_repository=InMemoryConversationRepository(),
        job_repository=InMemoryItineraryJobRepository(),
        planning_service=cast(PlanningService, FakePlanningService()),
    )

    assert service.create_job("missing") is None


def _complete_requirement() -> TravelRequirement:
    """
    Create a complete travel requirement for tests.

    Returns:
        A complete travel requirement.
    """
    return TravelRequirement(
        destination="Tokyo",
        trip_length_days=3,
        travelers=Travelers(adults=2),
        budget_level="medium",
        travel_pace="relaxed",
        interests=["food", "museums"],
        hotel_area="Shinjuku",
        transportation_mode="transit",
        language="en",
    )
