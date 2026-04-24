"""Tests for itinerary generation job service."""

from pathlib import Path
from typing import cast

import pytest

from smartour.api.routes.itineraries import (
    _format_sse_event,
    _job_event_payload,
    _job_event_stream,
)
from smartour.application.itinerary_job_service import ItineraryJobService
from smartour.application.planning_service import PlanningService
from smartour.core.errors import PlanningInputError, RateLimitError
from smartour.domain.conversation import Conversation, ConversationState
from smartour.domain.itinerary import Itinerary
from smartour.domain.itinerary_job import ItineraryJob, ItineraryJobStatus
from smartour.domain.requirement import Travelers, TravelRequirement
from smartour.infrastructure.database import SQLiteDatabase
from smartour.infrastructure.rate_limit import SimpleRateLimiter, SQLiteRateLimitStore
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


class FakeJobService:
    """
    Fake job service for SSE stream tests.
    """

    def __init__(self, job: ItineraryJob | None) -> None:
        """
        Initialize the fake job service.

        Args:
            job: The job returned by lookups.
        """
        self.job = job

    async def get_job(self, job_id: str) -> ItineraryJob | None:
        """
        Return the configured job.

        Args:
            job_id: The requested job ID.

        Returns:
            The configured job.
        """
        if self.job is None or self.job.id != job_id:
            return None
        return self.job


@pytest.mark.asyncio
async def test_create_job_sets_conversation_planning_state() -> None:
    """
    Verify that queued jobs move conversations into planning state.
    """
    conversation_repository = InMemoryConversationRepository()
    job_repository = InMemoryItineraryJobRepository()
    conversation = Conversation(requirement=_complete_requirement())
    await conversation_repository.save(conversation)
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, FakePlanningService()),
    )

    job = await service.create_job(conversation.id)

    assert job is not None
    assert job.status == ItineraryJobStatus.QUEUED
    saved_job = await job_repository.get(job.id)
    saved_conversation = await conversation_repository.get(conversation.id)
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
    await conversation_repository.save(conversation)
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
    job = await service.create_job(conversation.id)
    assert job is not None

    completed_job = await service.run_job(job.id, cast(GoogleMapsClient, object()))

    assert completed_job is not None
    assert completed_job.status == ItineraryJobStatus.SUCCEEDED
    assert completed_job.itinerary_id == itinerary.id
    assert fake_planning_service.calls == [conversation.id]
    saved_conversation = await conversation_repository.get(conversation.id)
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
    await conversation_repository.save(conversation)
    fake_planning_service = FakePlanningService(
        error=PlanningInputError("No attraction candidates were found")
    )
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, fake_planning_service),
    )
    job = await service.create_job(conversation.id)
    assert job is not None

    completed_job = await service.run_job(job.id, cast(GoogleMapsClient, object()))

    assert completed_job is not None
    assert completed_job.status == ItineraryJobStatus.FAILED
    assert completed_job.error_message == "No attraction candidates were found"
    saved_conversation = await conversation_repository.get(conversation.id)
    assert saved_conversation is not None
    assert saved_conversation.state == ConversationState.FAILED


@pytest.mark.asyncio
async def test_create_job_rejects_incomplete_requirements() -> None:
    """
    Verify that jobs cannot start until required slots are complete.
    """
    conversation_repository = InMemoryConversationRepository()
    job_repository = InMemoryItineraryJobRepository()
    conversation = Conversation()
    await conversation_repository.save(conversation)
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, FakePlanningService()),
    )

    with pytest.raises(PlanningInputError):
        await service.create_job(conversation.id)


@pytest.mark.asyncio
async def test_create_job_returns_none_for_missing_conversation() -> None:
    """
    Verify that missing conversations are not queued.
    """
    service = ItineraryJobService(
        conversation_repository=InMemoryConversationRepository(),
        job_repository=InMemoryItineraryJobRepository(),
        planning_service=cast(PlanningService, FakePlanningService()),
    )

    assert await service.create_job("missing") is None


@pytest.mark.asyncio
async def test_create_job_enforces_rate_limit(tmp_path: Path) -> None:
    """
    Verify that itinerary job creation enforces configured rate limits.
    """
    database = SQLiteDatabase(str(tmp_path / "smartour.sqlite3"))
    rate_limiter = SimpleRateLimiter(
        store=SQLiteRateLimitStore(database),
        max_events=1,
        window_seconds=3600,
    )
    conversation_repository = InMemoryConversationRepository()
    job_repository = InMemoryItineraryJobRepository()
    conversation = Conversation(requirement=_complete_requirement())
    await conversation_repository.save(conversation)
    service = ItineraryJobService(
        conversation_repository=conversation_repository,
        job_repository=job_repository,
        planning_service=cast(PlanningService, FakePlanningService()),
        rate_limiter=rate_limiter,
    )

    assert await service.create_job(conversation.id, "127.0.0.1") is not None
    with pytest.raises(RateLimitError):
        await service.create_job(conversation.id, "127.0.0.1")


def test_job_event_payload_is_json_ready() -> None:
    """
    Verify that job event payloads expose polling-friendly status fields.
    """
    job = ItineraryJob(conversation_id="conv_1")
    job.mark_succeeded("itin_1")

    payload = _job_event_payload(job)

    assert payload["job_id"] == job.id
    assert payload["conversation_id"] == "conv_1"
    assert payload["status"] == "succeeded"
    assert payload["itinerary_id"] == "itin_1"
    assert isinstance(payload["created_at"], str)
    assert isinstance(payload["completed_at"], str)


def test_format_sse_event_serializes_payload() -> None:
    """
    Verify that SSE events follow the event/data message shape.
    """
    event = _format_sse_event("itinerary_job", {"job_id": "job_1"})

    assert event == 'event: itinerary_job\ndata: {"job_id":"job_1"}\n\n'


@pytest.mark.asyncio
async def test_job_event_stream_stops_after_terminal_status() -> None:
    """
    Verify that terminal jobs emit one event and close the stream.
    """
    job = ItineraryJob(conversation_id="conv_1")
    job.mark_succeeded("itin_1")
    service = FakeJobService(job)

    events = [
        event
        async for event in _job_event_stream(job.id, cast(ItineraryJobService, service))
    ]

    assert len(events) == 1
    assert "succeeded" in events[0]
    assert "itin_1" in events[0]


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
