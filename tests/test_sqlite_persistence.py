"""Tests for SQLite persistence repositories."""

from pathlib import Path

import pytest

from smartour.domain.conversation import Conversation, ConversationState
from smartour.domain.itinerary import Itinerary
from smartour.domain.itinerary_job import ItineraryJob, ItineraryJobStatus
from smartour.infrastructure.database import SQLiteDatabase
from smartour.infrastructure.repositories.conversations import (
    SQLiteConversationRepository,
)
from smartour.infrastructure.repositories.itineraries import SQLiteItineraryRepository
from smartour.infrastructure.repositories.itinerary_jobs import (
    SQLiteItineraryJobRepository,
)


@pytest.mark.asyncio
async def test_sqlite_repositories_persist_domain_models(tmp_path: Path) -> None:
    """
    Verify that SQLite repositories persist and restore core domain models.
    """
    database = SQLiteDatabase(str(tmp_path / "smartour.sqlite3"))
    conversation_repository = SQLiteConversationRepository(database)
    itinerary_repository = SQLiteItineraryRepository(database)
    job_repository = SQLiteItineraryJobRepository(database)
    conversation = Conversation(state=ConversationState.CONFIRMING_REQUIREMENTS)
    itinerary = Itinerary(
        conversation_id=conversation.id,
        title="Tokyo Travel Guide",
        destination_name="Tokyo",
        guide_markdown="# Tokyo Travel Guide",
    )
    job = ItineraryJob(conversation_id=conversation.id)
    job.mark_succeeded(itinerary.id)

    await conversation_repository.save(conversation)
    await itinerary_repository.save(itinerary)
    await job_repository.save(job)

    saved_conversation = await conversation_repository.get(conversation.id)
    saved_itinerary = await itinerary_repository.get(itinerary.id)
    saved_job = await job_repository.get(job.id)
    assert saved_conversation is not None
    assert saved_conversation.state == ConversationState.CONFIRMING_REQUIREMENTS
    assert saved_itinerary is not None
    assert saved_itinerary.title == "Tokyo Travel Guide"
    assert saved_job is not None
    assert saved_job.status == ItineraryJobStatus.SUCCEEDED
    assert saved_job.itinerary_id == itinerary.id
