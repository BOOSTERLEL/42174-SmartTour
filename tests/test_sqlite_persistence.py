"""Tests for SQLite persistence repositories."""

from pathlib import Path

import aiosqlite
import pytest

from smartour.domain.conversation import Conversation, ConversationState
from smartour.domain.itinerary import Itinerary
from smartour.domain.itinerary_job import ItineraryJob, ItineraryJobStatus
from smartour.domain.share import ItineraryShareLink
from smartour.infrastructure.database import SQLiteDatabase
from smartour.infrastructure.repositories.conversations import (
    SQLiteConversationRepository,
)
from smartour.infrastructure.repositories.itineraries import SQLiteItineraryRepository
from smartour.infrastructure.repositories.itinerary_jobs import (
    SQLiteItineraryJobRepository,
)
from smartour.infrastructure.repositories.shares import SQLiteItineraryShareRepository


@pytest.mark.asyncio
async def test_sqlite_repositories_persist_domain_models(tmp_path: Path) -> None:
    """
    Verify that SQLite repositories persist and restore core domain models.
    """
    database = SQLiteDatabase(str(tmp_path / "smartour.sqlite3"))
    conversation_repository = SQLiteConversationRepository(database)
    itinerary_repository = SQLiteItineraryRepository(database)
    job_repository = SQLiteItineraryJobRepository(database)
    share_repository = SQLiteItineraryShareRepository(database)
    conversation = Conversation(state=ConversationState.CONFIRMING_REQUIREMENTS)
    itinerary = Itinerary(
        conversation_id=conversation.id,
        title="Tokyo Travel Guide",
        destination_name="Tokyo",
        guide_markdown="# Tokyo Travel Guide",
    )
    job = ItineraryJob(conversation_id=conversation.id)
    job.mark_succeeded(itinerary.id)
    share_link = ItineraryShareLink(itinerary_id=itinerary.id)

    await conversation_repository.save(conversation)
    await itinerary_repository.save(itinerary)
    await job_repository.save(job)
    await share_repository.save(share_link)

    saved_conversation = await conversation_repository.get(conversation.id)
    saved_itinerary = await itinerary_repository.get(itinerary.id)
    saved_job = await job_repository.get(job.id)
    saved_share_link = await share_repository.get_by_token(share_link.token)
    assert saved_conversation is not None
    assert saved_conversation.state == ConversationState.CONFIRMING_REQUIREMENTS
    assert saved_itinerary is not None
    assert saved_itinerary.title == "Tokyo Travel Guide"
    assert saved_job is not None
    assert saved_job.status == ItineraryJobStatus.SUCCEEDED
    assert saved_job.itinerary_id == itinerary.id
    assert saved_share_link is not None
    assert saved_share_link.itinerary_id == itinerary.id


@pytest.mark.asyncio
async def test_sqlite_database_adds_google_metric_job_id_column(
    tmp_path: Path,
) -> None:
    """
    Verify that schema initialization migrates legacy metric tables.
    """
    database_path = tmp_path / "legacy.sqlite3"
    async with aiosqlite.connect(str(database_path)) as connection:
        await connection.execute(
            """
            CREATE TABLE google_api_request_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                status_code INTEGER,
                duration_ms REAL NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO google_api_request_metrics (
                service, endpoint, cache_hit, status_code, duration_ms,
                error_message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("places", "https://example.test", 0, 200, 10.0, None, "2026-05-22"),
        )
        await connection.commit()
    database = SQLiteDatabase(str(database_path))

    await database.initialize()

    async with database.connect() as connection:
        async with connection.execute(
            "PRAGMA table_info(google_api_request_metrics)"
        ) as cursor:
            rows = await cursor.fetchall()
        async with connection.execute(
            "SELECT COUNT(*) AS metric_count FROM google_api_request_metrics"
        ) as cursor:
            metric_row = await cursor.fetchone()
    assert "job_id" in {row[1] for row in rows}
    assert metric_row is not None
    assert metric_row["metric_count"] == 1
