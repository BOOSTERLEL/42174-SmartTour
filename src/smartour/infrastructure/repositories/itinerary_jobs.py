"""Itinerary job repository implementations."""

from smartour.domain.itinerary_job import ItineraryJob
from smartour.infrastructure.database import SQLiteDatabase


class InMemoryItineraryJobRepository:
    """
    Process-local in-memory itinerary job repository.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """
        self.jobs: dict[str, ItineraryJob] = {}

    async def save(self, job: ItineraryJob) -> None:
        """
        Save an itinerary job.

        Args:
            job: The itinerary job to save.
        """
        self.jobs[job.id] = job.model_copy(deep=True)

    async def get(self, job_id: str) -> ItineraryJob | None:
        """
        Return an itinerary job by ID.

        Args:
            job_id: The itinerary job ID.

        Returns:
            The itinerary job when found.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return job.model_copy(deep=True)


class SQLiteItineraryJobRepository:
    """
    SQLite-backed itinerary job repository.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        """
        Initialize the repository.

        Args:
            database: The SQLite database.
        """
        self.database = database

    async def save(self, job: ItineraryJob) -> None:
        """
        Save an itinerary job.

        Args:
            job: The itinerary job to save.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO itinerary_jobs (
                    id, conversation_id, status, payload, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    job.id,
                    job.conversation_id,
                    job.status.value,
                    job.model_dump_json(),
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )

    async def get(self, job_id: str) -> ItineraryJob | None:
        """
        Return an itinerary job by ID.

        Args:
            job_id: The itinerary job ID.

        Returns:
            The itinerary job when found.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT payload FROM itinerary_jobs WHERE id = ?",
                (job_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return ItineraryJob.model_validate_json(row["payload"])
