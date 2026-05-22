"""Itinerary repository implementations."""

from smartour.domain.itinerary import Itinerary
from smartour.infrastructure.database import SQLiteDatabase


class InMemoryItineraryRepository:
    """
    Process-local in-memory itinerary repository.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """
        self.itineraries: dict[str, Itinerary] = {}

    async def save(self, itinerary: Itinerary) -> None:
        """
        Save an itinerary.

        Args:
            itinerary: The itinerary to save.
        """
        self.itineraries[itinerary.id] = itinerary.model_copy(deep=True)

    async def get(self, itinerary_id: str) -> Itinerary | None:
        """
        Return an itinerary by ID.

        Args:
            itinerary_id: The itinerary ID.

        Returns:
            The itinerary when found.
        """
        itinerary = self.itineraries.get(itinerary_id)
        if itinerary is None:
            return None
        return itinerary.model_copy(deep=True)

    async def list_by_user(self, user_id: str) -> list[Itinerary]:
        """
        Return itineraries owned by a user.

        Args:
            user_id: The owner user ID.

        Returns:
            The user's itineraries sorted by creation time descending.
        """
        itineraries = [
            itinerary.model_copy(deep=True)
            for itinerary in self.itineraries.values()
            if itinerary.user_id == user_id
        ]
        return sorted(
            itineraries,
            key=lambda itinerary: itinerary.created_at,
            reverse=True,
        )


class SQLiteItineraryRepository:
    """
    SQLite-backed itinerary repository.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        """
        Initialize the repository.

        Args:
            database: The SQLite database.
        """
        self.database = database

    async def save(self, itinerary: Itinerary) -> None:
        """
        Save an itinerary.

        Args:
            itinerary: The itinerary to save.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO itineraries (
                    id, conversation_id, user_id, payload, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    user_id = excluded.user_id,
                    payload = excluded.payload
                """,
                (
                    itinerary.id,
                    itinerary.conversation_id,
                    itinerary.user_id,
                    itinerary.model_dump_json(),
                    itinerary.created_at.isoformat(),
                ),
            )

    async def get(self, itinerary_id: str) -> Itinerary | None:
        """
        Return an itinerary by ID.

        Args:
            itinerary_id: The itinerary ID.

        Returns:
            The itinerary when found.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT payload FROM itineraries WHERE id = ?",
                (itinerary_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return Itinerary.model_validate_json(row["payload"])

    async def list_by_user(self, user_id: str) -> list[Itinerary]:
        """
        Return itineraries owned by a user.

        Args:
            user_id: The owner user ID.

        Returns:
            The user's itineraries sorted by creation time descending.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                """
                SELECT payload FROM itineraries
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [Itinerary.model_validate_json(row["payload"]) for row in rows]
