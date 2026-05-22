"""Itinerary share-link repository implementations."""

from smartour.domain.share import ItineraryShareLink
from smartour.infrastructure.database import SQLiteDatabase


class InMemoryItineraryShareRepository:
    """
    Process-local in-memory itinerary share-link repository.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """
        self.share_links: dict[str, ItineraryShareLink] = {}

    async def save(self, share_link: ItineraryShareLink) -> None:
        """
        Save an itinerary share link.

        Args:
            share_link: The share link to persist.
        """
        self.share_links[share_link.token] = share_link.model_copy(deep=True)

    async def get_by_token(self, token: str) -> ItineraryShareLink | None:
        """
        Return a share link by token.

        Args:
            token: The share token.

        Returns:
            The share link when found.
        """
        share_link = self.share_links.get(token)
        if share_link is None:
            return None
        return share_link.model_copy(deep=True)


class SQLiteItineraryShareRepository:
    """
    SQLite-backed itinerary share-link repository.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        """
        Initialize the repository.

        Args:
            database: The SQLite database.
        """
        self.database = database

    async def save(self, share_link: ItineraryShareLink) -> None:
        """
        Save an itinerary share link.

        Args:
            share_link: The share link to persist.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO itinerary_share_links (
                    token, itinerary_id, payload, created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET
                    itinerary_id = excluded.itinerary_id,
                    payload = excluded.payload
                """,
                (
                    share_link.token,
                    share_link.itinerary_id,
                    share_link.model_dump_json(),
                    share_link.created_at.isoformat(),
                ),
            )

    async def get_by_token(self, token: str) -> ItineraryShareLink | None:
        """
        Return a share link by token.

        Args:
            token: The share token.

        Returns:
            The share link when found.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT payload FROM itinerary_share_links WHERE token = ?",
                (token,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return ItineraryShareLink.model_validate_json(row["payload"])
