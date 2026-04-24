"""SQLite-backed rate limiting helpers."""

from datetime import UTC, datetime, timedelta

from smartour.core.errors import RateLimitError
from smartour.infrastructure.database import SQLiteDatabase


def _utc_now() -> datetime:
    """
    Return the current UTC datetime.

    Returns:
        The current UTC datetime.
    """
    return datetime.now(tz=UTC)


class SQLiteRateLimitStore:
    """
    Stores rate limit events in SQLite.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        """
        Initialize the store.

        Args:
            database: The SQLite database.
        """
        self.database = database

    async def record_event(self, scope: str, subject_key: str, event_name: str) -> None:
        """
        Record a rate limit event.

        Args:
            scope: The rate limit scope.
            subject_key: The scoped subject key.
            event_name: The event name.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO rate_limit_events (
                    scope, subject_key, event_name, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (scope, subject_key, event_name, _utc_now().isoformat()),
            )

    async def count_events_since(
        self,
        scope: str,
        subject_key: str,
        event_name: str,
        since: datetime,
    ) -> int:
        """
        Count rate limit events since a cutoff time.

        Args:
            scope: The rate limit scope.
            subject_key: The scoped subject key.
            event_name: The event name.
            since: The cutoff datetime.

        Returns:
            The matching event count.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                """
                SELECT COUNT(*) AS event_count
                FROM rate_limit_events
                WHERE scope = ?
                  AND subject_key = ?
                  AND event_name = ?
                  AND created_at >= ?
                """,
                (scope, subject_key, event_name, since.isoformat()),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return 0
        return int(row["event_count"])


class SimpleRateLimiter:
    """
    Applies fixed-window rate limits using a SQLite event store.
    """

    def __init__(
        self,
        store: SQLiteRateLimitStore,
        max_events: int,
        window_seconds: int,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            store: The rate limit event store.
            max_events: The maximum allowed events in the window.
            window_seconds: The fixed window length in seconds.
        """
        self.store = store
        self.max_events = max_events
        self.window_seconds = window_seconds

    async def check_and_record(
        self, scope: str, subject_key: str, event_name: str
    ) -> None:
        """
        Enforce a fixed-window rate limit and record a permitted event.

        Args:
            scope: The rate limit scope.
            subject_key: The scoped subject key.
            event_name: The event name.

        Raises:
            RateLimitError: Raised when the limit has been exceeded.
        """
        cutoff = _utc_now() - timedelta(seconds=self.window_seconds)
        event_count = await self.store.count_events_since(
            scope, subject_key, event_name, cutoff
        )
        if event_count >= self.max_events:
            raise RateLimitError("Too many itinerary generation requests")
        await self.store.record_event(scope, subject_key, event_name)
