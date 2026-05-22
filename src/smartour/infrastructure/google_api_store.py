"""SQLite-backed Google API cache and metrics storage."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from smartour.infrastructure.database import SQLiteDatabase


def _utc_now() -> datetime:
    """
    Return the current UTC datetime.

    Returns:
        The current UTC datetime.
    """
    return datetime.now(tz=UTC)


class SQLiteGoogleApiStore:
    """
    Stores Google API cache entries and request metrics in SQLite.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        """
        Initialize the store.

        Args:
            database: The SQLite database.
        """
        self.database = database

    async def get_cached_response(self, cache_key: str) -> dict[str, Any] | None:
        """
        Return a cached Google API response when it is still valid.

        Args:
            cache_key: The normalized cache key.

        Returns:
            The cached response payload when available.
        """
        now = _utc_now().isoformat()
        async with (
            self.database.connect() as connection,
            connection.execute(
                """
                SELECT response_json
                FROM google_api_cache_entries
                WHERE cache_key = ? AND expires_at > ?
                """,
                (cache_key, now),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        payload = json.loads(row["response_json"])
        if not isinstance(payload, dict):
            return None
        return payload

    async def save_cached_response(
        self,
        cache_key: str,
        service: str,
        endpoint: str,
        field_mask: str | None,
        request_hash: str,
        response_payload: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        """
        Save a Google API response in the cache.

        Args:
            cache_key: The normalized cache key.
            service: The Google API service name.
            endpoint: The requested endpoint.
            field_mask: The requested field mask.
            request_hash: The normalized request hash.
            response_payload: The response payload to cache.
            ttl_seconds: The cache lifetime in seconds.
        """
        if ttl_seconds <= 0:
            return
        now = _utc_now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO google_api_cache_entries (
                    cache_key, service, endpoint, field_mask, request_hash,
                    response_json, expires_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    service,
                    endpoint,
                    field_mask,
                    request_hash,
                    json.dumps(response_payload, sort_keys=True),
                    expires_at.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

    async def record_request_metric(
        self,
        service: str,
        endpoint: str,
        cache_hit: bool,
        status_code: int | None,
        duration_ms: float,
        job_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Record one Google API request metric.

        Args:
            service: The Google API service name.
            endpoint: The requested endpoint.
            cache_hit: Whether the request was served from cache.
            status_code: The HTTP status code when available.
            duration_ms: The request duration in milliseconds.
            job_id: The itinerary job that caused the request when available.
            error_message: The sanitized error message when available.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO google_api_request_metrics (
                    job_id, service, endpoint, cache_hit, status_code, duration_ms,
                    error_message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    service,
                    endpoint,
                    int(cache_hit),
                    status_code,
                    duration_ms,
                    error_message,
                    _utc_now().isoformat(),
                ),
            )

    async def summarize_request_metrics(
        self, since: datetime, job_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Return grouped Google API request metric summaries.

        Args:
            since: The inclusive lower bound for metric creation time.
            job_id: Optional itinerary job filter.

        Returns:
            Summary rows grouped by Google service and endpoint.
        """
        where_clauses = ["created_at >= ?"]
        parameters: list[Any] = [since.isoformat()]
        if job_id is not None:
            where_clauses.append("job_id = ?")
            parameters.append(job_id)
        where_sql = " AND ".join(where_clauses)
        async with (
            self.database.connect() as connection,
            connection.execute(
                f"""
                SELECT
                    service,
                    endpoint,
                    COUNT(*) AS total_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS cache_hits,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END)
                        AS estimated_billable_requests,
                    SUM(
                        CASE
                            WHEN status_code IS NULL
                                OR status_code >= 400
                                OR error_message IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS error_requests,
                    SUM(duration_ms) AS total_duration_ms,
                    AVG(duration_ms) AS average_duration_ms
                FROM google_api_request_metrics
                WHERE {where_sql}
                GROUP BY service, endpoint
                ORDER BY service, endpoint
                """,
                parameters,
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def count_metrics(self) -> int:
        """
        Return the total recorded Google API metric count.

        Returns:
            The metric row count.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT COUNT(*) AS metric_count FROM google_api_request_metrics"
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return 0
        return int(row["metric_count"])
