"""Admin dashboard service for backend operational statistics."""

from datetime import UTC, datetime

from pydantic import BaseModel

from smartour.application.cost_monitoring_service import (
    CostMonitoringService,
    GoogleMapsCostSummary,
)
from smartour.infrastructure.database import SQLiteDatabase


class AdminRecordCounts(BaseModel):
    """
    Aggregate backend record counts for the admin dashboard.
    """

    users: int
    conversations: int
    itineraries: int
    itinerary_jobs: int
    share_links: int


class AdminJobStatusCount(BaseModel):
    """
    Itinerary job count for one status.
    """

    status: str
    count: int


class AdminStatsResponse(BaseModel):
    """
    Admin dashboard statistics response.
    """

    generated_at: datetime
    record_counts: AdminRecordCounts
    job_status_counts: list[AdminJobStatusCount]
    google_maps_cost_summary: GoogleMapsCostSummary


class AdminService:
    """
    Builds admin dashboard statistics from backend persistence.
    """

    def __init__(
        self,
        database: SQLiteDatabase,
        cost_monitoring_service: CostMonitoringService,
    ) -> None:
        """
        Initialize the admin service.

        Args:
            database: The SQLite database.
            cost_monitoring_service: Service used to summarize Google Maps costs.
        """
        self.database = database
        self.cost_monitoring_service = cost_monitoring_service

    async def get_stats(self, window_hours: int = 24) -> AdminStatsResponse:
        """
        Return backend statistics for the admin dashboard.

        Args:
            window_hours: Google Maps cost-summary lookback window in hours.

        Returns:
            The admin dashboard statistics.
        """
        google_maps_cost_summary = (
            await self.cost_monitoring_service.get_google_maps_summary(
                window_hours=window_hours
            )
        )
        return AdminStatsResponse(
            generated_at=datetime.now(tz=UTC),
            record_counts=await self._record_counts(),
            job_status_counts=await self._job_status_counts(),
            google_maps_cost_summary=google_maps_cost_summary,
        )

    async def _record_counts(self) -> AdminRecordCounts:
        """
        Return aggregate record counts.

        Returns:
            Backend record counts.
        """
        return AdminRecordCounts(
            users=await self._count_table("users"),
            conversations=await self._count_table("conversations"),
            itineraries=await self._count_table("itineraries"),
            itinerary_jobs=await self._count_table("itinerary_jobs"),
            share_links=await self._count_table("itinerary_share_links"),
        )

    async def _count_table(self, table_name: str) -> int:
        """
        Count rows in a known SQLite table.

        Args:
            table_name: The table name.

        Returns:
            The table row count.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                f"SELECT COUNT(*) AS record_count FROM {table_name}"
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return 0
        return int(row["record_count"])

    async def _job_status_counts(self) -> list[AdminJobStatusCount]:
        """
        Count itinerary jobs grouped by status.

        Returns:
            Job status counts.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                """
                SELECT status, COUNT(*) AS status_count
                FROM itinerary_jobs
                GROUP BY status
                ORDER BY status
                """
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [
            AdminJobStatusCount(status=row["status"], count=int(row["status_count"]))
            for row in rows
        ]
