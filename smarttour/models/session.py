"""
Planning session persistence models.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Column, Integer, Text
from sqlmodel import Field, SQLModel

from smarttour.models.itinerary import Itinerary
from smarttour.models.preferences import TravelPreferences


class PlanningSession(SQLModel):
    """
    Public planning session model.
    """

    id: str
    created_at: datetime
    user_input: str = ""
    preferences: TravelPreferences
    itinerary: Itinerary
    version: int = Field(default=1, ge=1)


class PlanningSessionRecord(SQLModel, table=True):
    """
    SQLite table for saved planning sessions.
    """

    __tablename__ = "sessions"

    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_input: str = Field(
        default="", sa_column=Column(Text, nullable=False, default="")
    )
    preferences_json: str = Field(
        default="{}",
        sa_column=Column("preferences", Text, nullable=False, default="{}"),
    )
    itinerary_json: str = Field(
        default="{}",
        sa_column=Column("itinerary", Text, nullable=False, default="{}"),
    )
    version: int = Field(
        default=1, sa_column=Column(Integer, nullable=False, default=1)
    )

    def to_model(self) -> PlanningSession:
        """
        Convert the record to a public planning session model.

        Returns:
            The planning session representation.
        """

        preferences = TravelPreferences.model_validate_json(self.preferences_json)
        itinerary = Itinerary.model_validate_json(self.itinerary_json)
        return PlanningSession(
            id=self.id,
            created_at=self.created_at,
            user_input=self.user_input,
            preferences=preferences,
            itinerary=itinerary,
            version=self.version,
        )

    @classmethod
    def from_model(cls, planning_session: PlanningSession) -> PlanningSessionRecord:
        """
        Create a database record from a planning session.

        Args:
            planning_session: The public planning session model.

        Returns:
            A database-ready planning session record.
        """

        return cls(
            id=planning_session.id,
            created_at=planning_session.created_at,
            user_input=planning_session.user_input,
            preferences_json=planning_session.preferences.model_dump_json(),
            itinerary_json=planning_session.itinerary.model_dump_json(),
            version=planning_session.version,
        )
