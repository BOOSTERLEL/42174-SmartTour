"""
Xiaohongshu popularity cache models.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Text
from sqlmodel import Field, SQLModel


class XhsDiscovery(SQLModel):
    """
    Categorized Xiaohongshu discovery results for a destination.
    """

    destination: str
    hints_by_category: dict[str, list[tuple[str, float]]] = Field(default_factory=dict)
    popularity: dict[str, float] = Field(default_factory=dict)
    place_notes: dict[str, list[str]] = Field(default_factory=dict)


class XhsPopularityRecord(SQLModel, table=True):
    """
    SQLite table for cached Xiaohongshu popularity signals.
    """

    __tablename__ = "xhs_popularity"

    id: str = Field(primary_key=True)
    destination: str = Field(index=True)
    popularity_json: str = Field(
        default="{}",
        sa_column=Column(Text, nullable=False, default="{}"),
    )
    hints_json: str = Field(
        default="{}",
        sa_column=Column(
            Text,
            nullable=False,
            default="{}",
            doc=(
                "JSON object keyed by category, where each value is a list of "
                "[title, score] pairs."
            ),
        ),
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    def to_popularity_map(self) -> dict[str, float]:
        """
        Convert the record into a popularity map.

        Returns:
            A lowercase place-title to popularity-score mapping.
        """

        try:
            raw_data = json.loads(self.popularity_json)
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw_data, dict):
            return {}
        popularity: dict[str, float] = {}
        for title, score in raw_data.items():
            if not isinstance(title, str):
                continue
            try:
                popularity[title] = float(score)
            except (TypeError, ValueError):
                continue
        return popularity

    def to_discovery(self) -> XhsDiscovery:
        """
        Convert the record into a discovery payload.

        Returns:
            Structured discovery data with categorized hints and popularity.
        """

        try:
            raw_hints = json.loads(self.hints_json)
        except json.JSONDecodeError:
            raw_hints = {}
        hints_by_category: dict[str, list[tuple[str, float]]] = {}
        if isinstance(raw_hints, dict):
            for category, entries in raw_hints.items():
                if not isinstance(category, str) or not isinstance(entries, list):
                    continue
                normalized_entries: list[tuple[str, float]] = []
                for entry in entries:
                    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                        continue
                    title, score = entry
                    if not isinstance(title, str):
                        continue
                    try:
                        normalized_entries.append((title, float(score)))
                    except (TypeError, ValueError):
                        continue
                hints_by_category[category] = normalized_entries
        return XhsDiscovery(
            destination=self.destination,
            hints_by_category=hints_by_category,
            popularity=self.to_popularity_map(),
        )

    @classmethod
    def from_popularity(
        cls,
        destination: str,
        popularity: dict[str, float],
        fetched_at: datetime | None = None,
    ) -> XhsPopularityRecord:
        """
        Create a cache record from a popularity mapping.

        Args:
            destination: Destination name.
            popularity: Popularity mapping to persist.
            fetched_at: Optional explicit cache timestamp.

        Returns:
            A database-ready popularity cache record.
        """

        return cls(
            id=cls.cache_id(destination),
            destination=destination,
            popularity_json=json.dumps(popularity, ensure_ascii=False),
            fetched_at=fetched_at or datetime.now(UTC),
        )

    @classmethod
    def from_discovery(
        cls,
        discovery: XhsDiscovery,
        fetched_at: datetime | None = None,
    ) -> XhsPopularityRecord:
        """
        Create a cache record from a discovery payload.

        Args:
            discovery: Discovery payload to persist.
            fetched_at: Optional explicit cache timestamp.

        Returns:
            A database-ready discovery cache record.
        """

        return cls(
            id=cls.cache_id(discovery.destination),
            destination=discovery.destination,
            popularity_json=json.dumps(discovery.popularity, ensure_ascii=False),
            hints_json=json.dumps(
                {
                    category: [[title, score] for title, score in entries]
                    for category, entries in discovery.hints_by_category.items()
                },
                ensure_ascii=False,
            ),
            fetched_at=fetched_at or datetime.now(UTC),
        )

    @staticmethod
    def cache_id(destination: str) -> str:
        """
        Build a stable cache key for a destination.

        Args:
            destination: Destination name.

        Returns:
            A deterministic SHA-256 cache identifier.
        """

        normalized = destination.strip().lower().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()
