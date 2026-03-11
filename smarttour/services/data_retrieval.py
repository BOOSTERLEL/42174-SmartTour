"""
Database-backed attraction and accommodation retrieval.
"""

from __future__ import annotations

from sqlmodel import Session, select

from smarttour.models import AccommodationRecord, Attraction, AttractionRecord, Hotel


class DataRetrievalService:
    """
    Retrieve attractions and accommodations from SQLite.
    """

    def search_attractions(
        self,
        session: Session,
        destination: str,
        categories: list[str] | None = None,
        max_cost: float | None = None,
        limit: int = 12,
    ) -> list[Attraction]:
        """
        Search attractions using destination and optional filters.

        Args:
            session: Active database session.
            destination: Destination name.
            categories: Optional desired categories or tags.
            max_cost: Optional cost ceiling per attraction.
            limit: Maximum number of results.

        Returns:
            Matching attractions ordered by quality.
        """

        statement = select(AttractionRecord).where(
            AttractionRecord.destination == destination
        )
        records = session.exec(statement).all()
        attractions = [record.to_model() for record in records]
        lowered_categories = {item.lower() for item in categories or []}
        if lowered_categories:
            attractions = [
                attraction
                for attraction in attractions
                if attraction.category.lower() in lowered_categories
                or bool(
                    lowered_categories.intersection(
                        {tag.lower() for tag in attraction.tags}
                    )
                )
            ]
        if max_cost is not None:
            attractions = [
                attraction for attraction in attractions if attraction.cost <= max_cost
            ]
        attractions.sort(key=lambda item: (-item.rating, item.cost, item.name))
        return attractions[:limit]

    def get_attraction(self, session: Session, attraction_id: str) -> Attraction | None:
        """
        Retrieve a single attraction by identifier.

        Args:
            session: Active database session.
            attraction_id: Attraction primary key.

        Returns:
            The matching attraction or `None`.
        """

        record = session.get(AttractionRecord, attraction_id)
        if record is None:
            return None
        return record.to_model()

    def list_accommodations(
        self,
        session: Session,
        destination: str,
        nightly_budget: float | None = None,
        limit: int = 5,
    ) -> list[Hotel]:
        """
        Retrieve accommodation options.

        Args:
            session: Active database session.
            destination: Destination name.
            nightly_budget: Optional budget ceiling per night.
            limit: Maximum number of options.

        Returns:
            Matching hotel options.
        """

        statement = select(AccommodationRecord).where(
            AccommodationRecord.destination == destination
        )
        records = session.exec(statement).all()
        hotels = [record.to_model() for record in records]
        if nightly_budget is not None and nightly_budget > 0:
            hotels = [
                hotel for hotel in hotels if hotel.price_per_night <= nightly_budget
            ]
        hotels.sort(
            key=lambda item: (-item.star_rating, item.price_per_night, item.name)
        )
        return hotels[:limit]
