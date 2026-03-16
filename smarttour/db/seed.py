"""
Database seeding utilities.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import Session, select

from smarttour.config import DATA_DIR
from smarttour.db.engine import create_db_and_tables, get_engine
from smarttour.models import (
    AccommodationRecord,
    Attraction,
    AttractionRecord,
    Hotel,
    Restaurant,
    RestaurantRecord,
)

ATTRACTIONS_PATH = DATA_DIR / "attractions.json"
HOTELS_PATH = DATA_DIR / "hotels.json"
RESTAURANTS_PATH = DATA_DIR / "restaurants.json"


def _load_json(path: Path) -> list[dict[str, object]]:
    """
    Load a JSON array from disk.

    Args:
        path: The path to read.

    Returns:
        The parsed JSON array.
    """

    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def seed_database(force: bool = False) -> None:
    """
    Populate the database from local JSON files.

    Args:
        force: Whether to replace existing seed data.
    """

    create_db_and_tables()
    engine = get_engine()
    attractions_payload = _load_json(ATTRACTIONS_PATH)
    hotels_payload = _load_json(HOTELS_PATH)
    restaurants_payload = _load_json(RESTAURANTS_PATH)
    with Session(engine) as session:
        has_attractions = session.exec(select(AttractionRecord)).first() is not None
        has_hotels = session.exec(select(AccommodationRecord)).first() is not None
        has_restaurants = session.exec(select(RestaurantRecord)).first() is not None
        if force:
            for attraction_record in session.exec(select(AttractionRecord)).all():
                session.delete(attraction_record)
            for accommodation_record in session.exec(select(AccommodationRecord)).all():
                session.delete(accommodation_record)
            for restaurant_record in session.exec(select(RestaurantRecord)).all():
                session.delete(restaurant_record)
            session.commit()
            has_attractions = False
            has_hotels = False
            has_restaurants = False
        if not has_attractions:
            for payload in attractions_payload:
                attraction = Attraction.model_validate(payload)
                session.add(AttractionRecord.from_model(attraction))
        if not has_hotels:
            for payload in hotels_payload:
                hotel = Hotel.model_validate(payload)
                session.add(AccommodationRecord.from_model(hotel))
        if not has_restaurants:
            for payload in restaurants_payload:
                restaurant = Restaurant.model_validate(payload)
                session.add(RestaurantRecord.from_model(restaurant))
        session.commit()


def main() -> None:
    """
    CLI entrypoint for seeding the database.
    """

    seed_database()


if __name__ == "__main__":
    main()
