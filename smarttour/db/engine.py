"""
Database engine and session management.
"""

from collections.abc import Generator
from functools import lru_cache

from sqlmodel import Session, SQLModel, create_engine

from smarttour.config import get_settings
from smarttour.models import (
    AccommodationRecord,
    AttractionRecord,
    DistanceCacheRecord,
    PlanningSessionRecord,
)

REGISTERED_TABLE_MODELS = (
    AccommodationRecord,
    AttractionRecord,
    DistanceCacheRecord,
    PlanningSessionRecord,
)


def _sqlite_connect_args(database_url: str) -> dict[str, bool]:
    """
    Return SQLite-specific connection arguments when applicable.

    Args:
        database_url: The configured database URL.

    Returns:
        Connection arguments for `create_engine`.
    """

    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


@lru_cache(maxsize=8)
def get_engine(database_url: str | None = None):
    """
    Create or return a cached SQLModel engine.

    Args:
        database_url: Optional explicit database URL.

    Returns:
        A SQLModel engine instance.
    """

    resolved_url = database_url or get_settings().database_url
    return create_engine(resolved_url, connect_args=_sqlite_connect_args(resolved_url))


def create_db_and_tables(database_url: str | None = None) -> None:
    """
    Create all configured database tables.

    Args:
        database_url: Optional database URL override.
    """

    _ = REGISTERED_TABLE_MODELS
    engine = get_engine(database_url)
    SQLModel.metadata.create_all(engine)


def get_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """
    Yield a SQLModel session.

    Args:
        database_url: Optional database URL override.

    Yields:
        An active SQLModel session.
    """

    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session


def reset_engine_cache() -> None:
    """
    Clear cached engines for tests.
    """

    get_engine.cache_clear()
