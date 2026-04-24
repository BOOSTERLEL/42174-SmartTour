"""SQLite database utilities for local Smartour persistence."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS itineraries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS itinerary_jobs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS google_api_cache_entries (
    cache_key TEXT PRIMARY KEY,
    service TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    field_mask TEXT,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS google_api_request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    status_code INTEGER,
    duration_ms REAL NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    event_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_google_api_cache_expires_at
ON google_api_cache_entries(expires_at);

CREATE INDEX IF NOT EXISTS idx_google_api_metrics_created_at
ON google_api_request_metrics(created_at);

CREATE INDEX IF NOT EXISTS idx_rate_limit_events_lookup
ON rate_limit_events(scope, subject_key, event_name, created_at);
"""


class SQLiteDatabase:
    """
    Owns a local SQLite database path and schema initialization.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the SQLite database.

        Args:
            path: The SQLite database file path.
        """
        self.path = Path(path)
        self.is_initialized = False
        self.initialize_lock = asyncio.Lock()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Open a SQLite connection with row dictionaries enabled.

        Yields:
            A SQLite connection.
        """
        await self.initialize()
        connection = await aiosqlite.connect(str(self.path))
        connection.row_factory = aiosqlite.Row
        try:
            yield connection
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        finally:
            await connection.close()

    async def initialize(self) -> None:
        """
        Create all required tables and indexes if they are missing.
        """
        if self.is_initialized:
            return
        async with self.initialize_lock:
            if self.is_initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(str(self.path)) as connection:
                await connection.executescript(SCHEMA_SQL)
                await connection.commit()
            self.is_initialized = True
