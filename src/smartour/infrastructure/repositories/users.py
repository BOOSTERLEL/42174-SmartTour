"""User repository implementations."""

from datetime import datetime
from typing import Any

from smartour.domain.user import User, UserSession, normalize_username
from smartour.infrastructure.database import SQLiteDatabase


class InMemoryUserRepository:
    """
    Process-local in-memory user repository.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """
        self.users_by_id: dict[str, User] = {}
        self.users_by_normalized_username: dict[str, User] = {}
        self.sessions_by_token: dict[str, UserSession] = {}

    async def save_user(self, user: User) -> None:
        """
        Save a user account.

        Args:
            user: The user to save.
        """
        user_copy = user.model_copy(deep=True)
        self.users_by_id[user.id] = user_copy
        self.users_by_normalized_username[user.normalized_username] = user_copy

    async def get_user_by_id(self, user_id: str) -> User | None:
        """
        Return a user by ID.

        Args:
            user_id: The user ID.

        Returns:
            The user when found.
        """
        user = self.users_by_id.get(user_id)
        if user is None:
            return None
        return user.model_copy(deep=True)

    async def get_user_by_username(self, username: str) -> User | None:
        """
        Return a user by username.

        Args:
            username: The raw username.

        Returns:
            The user when found.
        """
        user = self.users_by_normalized_username.get(normalize_username(username))
        if user is None:
            return None
        return user.model_copy(deep=True)

    async def save_session(self, session: UserSession) -> None:
        """
        Save a user session.

        Args:
            session: The session to save.
        """
        self.sessions_by_token[session.token] = session.model_copy(deep=True)

    async def get_session(self, token: str) -> UserSession | None:
        """
        Return a session by token.

        Args:
            token: The session token.

        Returns:
            The session when found.
        """
        session = self.sessions_by_token.get(token)
        if session is None:
            return None
        return session.model_copy(deep=True)

    async def delete_session(self, token: str) -> None:
        """
        Delete a session by token.

        Args:
            token: The session token.
        """
        self.sessions_by_token.pop(token, None)


class SQLiteUserRepository:
    """
    SQLite-backed user repository.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        """
        Initialize the repository.

        Args:
            database: The SQLite database.
        """
        self.database = database

    async def save_user(self, user: User) -> None:
        """
        Save a user account.

        Args:
            user: The user to save.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO users (
                    id, username, normalized_username, password_hash,
                    password_salt, is_admin, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    username = excluded.username,
                    normalized_username = excluded.normalized_username,
                    password_hash = excluded.password_hash,
                    password_salt = excluded.password_salt,
                    is_admin = excluded.is_admin,
                    updated_at = excluded.updated_at
                """,
                (
                    user.id,
                    user.username,
                    user.normalized_username,
                    user.password_hash,
                    user.password_salt,
                    int(user.is_admin),
                    user.created_at.isoformat(),
                    user.updated_at.isoformat(),
                ),
            )

    async def get_user_by_id(self, user_id: str) -> User | None:
        """
        Return a user by ID.

        Args:
            user_id: The user ID.

        Returns:
            The user when found.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_user(row)

    async def get_user_by_username(self, username: str) -> User | None:
        """
        Return a user by username.

        Args:
            username: The raw username.

        Returns:
            The user when found.
        """
        normalized_username = normalize_username(username)
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT * FROM users WHERE normalized_username = ?",
                (normalized_username,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_user(row)

    async def save_session(self, session: UserSession) -> None:
        """
        Save a user session.

        Args:
            session: The session to save.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO user_sessions (token, user_id, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET
                    user_id = excluded.user_id,
                    expires_at = excluded.expires_at
                """,
                (
                    session.token,
                    session.user_id,
                    session.expires_at.isoformat(),
                    session.created_at.isoformat(),
                ),
            )

    async def get_session(self, token: str) -> UserSession | None:
        """
        Return a session by token.

        Args:
            token: The session token.

        Returns:
            The session when found.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT * FROM user_sessions WHERE token = ?",
                (token,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_session(row)

    async def delete_session(self, token: str) -> None:
        """
        Delete a session by token.

        Args:
            token: The session token.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                "DELETE FROM user_sessions WHERE token = ?",
                (token,),
            )


def _row_to_user(row: Any) -> User:
    """
    Convert a SQLite row into a user.

    Args:
        row: The SQLite row.

    Returns:
        The user.
    """
    return User(
        id=row["id"],
        username=row["username"],
        normalized_username=row["normalized_username"],
        password_hash=row["password_hash"],
        password_salt=row["password_salt"],
        is_admin=bool(row["is_admin"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_session(row: Any) -> UserSession:
    """
    Convert a SQLite row into a user session.

    Args:
        row: The SQLite row.

    Returns:
        The user session.
    """
    return UserSession(
        token=row["token"],
        user_id=row["user_id"],
        expires_at=datetime.fromisoformat(row["expires_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
