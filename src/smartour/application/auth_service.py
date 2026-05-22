"""Authentication service for local Smartour users."""

import hashlib
import hmac
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from secrets import token_bytes
from typing import Any

from pydantic import BaseModel

from smartour.domain.user import User, UserSession, normalize_username

SESSION_COOKIE_NAME = "smartour_session"
DEFAULT_SESSION_TTL_SECONDS = 604800
PASSWORD_HASH_ITERATIONS = 210000
MINIMUM_USERNAME_LENGTH = 3
MAXIMUM_USERNAME_LENGTH = 64
MINIMUM_PASSWORD_LENGTH = 8


class AuthValidationError(Exception):
    """
    Error raised when auth input fails validation.
    """


class AuthConflictError(Exception):
    """
    Error raised when an auth resource already exists.
    """


class AuthInvalidCredentialsError(Exception):
    """
    Error raised when username or password authentication fails.
    """


class AuthenticatedSession(BaseModel):
    """
    A successfully authenticated user and session.
    """

    user: User
    session: UserSession


class AuthService:
    """
    Coordinates local user registration, login, and sessions.
    """

    def __init__(
        self,
        user_repository: Any,
        admin_usernames: Iterable[str],
        session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        """
        Initialize the auth service.

        Args:
            user_repository: The repository used to persist users and sessions.
            admin_usernames: Usernames that should receive admin access.
            session_ttl_seconds: Session lifetime in seconds.
        """
        self.user_repository = user_repository
        self.admin_usernames = {
            normalize_username(username)
            for username in admin_usernames
            if normalize_username(username)
        }
        self.session_ttl_seconds = session_ttl_seconds

    async def register(self, username: str, password: str) -> AuthenticatedSession:
        """
        Register a local user and create a browser session.

        Args:
            username: The requested username.
            password: The raw password.

        Returns:
            The authenticated user and session.

        Raises:
            AuthValidationError: Raised when input is invalid.
            AuthConflictError: Raised when the username already exists.
        """
        self._validate_credentials(username, password)
        normalized_username = normalize_username(username)
        existing_user = await self.user_repository.get_user_by_username(
            normalized_username
        )
        if existing_user is not None:
            raise AuthConflictError("Username already exists")
        salt = token_bytes(16).hex()
        user = User(
            username=username.strip(),
            normalized_username=normalized_username,
            password_hash=_hash_password(password, salt),
            password_salt=salt,
            is_admin=normalized_username in self.admin_usernames,
        )
        await self.user_repository.save_user(user)
        session = await self._create_session(user)
        return AuthenticatedSession(user=user, session=session)

    async def login(self, username: str, password: str) -> AuthenticatedSession:
        """
        Authenticate a local user and create a browser session.

        Args:
            username: The raw username.
            password: The raw password.

        Returns:
            The authenticated user and session.

        Raises:
            AuthInvalidCredentialsError: Raised when credentials do not match.
        """
        user = await self.user_repository.get_user_by_username(username)
        if user is None or not _verify_password(
            password, user.password_salt, user.password_hash
        ):
            raise AuthInvalidCredentialsError("Invalid username or password")
        user = await self._refresh_admin_flag(user)
        session = await self._create_session(user)
        return AuthenticatedSession(user=user, session=session)

    async def logout(self, token: str) -> None:
        """
        Delete a browser session.

        Args:
            token: The session token.
        """
        await self.user_repository.delete_session(token)

    async def get_user_for_session(self, token: str | None) -> User | None:
        """
        Return the user for an active session token.

        Args:
            token: The session token.

        Returns:
            The authenticated user when the session is valid.
        """
        if not token:
            return None
        session = await self.user_repository.get_session(token)
        if session is None:
            return None
        if session.is_expired():
            await self.user_repository.delete_session(token)
            return None
        return await self.user_repository.get_user_by_id(session.user_id)

    async def _create_session(self, user: User) -> UserSession:
        """
        Create and persist a session for a user.

        Args:
            user: The authenticated user.

        Returns:
            The persisted user session.
        """
        expires_at = datetime.now(tz=UTC) + timedelta(seconds=self.session_ttl_seconds)
        session = UserSession(user_id=user.id, expires_at=expires_at)
        await self.user_repository.save_session(session)
        return session

    async def _refresh_admin_flag(self, user: User) -> User:
        """
        Keep admin status synchronized with configured usernames.

        Args:
            user: The authenticated user.

        Returns:
            The updated user.
        """
        should_be_admin = user.normalized_username in self.admin_usernames
        if user.is_admin == should_be_admin:
            return user
        updated_user = user.model_copy(
            update={
                "is_admin": should_be_admin,
                "updated_at": datetime.now(tz=UTC),
            }
        )
        await self.user_repository.save_user(updated_user)
        return updated_user

    def _validate_credentials(self, username: str, password: str) -> None:
        """
        Validate registration credentials.

        Args:
            username: The requested username.
            password: The requested password.

        Raises:
            AuthValidationError: Raised when either field is invalid.
        """
        normalized_username = normalize_username(username)
        if not (
            MINIMUM_USERNAME_LENGTH
            <= len(normalized_username)
            <= MAXIMUM_USERNAME_LENGTH
        ):
            raise AuthValidationError("Username must be between 3 and 64 characters")
        if not all(
            character.isalnum() or character in "._-"
            for character in normalized_username
        ):
            raise AuthValidationError(
                "Username may only contain letters, numbers, dots, dashes, "
                "and underscores"
            )
        if len(password) < MINIMUM_PASSWORD_LENGTH:
            raise AuthValidationError("Password must be at least 8 characters long")


def _hash_password(password: str, salt: str) -> str:
    """
    Hash a password with PBKDF2-HMAC-SHA256.

    Args:
        password: The raw password.
        salt: The hexadecimal salt.

    Returns:
        The hexadecimal password hash.
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_HASH_ITERATIONS,
    ).hex()


def _verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """
    Verify a raw password against a stored password hash.

    Args:
        password: The raw password.
        salt: The hexadecimal salt.
        expected_hash: The expected hexadecimal password hash.

    Returns:
        True when the password matches.
    """
    password_hash = _hash_password(password, salt)
    return hmac.compare_digest(password_hash, expected_hash)
