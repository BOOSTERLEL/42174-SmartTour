"""Tests for local user authentication."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException, Response, status
from starlette.requests import Request

from smartour.api.routes.auth import (
    AuthCredentialsRequest,
    get_auth_state,
    login_auth_user,
    logout_auth_user,
    register_auth_user,
)
from smartour.application.auth_service import (
    SESSION_COOKIE_NAME,
    AuthConflictError,
    AuthInvalidCredentialsError,
    AuthService,
    AuthValidationError,
)
from smartour.domain.user import UserSession
from smartour.infrastructure.repositories.users import InMemoryUserRepository


@pytest.mark.asyncio
async def test_auth_service_registers_user_with_hashed_password() -> None:
    """
    Verify that registration stores a salted hash and creates a session.
    """
    repository = InMemoryUserRepository()
    service = AuthService(repository, admin_usernames={"admin"})

    authenticated_session = await service.register("admin", "password123")

    saved_user = await repository.get_user_by_username("ADMIN")
    assert saved_user is not None
    assert saved_user.username == "admin"
    assert saved_user.password_hash != "password123"
    assert saved_user.password_salt
    assert saved_user.is_admin
    assert authenticated_session.session.user_id == saved_user.id
    assert (
        await service.get_user_for_session(authenticated_session.session.token)
    ) == saved_user


@pytest.mark.asyncio
async def test_auth_service_rejects_duplicate_usernames() -> None:
    """
    Verify that username uniqueness uses normalized usernames.
    """
    service = AuthService(InMemoryUserRepository(), admin_usernames=set())

    await service.register("Traveler", "password123")

    with pytest.raises(AuthConflictError):
        await service.register(" traveler ", "password123")


@pytest.mark.asyncio
async def test_auth_service_rejects_invalid_registration_input() -> None:
    """
    Verify that invalid usernames and short passwords are rejected.
    """
    service = AuthService(InMemoryUserRepository(), admin_usernames=set())

    with pytest.raises(AuthValidationError):
        await service.register("ab", "password123")
    with pytest.raises(AuthValidationError):
        await service.register("traveler", "short")


@pytest.mark.asyncio
async def test_auth_service_login_rejects_wrong_password() -> None:
    """
    Verify that invalid credentials do not create a new session.
    """
    service = AuthService(InMemoryUserRepository(), admin_usernames=set())
    await service.register("traveler", "password123")

    with pytest.raises(AuthInvalidCredentialsError):
        await service.login("traveler", "wrong-password")


@pytest.mark.asyncio
async def test_auth_service_logout_and_expiry_remove_sessions() -> None:
    """
    Verify that logout and expired sessions clear authentication.
    """
    repository = InMemoryUserRepository()
    service = AuthService(repository, admin_usernames=set())
    authenticated_session = await service.register("traveler", "password123")

    await service.logout(authenticated_session.session.token)

    assert (
        await service.get_user_for_session(authenticated_session.session.token) is None
    )

    user = authenticated_session.user
    expired_session = UserSession(
        user_id=user.id,
        expires_at=datetime.now(tz=UTC) - timedelta(seconds=1),
    )
    await repository.save_session(expired_session)

    assert await service.get_user_for_session(expired_session.token) is None
    assert await repository.get_session(expired_session.token) is None


@pytest.mark.asyncio
async def test_auth_routes_set_and_clear_session_cookie() -> None:
    """
    Verify that auth routes expose cookie-backed browser state.
    """
    service = AuthService(InMemoryUserRepository(), admin_usernames={"admin"})
    register_response = Response()

    registered_state = await register_auth_user(
        AuthCredentialsRequest(username="admin", password="password123"),
        register_response,
        service,
    )

    assert registered_state.authenticated
    assert registered_state.user is not None
    assert registered_state.user.is_admin
    register_cookie = register_response.headers["set-cookie"]
    assert f"{SESSION_COOKIE_NAME}=" in register_cookie
    assert "HttpOnly" in register_cookie

    login_response = Response()
    logged_in_state = await login_auth_user(
        AuthCredentialsRequest(username="admin", password="password123"),
        login_response,
        service,
    )
    assert logged_in_state.authenticated

    token = _session_cookie_value(login_response)
    current_state = await get_auth_state(_request_with_session(token), service)
    assert current_state.authenticated
    assert current_state.user is not None
    assert current_state.user.username == "admin"

    logout_response = Response()
    logged_out_state = await logout_auth_user(
        _request_with_session(token),
        logout_response,
        service,
    )
    assert not logged_out_state.authenticated
    assert "Max-Age=0" in logout_response.headers["set-cookie"]
    assert not (
        await get_auth_state(_request_with_session(token), service)
    ).authenticated


@pytest.mark.asyncio
async def test_login_route_returns_unauthorized_for_wrong_password() -> None:
    """
    Verify that the login route maps invalid credentials to 401.
    """
    service = AuthService(InMemoryUserRepository(), admin_usernames=set())
    await service.register("traveler", "password123")

    with pytest.raises(HTTPException) as error_info:
        await login_auth_user(
            AuthCredentialsRequest(username="traveler", password="wrong-password"),
            Response(),
            service,
        )

    assert error_info.value.status_code == status.HTTP_401_UNAUTHORIZED


def _request_with_session(token: str) -> Request:
    """
    Create a Starlette request with a session cookie.

    Args:
        token: The session token.

    Returns:
        The request.
    """
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/auth/me",
            "headers": [(b"cookie", f"{SESSION_COOKIE_NAME}={token}".encode())],
        }
    )


def _session_cookie_value(response: Response) -> str:
    """
    Extract the session cookie value from a response.

    Args:
        response: The response with a set-cookie header.

    Returns:
        The session cookie value.
    """
    cookie_header = response.headers["set-cookie"]
    first_cookie_part = cookie_header.split(";", maxsplit=1)[0]
    return first_cookie_part.split("=", maxsplit=1)[1]
