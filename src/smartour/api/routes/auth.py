"""Authentication API routes."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from smartour.api.dependencies import get_auth_service
from smartour.application.auth_service import (
    SESSION_COOKIE_NAME,
    AuthConflictError,
    AuthenticatedSession,
    AuthInvalidCredentialsError,
    AuthService,
    AuthValidationError,
)
from smartour.domain.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthCredentialsRequest(BaseModel):
    """
    Request model for username/password authentication.
    """

    username: str
    password: str


class AuthUserResponse(BaseModel):
    """
    Public response model for an authenticated user.
    """

    user_id: str
    username: str
    is_admin: bool


class AuthStateResponse(BaseModel):
    """
    Response model for browser authentication state.
    """

    authenticated: bool
    user: AuthUserResponse | None = None


@router.post(
    "/register",
    response_model=AuthStateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_auth_user(
    request: AuthCredentialsRequest,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthStateResponse:
    """
    Register a local user and start a browser session.

    Args:
        request: The registration request.
        response: The HTTP response used to set cookies.
        auth_service: The authentication service.

    Returns:
        The authenticated browser state.
    """
    try:
        authenticated_session = await auth_service.register(
            request.username, request.password
        )
    except AuthValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except AuthConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error
    _set_session_cookie(response, authenticated_session)
    return _auth_state_response(authenticated_session.user)


@router.post("/login", response_model=AuthStateResponse)
async def login_auth_user(
    request: AuthCredentialsRequest,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthStateResponse:
    """
    Login a local user and start a browser session.

    Args:
        request: The login request.
        response: The HTTP response used to set cookies.
        auth_service: The authentication service.

    Returns:
        The authenticated browser state.
    """
    try:
        authenticated_session = await auth_service.login(
            request.username, request.password
        )
    except AuthInvalidCredentialsError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)
        ) from error
    _set_session_cookie(response, authenticated_session)
    return _auth_state_response(authenticated_session.user)


@router.get("/me", response_model=AuthStateResponse)
async def get_auth_state(
    request: Request,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthStateResponse:
    """
    Return the current browser authentication state.

    Args:
        request: The inbound HTTP request.
        auth_service: The authentication service.

    Returns:
        The current browser authentication state.
    """
    user = await auth_service.get_user_for_session(
        request.cookies.get(SESSION_COOKIE_NAME)
    )
    return _auth_state_response(user)


@router.post("/logout", response_model=AuthStateResponse)
async def logout_auth_user(
    request: Request,
    response: Response,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> AuthStateResponse:
    """
    Logout the current browser session.

    Args:
        request: The inbound HTTP request.
        response: The HTTP response used to clear cookies.
        auth_service: The authentication service.

    Returns:
        An unauthenticated browser state.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        await auth_service.logout(token)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/", samesite="lax")
    return AuthStateResponse(authenticated=False, user=None)


def _set_session_cookie(
    response: Response, authenticated_session: AuthenticatedSession
) -> None:
    """
    Attach an HttpOnly session cookie to the response.

    Args:
        response: The HTTP response used to set cookies.
        authenticated_session: The authenticated user session.
    """
    max_age = int(
        (
            authenticated_session.session.expires_at - datetime.now(tz=UTC)
        ).total_seconds()
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=authenticated_session.session.token,
        max_age=max(0, max_age),
        httponly=True,
        samesite="lax",
        path="/",
    )


def _auth_state_response(user: User | None) -> AuthStateResponse:
    """
    Convert a domain user to an auth state response.

    Args:
        user: The authenticated user when available.

    Returns:
        The authentication state response.
    """
    if user is None:
        return AuthStateResponse(authenticated=False, user=None)
    return AuthStateResponse(
        authenticated=True,
        user=AuthUserResponse(
            user_id=user.id,
            username=user.username,
            is_admin=user.is_admin,
        ),
    )
