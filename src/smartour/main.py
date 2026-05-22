"""FastAPI application entrypoint for the Smartour backend."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smartour.api.routes.admin import router as admin_router
from smartour.api.routes.auth import router as auth_router
from smartour.api.routes.conversations import router as conversations_router
from smartour.api.routes.costs import router as costs_router
from smartour.api.routes.google_maps import router as google_maps_router
from smartour.api.routes.health import router as health_router
from smartour.api.routes.itineraries import router as itineraries_router
from smartour.api.routes.reports import router as reports_router
from smartour.api.routes.shares import router as shares_router
from smartour.api.routes.users import router as users_router

DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
)


def create_app() -> FastAPI:
    """
    Create the Smartour FastAPI application.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(title="Smartour API", version="0.1.0", lifespan=_lifespan)
    _configure_cors(app)
    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(admin_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
    app.include_router(itineraries_router, prefix="/api")
    app.include_router(reports_router, prefix="/api")
    app.include_router(shares_router, prefix="/api")
    app.include_router(users_router, prefix="/api")
    app.include_router(costs_router, prefix="/api")
    app.include_router(google_maps_router, prefix="/api")
    return app


def run() -> None:
    """
    Run the local Smartour API server.
    """
    _configure_windows_event_loop_policy()
    uvicorn.run("smartour.main:app", host="127.0.0.1", port=8000, reload=False)


def _configure_windows_event_loop_policy() -> None:
    """
    Use the selector event loop for local Windows API serving.
    """
    if sys.platform != "win32":
        return
    event_loop_policy_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if event_loop_policy_factory is None:
        return
    asyncio.set_event_loop_policy(event_loop_policy_factory())


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """
    Configure process-local runtime behavior during application startup.

    Args:
        application: The FastAPI application instance.

    Yields:
        Control back to FastAPI while the application is running.
    """
    _configure_windows_connection_reset_handler()
    yield


def _configure_windows_connection_reset_handler() -> None:
    """
    Suppress benign Windows Proactor disconnect callback noise.
    """
    if sys.platform != "win32":
        return
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        _windows_connection_reset_exception_handler(previous_handler)
    )


def _windows_connection_reset_exception_handler(
    previous_handler: Callable[[asyncio.AbstractEventLoop, dict[str, Any]], object]
    | None,
) -> Callable[[asyncio.AbstractEventLoop, dict[str, Any]], None]:
    """
    Build an asyncio exception handler for Windows connection resets.

    Args:
        previous_handler: The previous asyncio exception handler.

    Returns:
        An asyncio exception handler.
    """

    def handle_exception(
        loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        """
        Handle asyncio loop exceptions.

        Args:
            loop: The running asyncio event loop.
            context: The exception context from asyncio.
        """
        if _is_windows_proactor_connection_reset(context):
            return
        if previous_handler is not None:
            previous_handler(loop, context)
            return
        loop.default_exception_handler(context)

    return handle_exception


def _is_windows_proactor_connection_reset(context: dict[str, Any]) -> bool:
    """
    Return whether an asyncio exception is the known Proactor reset callback.

    Args:
        context: The exception context from asyncio.

    Returns:
        True when the context matches the benign Windows Proactor reset callback.
    """
    exception = context.get("exception")
    handle = context.get("handle")
    message = context.get("message")
    if not isinstance(exception, ConnectionResetError):
        return False
    if getattr(exception, "winerror", None) != 10054:
        return False
    return "Exception in callback" in str(
        message
    ) and "_ProactorBasePipeTransport._call_connection_lost" in str(handle)


def _configure_cors(app: FastAPI) -> None:
    """
    Configure browser access for the local Next.js frontend.

    Args:
        app: The FastAPI application instance.
    """
    allowed_origins = _cors_allowed_origins()
    if not allowed_origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _cors_allowed_origins() -> list[str]:
    """
    Load allowed browser origins from the environment.

    Returns:
        A list of allowed CORS origins.
    """
    raw_origins = os.getenv("SMARTOUR_CORS_ALLOWED_ORIGINS")
    if raw_origins is None:
        return list(DEFAULT_CORS_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


_configure_windows_event_loop_policy()
app = create_app()
