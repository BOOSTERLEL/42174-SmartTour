"""FastAPI application entrypoint for the Smartour backend."""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smartour.api.routes.conversations import router as conversations_router
from smartour.api.routes.google_maps import router as google_maps_router
from smartour.api.routes.health import router as health_router
from smartour.api.routes.itineraries import router as itineraries_router

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
    app = FastAPI(title="Smartour API", version="0.1.0")
    _configure_cors(app)
    app.include_router(health_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
    app.include_router(itineraries_router, prefix="/api")
    app.include_router(google_maps_router, prefix="/api")
    return app


def run() -> None:
    """
    Run the local Smartour API server.
    """
    uvicorn.run("smartour.main:app", host="127.0.0.1", port=8000, reload=False)


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


app = create_app()
