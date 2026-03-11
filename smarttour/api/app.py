"""
FastAPI application entrypoint.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from smarttour.api.routers import (
    attractions,
    guidance,
    health,
    itinerary,
    preferences,
    routes,
)
from smarttour.config import get_settings
from smarttour.db import create_db_and_tables
from smarttour.db.seed import seed_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize tables and seed data on application startup.

    Args:
        app: The FastAPI application.

    Yields:
        Control back to FastAPI after startup work completes.
    """

    _ = app
    create_db_and_tables()
    seed_database()
    yield


def create_app() -> FastAPI:
    """
    Build the FastAPI application.

    Returns:
        A configured FastAPI app instance.
    """

    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(health.router)
    application.include_router(preferences.router)
    application.include_router(itinerary.router)
    application.include_router(attractions.router)
    application.include_router(routes.router)
    application.include_router(guidance.router)
    return application


app = create_app()
