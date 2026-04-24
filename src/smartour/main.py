"""FastAPI application entrypoint for the Smartour backend."""

import uvicorn
from fastapi import FastAPI

from smartour.api.routes.conversations import router as conversations_router
from smartour.api.routes.google_maps import router as google_maps_router
from smartour.api.routes.health import router as health_router


def create_app() -> FastAPI:
    """
    Create the Smartour FastAPI application.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(title="Smartour API", version="0.1.0")
    app.include_router(health_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
    app.include_router(google_maps_router, prefix="/api")
    return app


def run() -> None:
    """
    Run the local Smartour API server.
    """
    uvicorn.run("smartour.main:app", host="127.0.0.1", port=8000, reload=False)


app = create_app()
