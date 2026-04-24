"""Itinerary API routes."""

from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from smartour.api.dependencies import (
    get_google_maps_client,
    get_itinerary_job_service,
    get_planning_service,
    get_settings,
)
from smartour.application.itinerary_job_service import ItineraryJobService
from smartour.application.planning_service import PlanningService
from smartour.core.config import Settings
from smartour.core.errors import ExternalServiceError, PlanningInputError
from smartour.domain.itinerary import Itinerary
from smartour.domain.itinerary_job import ItineraryJob
from smartour.integrations.google_maps.client import (
    GoogleMapsClient,
    create_google_maps_client,
)

router = APIRouter(tags=["itineraries"])


@router.post(
    "/conversations/{conversation_id}/itinerary",
    response_model=Itinerary,
    status_code=status.HTTP_201_CREATED,
)
async def generate_itinerary(
    conversation_id: str,
    planning_service: Annotated[PlanningService, Depends(get_planning_service)],
    google_maps_client: Annotated[GoogleMapsClient, Depends(get_google_maps_client)],
) -> Itinerary:
    """
    Generate an itinerary for a confirmed conversation.

    Args:
        conversation_id: The source conversation ID.
        planning_service: The itinerary planning service.
        google_maps_client: The Google Maps client group.

    Returns:
        The generated itinerary.
    """
    try:
        itinerary = await planning_service.generate_for_conversation(
            conversation_id, google_maps_client
        )
    except PlanningInputError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error
    except ExternalServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)
        ) from error
    if itinerary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return itinerary


@router.post(
    "/conversations/{conversation_id}/itinerary-jobs",
    response_model=ItineraryJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_itinerary_job(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    job_service: Annotated[ItineraryJobService, Depends(get_itinerary_job_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ItineraryJob:
    """
    Queue an itinerary generation job for a confirmed conversation.

    Args:
        conversation_id: The source conversation ID.
        background_tasks: The FastAPI background task manager.
        job_service: The itinerary job service.
        settings: Runtime application settings.

    Returns:
        The queued itinerary job.
    """
    try:
        job = job_service.create_job(conversation_id)
    except PlanningInputError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    background_tasks.add_task(_run_itinerary_job, job.id, job_service, settings)
    return job


@router.get("/itinerary-jobs/{job_id}", response_model=ItineraryJob)
async def get_itinerary_job(
    job_id: str,
    job_service: Annotated[ItineraryJobService, Depends(get_itinerary_job_service)],
) -> ItineraryJob:
    """
    Return an itinerary generation job by ID.

    Args:
        job_id: The itinerary job ID.
        job_service: The itinerary job service.

    Returns:
        The itinerary job.
    """
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Itinerary job not found"
        )
    return job


@router.get("/itineraries/{itinerary_id}", response_model=Itinerary)
async def get_itinerary(
    itinerary_id: str,
    planning_service: Annotated[PlanningService, Depends(get_planning_service)],
) -> Itinerary:
    """
    Return a generated itinerary by ID.

    Args:
        itinerary_id: The itinerary ID.
        planning_service: The itinerary planning service.

    Returns:
        The generated itinerary.
    """
    itinerary = planning_service.get_itinerary(itinerary_id)
    if itinerary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Itinerary not found"
        )
    return itinerary


async def _run_itinerary_job(
    job_id: str,
    job_service: ItineraryJobService,
    settings: Settings,
) -> None:
    """
    Run a queued itinerary job with a fresh Google Maps HTTP client.

    Args:
        job_id: The itinerary job ID.
        job_service: The itinerary job service.
        settings: Runtime application settings.
    """
    timeout = httpx.Timeout(settings.google_maps_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        google_maps_client = create_google_maps_client(
            settings.google_maps_api_key, http_client
        )
        await job_service.run_job(job_id, google_maps_client)
