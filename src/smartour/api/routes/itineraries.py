"""Itinerary API routes."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from smartour.api.dependencies import (
    get_google_api_store,
    get_google_maps_client,
    get_itinerary_job_service,
    get_planning_service,
    get_settings,
)
from smartour.application.itinerary_job_service import ItineraryJobService
from smartour.application.planning_service import PlanningService
from smartour.core.config import Settings
from smartour.core.errors import (
    ExternalServiceError,
    PlanningInputError,
    RateLimitError,
)
from smartour.domain.itinerary import Itinerary
from smartour.domain.itinerary_job import ItineraryJob, ItineraryJobStatus
from smartour.infrastructure.google_api_store import SQLiteGoogleApiStore
from smartour.integrations.google_maps.client import (
    GoogleMapsClient,
    create_google_maps_client,
)

router = APIRouter(tags=["itineraries"])
JOB_EVENT_POLL_SECONDS = 1.0
JOB_EVENT_MAX_POLLS = 300
TERMINAL_JOB_STATUSES = {
    ItineraryJobStatus.SUCCEEDED,
    ItineraryJobStatus.FAILED,
}


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
    request: Request,
    background_tasks: BackgroundTasks,
    job_service: Annotated[ItineraryJobService, Depends(get_itinerary_job_service)],
    api_store: Annotated[SQLiteGoogleApiStore, Depends(get_google_api_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ItineraryJob:
    """
    Queue an itinerary generation job for a confirmed conversation.

    Args:
        conversation_id: The source conversation ID.
        request: The inbound HTTP request.
        background_tasks: The FastAPI background task manager.
        job_service: The itinerary job service.
        api_store: The Google API cache and metrics store.
        settings: Runtime application settings.

    Returns:
        The queued itinerary job.
    """
    try:
        client_host = request.client.host if request.client else None
        job = await job_service.create_job(conversation_id, client_host)
    except PlanningInputError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error
    except RateLimitError as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(error)
        ) from error
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    background_tasks.add_task(
        _run_itinerary_job, job.id, job_service, api_store, settings
    )
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
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Itinerary job not found"
        )
    return job


@router.get("/itinerary-jobs/{job_id}/events")
async def stream_itinerary_job_events(
    job_id: str,
    job_service: Annotated[ItineraryJobService, Depends(get_itinerary_job_service)],
) -> StreamingResponse:
    """
    Stream itinerary job status updates as server-sent events.

    Args:
        job_id: The itinerary job ID.
        job_service: The itinerary job service.

    Returns:
        A text/event-stream response with job status payloads.
    """
    job = await job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Itinerary job not found"
        )
    return StreamingResponse(
        _job_event_stream(job.id, job_service),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


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
    itinerary = await planning_service.get_itinerary(itinerary_id)
    if itinerary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Itinerary not found"
        )
    return itinerary


async def _run_itinerary_job(
    job_id: str,
    job_service: ItineraryJobService,
    api_store: SQLiteGoogleApiStore,
    settings: Settings,
) -> None:
    """
    Run a queued itinerary job with a fresh Google Maps HTTP client.

    Args:
        job_id: The itinerary job ID.
        job_service: The itinerary job service.
        api_store: The Google API cache and metrics store.
        settings: Runtime application settings.
    """
    timeout = httpx.Timeout(settings.google_maps_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        google_maps_client = create_google_maps_client(
            settings.google_maps_api_key,
            http_client,
            api_store=api_store,
            default_cache_ttl_seconds=settings.google_maps_cache_ttl_seconds,
            routes_cache_ttl_seconds=settings.google_maps_routes_cache_ttl_seconds,
        )
        await job_service.run_job(job_id, google_maps_client)


async def _job_event_stream(
    job_id: str, job_service: ItineraryJobService
) -> AsyncIterator[str]:
    """
    Yield itinerary job status updates as SSE messages.

    Args:
        job_id: The itinerary job ID.
        job_service: The itinerary job service.

    Yields:
        Formatted server-sent event messages.
    """
    previous_payload: dict[str, object] | None = None
    poll_count = 0
    while poll_count < JOB_EVENT_MAX_POLLS:
        job = await job_service.get_job(job_id)
        if job is None:
            yield _format_sse_event(
                "itinerary_job_error",
                {
                    "job_id": job_id,
                    "error_message": "Itinerary job not found",
                },
            )
            return
        payload = _job_event_payload(job)
        if payload != previous_payload:
            yield _format_sse_event("itinerary_job", payload)
            previous_payload = payload
        if job.status in TERMINAL_JOB_STATUSES:
            return
        poll_count += 1
        await asyncio.sleep(JOB_EVENT_POLL_SECONDS)


def _job_event_payload(job: ItineraryJob) -> dict[str, object]:
    """
    Convert an itinerary job into an SSE payload.

    Args:
        job: The itinerary job.

    Returns:
        A JSON-serializable event payload.
    """
    return {
        "job_id": job.id,
        "conversation_id": job.conversation_id,
        "status": job.status.value,
        "itinerary_id": job.itinerary_id,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def _format_sse_event(event_name: str, payload: dict[str, object]) -> str:
    """
    Format a server-sent event message.

    Args:
        event_name: The SSE event name.
        payload: The event JSON payload.

    Returns:
        A formatted SSE message.
    """
    data = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n"
