"""
Itinerary router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from smarttour.api.dependencies import get_db_session, get_itinerary_generator
from smarttour.models import (
    ItineraryGenerateRequest,
    ItineraryRegenerateRequest,
    PlanningSession,
)
from smarttour.services import ItineraryGenerator

router = APIRouter(prefix="/api/itinerary", tags=["itinerary"])


@router.post("/generate", response_model=PlanningSession)
def generate_itinerary(
    payload: ItineraryGenerateRequest,
    session: Annotated[Session, Depends(get_db_session)],
    generator: Annotated[ItineraryGenerator, Depends(get_itinerary_generator)],
) -> PlanningSession:
    """
    Generate a multi-day itinerary.

    Args:
        payload: Generation payload.
        session: Database session.
        generator: Itinerary generator dependency.

    Returns:
        The saved planning session.
    """

    try:
        return generator.generate(session, payload.preferences, payload.user_input)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/{session_id}", response_model=PlanningSession)
def get_itinerary(
    session_id: str,
    session: Annotated[Session, Depends(get_db_session)],
    generator: Annotated[ItineraryGenerator, Depends(get_itinerary_generator)],
) -> PlanningSession:
    """
    Retrieve a generated itinerary session.

    Args:
        session_id: Saved session identifier.
        session: Database session.
        generator: Itinerary generator dependency.

    Returns:
        The saved planning session.
    """

    planning_session = generator.get_session(session, session_id)
    if planning_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return planning_session


@router.post("/regenerate", response_model=PlanningSession)
def regenerate_itinerary(
    payload: ItineraryRegenerateRequest,
    session: Annotated[Session, Depends(get_db_session)],
    generator: Annotated[ItineraryGenerator, Depends(get_itinerary_generator)],
) -> PlanningSession:
    """
    Regenerate an itinerary linked to a previous session with incremented version.

    Args:
        payload: Regeneration payload containing previous session_id and updated preferences.
        session: Database session.
        generator: Itinerary generator dependency.

    Returns:
        The new planning session with incremented version.
    """

    try:
        return generator.regenerate(
            session, payload.session_id, payload.preferences, payload.user_input
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
