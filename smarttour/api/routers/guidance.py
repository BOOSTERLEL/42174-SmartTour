"""
Guidance generation router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from smarttour.api.dependencies import get_db_session, get_guidance_generator
from smarttour.models import GuidanceRequest, GuidanceResponse
from smarttour.services import GuidanceGenerator

router = APIRouter(prefix="/api/guidance", tags=["guidance"])


@router.post("/explain", response_model=GuidanceResponse)
def explain_attraction(
    payload: GuidanceRequest,
    session: Annotated[Session, Depends(get_db_session)],
    generator: Annotated[GuidanceGenerator, Depends(get_guidance_generator)],
) -> GuidanceResponse:
    """
    Generate guidance for one attraction.

    Args:
        payload: Guidance request payload.
        session: Database session.
        generator: Guidance service dependency.

    Returns:
        The generated guidance response.
    """

    try:
        return generator.explain(session, payload.attraction_id, payload.preferences)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
