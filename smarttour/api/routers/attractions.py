"""
Attraction retrieval router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from smarttour.api.dependencies import get_data_retrieval_service, get_db_session
from smarttour.models import Attraction, AttractionSearchResponse
from smarttour.services import DataRetrievalService

router = APIRouter(prefix="/api/attractions", tags=["attractions"])


@router.get("/search", response_model=AttractionSearchResponse)
def search_attractions(
    destination: str,
    session: Annotated[Session, Depends(get_db_session)],
    service: Annotated[DataRetrievalService, Depends(get_data_retrieval_service)],
    categories: Annotated[list[str] | None, Query()] = None,
    max_cost: float | None = None,
    limit: int = 12,
) -> AttractionSearchResponse:
    """
    Search attractions by destination and optional filters.

    Args:
        destination: Destination name.
        session: Database session.
        service: Retrieval service.
        categories: Optional category list.
        max_cost: Optional attraction cost ceiling.
        limit: Maximum result count.

    Returns:
        Search results and total count.
    """

    results = service.search_attractions(
        session, destination, categories, max_cost, limit
    )
    return AttractionSearchResponse(results=results, total=len(results))


@router.get("/{attraction_id}", response_model=Attraction)
def get_attraction(
    attraction_id: str,
    session: Annotated[Session, Depends(get_db_session)],
    service: Annotated[DataRetrievalService, Depends(get_data_retrieval_service)],
) -> Attraction:
    """
    Retrieve one attraction.

    Args:
        attraction_id: Attraction identifier.
        session: Database session.
        service: Retrieval service.

    Returns:
        The requested attraction.
    """

    attraction = service.get_attraction(session, attraction_id)
    if attraction is None:
        raise HTTPException(status_code=404, detail="Attraction not found")
    return attraction
