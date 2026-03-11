"""
Route optimization router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session

from smarttour.api.dependencies import (
    get_db_session,
    get_maps_client,
    get_route_optimizer,
)
from smarttour.models import RouteOptimizationRequest, RouteOptimizationResponse
from smarttour.services import MapsClient, RouteOptimizer

router = APIRouter(prefix="/api/routes", tags=["routes"])


@router.post("/optimize", response_model=RouteOptimizationResponse)
def optimize_route(
    payload: RouteOptimizationRequest,
    session: Annotated[Session, Depends(get_db_session)],
    maps_client: Annotated[MapsClient, Depends(get_maps_client)],
    route_optimizer: Annotated[RouteOptimizer, Depends(get_route_optimizer)],
) -> RouteOptimizationResponse:
    """
    Optimize a route for a single list of attractions.

    Args:
        payload: Route optimization payload.
        session: Database session.
        maps_client: Maps client dependency.
        route_optimizer: Route optimizer dependency.

    Returns:
        The optimized route.
    """

    graph = maps_client.build_route_graph(
        session, payload.attractions, payload.travel_mode
    )
    route = route_optimizer.optimize(payload.attractions, graph, payload.travel_mode)
    return RouteOptimizationResponse(route=route)
