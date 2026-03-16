"""
FastAPI dependency providers.
"""

from collections.abc import Generator

from sqlmodel import Session

from smarttour.db import get_session
from smarttour.services import (
    DataRetrievalService,
    GuidanceGenerator,
    ItineraryGenerator,
    ItineraryPlanner,
    MapsClient,
    PreferenceParser,
    RouteOptimizer,
)


def get_db_session() -> Generator[Session, None, None]:
    """
    Yield a database session for request handlers.

    Yields:
        An active SQLModel session.
    """

    yield from get_session()


def get_preference_parser() -> PreferenceParser:
    """
    Return the preference parser service.

    Returns:
        A preference parser instance.
    """

    return PreferenceParser()


def get_data_retrieval_service() -> DataRetrievalService:
    """
    Return the data retrieval service.

    Returns:
        A retrieval service instance.
    """

    return DataRetrievalService()


def get_maps_client() -> MapsClient:
    """
    Return the maps client.

    Returns:
        A maps client instance.
    """

    return MapsClient()


def get_route_optimizer() -> RouteOptimizer:
    """
    Return the route optimizer.

    Returns:
        A route optimizer instance.
    """

    return RouteOptimizer()


def get_itinerary_planner() -> ItineraryPlanner:
    """
    Return the itinerary planner service.

    Returns:
        An itinerary planner instance.
    """

    return ItineraryPlanner()


def get_itinerary_generator() -> ItineraryGenerator:
    """
    Return the itinerary generator.

    Returns:
        An itinerary generator instance.
    """

    return ItineraryGenerator(
        data_retrieval=get_data_retrieval_service(),
        maps_client=get_maps_client(),
        route_optimizer=get_route_optimizer(),
        planner=get_itinerary_planner(),
    )


def get_guidance_generator() -> GuidanceGenerator:
    """
    Return the guidance generator.

    Returns:
        A guidance generator instance.
    """

    return GuidanceGenerator()
