"""
Preference parsing router.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from smarttour.api.dependencies import get_preference_parser
from smarttour.models import (
    PreferenceParseRequest,
    PreferenceParseResponse,
    PreferenceRefineRequest,
)
from smarttour.services import PreferenceParser

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


@router.post("/parse", response_model=PreferenceParseResponse)
def parse_preferences(
    payload: PreferenceParseRequest,
    parser: Annotated[PreferenceParser, Depends(get_preference_parser)],
) -> PreferenceParseResponse:
    """
    Parse natural language into structured preferences.

    Args:
        payload: Incoming parse payload.
        parser: Preference parser dependency.

    Returns:
        Parsed preferences and warnings.
    """

    return parser.parse(payload.user_input.text)


@router.post("/refine", response_model=PreferenceParseResponse)
def refine_preferences(
    payload: PreferenceRefineRequest,
    parser: Annotated[PreferenceParser, Depends(get_preference_parser)],
) -> PreferenceParseResponse:
    """
    Refine existing preferences from follow-up instructions.

    Args:
        payload: Refinement payload.
        parser: Preference parser dependency.

    Returns:
        Refined preferences and warnings.
    """

    return parser.refine(payload.base_preferences, payload.feedback)
