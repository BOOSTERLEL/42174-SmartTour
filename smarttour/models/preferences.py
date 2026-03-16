"""
Preference models shared across the application.
"""

from typing import Literal

from sqlmodel import Field, SQLModel

TravelMode = Literal["walking", "transit", "driving", "walking_transit"]
TravelPace = Literal["relaxed", "balanced", "packed"]


class UserInput(SQLModel):
    """
    Raw natural language user input.
    """

    text: str = Field(min_length=1)


class TravelPreferences(SQLModel):
    """
    Structured travel preferences derived from user input.
    """

    destination: str
    trip_days: int = Field(default=3, ge=1, le=14)
    budget_total: float = Field(default=0.0, ge=0.0)
    interests: list[str] = Field(default_factory=list)
    travel_mode: TravelMode = "walking_transit"
    pace: TravelPace = "balanced"
    travelers: int = Field(default=1, ge=1, le=20)
    accessibility_needs: list[str] = Field(default_factory=list)
    preferred_start_hour: int = Field(default=9, ge=6, le=12)
    preferred_end_hour: int = Field(default=21, ge=12, le=23)
    origin_summary: str = ""


class PreferenceParseRequest(SQLModel):
    """
    API payload for preference parsing.
    """

    user_input: UserInput


class PreferenceParseResponse(SQLModel):
    """
    API response for preference parsing.
    """

    preferences: TravelPreferences
    warnings: list[str] = Field(default_factory=list)


class PreferenceRefineRequest(SQLModel):
    """
    API payload for refining existing preferences.
    """

    base_preferences: TravelPreferences
    feedback: str = Field(min_length=1)
