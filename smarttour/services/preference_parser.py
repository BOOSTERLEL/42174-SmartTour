"""
Natural language preference parsing backed by LLM structured output.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from smarttour.config import get_settings
from smarttour.models import PreferenceParseResponse, TravelPreferences
from smarttour.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a travel-planning assistant. "
    "Extract structured travel preferences from the user's natural language input. "
    "If information is not explicitly stated, use sensible defaults:\n"
    "- destination: the city or region mentioned (default 'Melbourne')\n"
    "- trip_days: number of days (default 3)\n"
    "- budget_total: total budget as a number, 0.0 if not mentioned\n"
    "- interests: list of interest categories from "
    "[museum, park, food, shopping, landmark, wildlife, beach, nightlife, culture, sport]\n"
    "- travel_mode: one of 'walking', 'transit', 'driving' (default 'walking')\n"
    "- pace: one of 'relaxed', 'balanced', 'packed' (default 'balanced')\n"
    "- travelers: number of travelers (default 1)\n"
    "- accessibility_needs: list of needs like 'wheelchair', 'family-friendly' (default empty)\n"
)

REFINE_SYSTEM_PROMPT = (
    "You are a travel-planning assistant. "
    "The user already has structured travel preferences and wants to refine them. "
    "Apply the user's feedback to update the existing preferences. "
    "Only change fields the user explicitly mentions; keep everything else the same."
)

INTEREST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "museum": ("museum", "gallery", "art"),
    "park": ("park", "garden", "nature", "outdoor"),
    "food": ("food", "restaurant", "cafe", "coffee", "brunch"),
    "shopping": ("shopping", "market", "mall"),
    "landmark": ("landmark", "architecture", "viewpoint"),
    "wildlife": ("wildlife", "zoo", "penguin", "animals"),
}


class LLMPreferenceOutput(BaseModel):
    """
    Schema for LLM structured output of travel preferences.
    """

    destination: str = Field(default="Melbourne")
    trip_days: int = Field(default=3, ge=1, le=14)
    budget_total: float = Field(default=0.0, ge=0.0)
    interests: list[str] = Field(default_factory=list)
    travel_mode: str = Field(default="walking")
    pace: str = Field(default="balanced")
    travelers: int = Field(default=1, ge=1, le=20)
    accessibility_needs: list[str] = Field(default_factory=list)


class PreferenceParser:
    """
    Parse natural language text into structured preferences using LLM
    with a regex-based fallback.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """
        Initialize the parser.

        Args:
            llm_client: Optional LLM client override.
        """

        self.llm_client = llm_client or LLMClient()
        self.settings = get_settings()

    def parse(self, text: str) -> PreferenceParseResponse:
        """
        Parse natural language input into structured preferences.

        Uses LLM structured output when available, falls back to regex extraction.

        Args:
            text: Raw user input.

        Returns:
            Structured preferences and warnings.
        """

        normalized_text = text.strip()
        warnings: list[str] = []

        if self.llm_client.enabled:
            try:
                llm_output = self.llm_client.parse(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=normalized_text,
                    response_model=LLMPreferenceOutput,
                )
                travel_mode = llm_output.travel_mode
                if travel_mode not in ("walking", "transit", "driving"):
                    travel_mode = "walking"
                pace = llm_output.pace
                if pace not in ("relaxed", "balanced", "packed"):
                    pace = "balanced"
                preferences = TravelPreferences(
                    destination=llm_output.destination
                    or self.settings.default_destination,
                    trip_days=llm_output.trip_days,
                    budget_total=llm_output.budget_total,
                    interests=llm_output.interests,
                    travel_mode=travel_mode,
                    pace=pace,
                    travelers=llm_output.travelers,
                    accessibility_needs=llm_output.accessibility_needs,
                    preferred_start_hour=self.settings.default_start_hour,
                    preferred_end_hour=self.settings.default_end_hour,
                    origin_summary=normalized_text,
                )
                if preferences.budget_total <= 0:
                    warnings.append(
                        "Budget was not explicit. The planner will prioritize feasibility over cost control."
                    )
                if not preferences.interests:
                    warnings.append(
                        "No specific interests were detected. The planner will build a balanced itinerary."
                    )
                return PreferenceParseResponse(
                    preferences=preferences, warnings=warnings
                )
            except Exception:
                logger.exception("LLM preference parsing failed, using regex fallback")

        preferences = TravelPreferences(
            destination=self._extract_destination(normalized_text),
            trip_days=self._extract_days(normalized_text),
            budget_total=self._extract_budget(normalized_text),
            interests=self._extract_interests(normalized_text),
            travel_mode=self._extract_travel_mode(normalized_text),
            pace=self._extract_pace(normalized_text),
            travelers=self._extract_travelers(normalized_text),
            accessibility_needs=self._extract_accessibility_needs(normalized_text),
            preferred_start_hour=self.settings.default_start_hour,
            preferred_end_hour=self.settings.default_end_hour,
            origin_summary=normalized_text,
        )
        if preferences.budget_total <= 0:
            warnings.append(
                "Budget was not explicit. The planner will prioritize feasibility over cost control."
            )
        if not preferences.interests:
            warnings.append(
                "No specific interests were detected. The planner will build a balanced itinerary."
            )
        return PreferenceParseResponse(preferences=preferences, warnings=warnings)

    def refine(
        self, base_preferences: TravelPreferences, feedback: str
    ) -> PreferenceParseResponse:
        """
        Refine existing preferences using additional feedback.

        Uses LLM when available, falls back to merge-based refinement.

        Args:
            base_preferences: Existing structured preferences.
            feedback: Follow-up user instructions.

        Returns:
            Updated preferences and warnings.
        """

        if self.llm_client.enabled:
            try:
                current_json = base_preferences.model_dump_json(indent=2)
                user_prompt = (
                    f"Current preferences:\n{current_json}\n\n"
                    f"User feedback: {feedback}\n\n"
                    f"Return the updated preferences as JSON."
                )
                llm_output = self.llm_client.parse(
                    system_prompt=REFINE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_model=LLMPreferenceOutput,
                )
                travel_mode = llm_output.travel_mode
                if travel_mode not in ("walking", "transit", "driving"):
                    travel_mode = base_preferences.travel_mode
                pace = llm_output.pace
                if pace not in ("relaxed", "balanced", "packed"):
                    pace = base_preferences.pace
                updated = base_preferences.model_copy(
                    update={
                        "destination": llm_output.destination
                        or base_preferences.destination,
                        "trip_days": llm_output.trip_days,
                        "budget_total": llm_output.budget_total,
                        "interests": llm_output.interests or base_preferences.interests,
                        "travel_mode": travel_mode,
                        "pace": pace,
                        "travelers": llm_output.travelers,
                        "accessibility_needs": llm_output.accessibility_needs,
                        "origin_summary": f"{base_preferences.origin_summary}\n{feedback.strip()}".strip(),
                    }
                )
                return PreferenceParseResponse(preferences=updated, warnings=[])
            except Exception:
                logger.exception("LLM refinement failed, using merge fallback")

        refined = self.parse(feedback).preferences
        merged_interests = sorted({*base_preferences.interests, *refined.interests})
        updated = base_preferences.model_copy(
            update={
                "destination": refined.destination or base_preferences.destination,
                "trip_days": refined.trip_days or base_preferences.trip_days,
                "budget_total": refined.budget_total or base_preferences.budget_total,
                "interests": merged_interests or base_preferences.interests,
                "travel_mode": refined.travel_mode or base_preferences.travel_mode,
                "pace": refined.pace or base_preferences.pace,
                "travelers": refined.travelers or base_preferences.travelers,
                "accessibility_needs": sorted(
                    {
                        *base_preferences.accessibility_needs,
                        *refined.accessibility_needs,
                    }
                ),
                "origin_summary": f"{base_preferences.origin_summary}\n{feedback.strip()}".strip(),
            }
        )
        return PreferenceParseResponse(preferences=updated, warnings=[])

    def _extract_destination(self, text: str) -> str:
        """
        Infer the travel destination from free text.

        Args:
            text: Raw user input.

        Returns:
            The inferred destination.
        """

        match = re.search(
            r"(?:visit|travel to|go to|trip to|in)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            text,
        )
        if match:
            return match.group(1).strip()
        known_destinations = ("Melbourne", "Sydney", "Brisbane", "Adelaide", "Perth")
        for destination in known_destinations:
            if destination.lower() in text.lower():
                return destination
        return self.settings.default_destination

    def _extract_days(self, text: str) -> int:
        """
        Extract trip duration from free text.

        Args:
            text: Raw user input.

        Returns:
            The inferred trip length in days.
        """

        match = re.search(
            r"(\d+)\s*(?:day|days|night|nights)", text, flags=re.IGNORECASE
        )
        if match:
            return max(1, min(14, int(match.group(1))))
        return self.settings.default_trip_days

    def _extract_budget(self, text: str) -> float:
        """
        Extract trip budget from free text.

        Args:
            text: Raw user input.

        Returns:
            The inferred total budget.
        """

        match = re.search(
            r"(?:\$|AUD\s*|USD\s*)(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE
        )
        if match:
            return float(match.group(1))
        match = re.search(
            r"budget(?:\s+of)?\s+(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE
        )
        if match:
            return float(match.group(1))
        return 0.0

    def _extract_interests(self, text: str) -> list[str]:
        """
        Extract interest categories from free text.

        Args:
            text: Raw user input.

        Returns:
            Normalized interest categories.
        """

        lowered = text.lower()
        interests = [
            category
            for category, keywords in INTEREST_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        return sorted(interests)

    def _extract_travel_mode(self, text: str) -> str:
        """
        Extract the preferred transport mode.

        Args:
            text: Raw user input.

        Returns:
            The inferred travel mode.
        """

        lowered = text.lower()
        if any(token in lowered for token in ("drive", "car", "driving")):
            return "driving"
        if any(
            token in lowered
            for token in ("tram", "train", "bus", "public transport", "transit")
        ):
            return "transit"
        return "walking"

    def _extract_pace(self, text: str) -> str:
        """
        Extract the preferred trip pace.

        Args:
            text: Raw user input.

        Returns:
            The inferred pace.
        """

        lowered = text.lower()
        if any(token in lowered for token in ("relaxed", "slow", "easy")):
            return "relaxed"
        if any(token in lowered for token in ("packed", "busy", "maximize")):
            return "packed"
        return "balanced"

    def _extract_travelers(self, text: str) -> int:
        """
        Extract traveler count from free text.

        Args:
            text: Raw user input.

        Returns:
            The inferred traveler count.
        """

        match = re.search(
            r"(\d+)\s*(?:people|persons|travellers|travelers|adults)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return max(1, min(20, int(match.group(1))))
        if "couple" in text.lower():
            return 2
        return 1

    def _extract_accessibility_needs(self, text: str) -> list[str]:
        """
        Extract accessibility requirements from free text.

        Args:
            text: Raw user input.

        Returns:
            The inferred accessibility needs.
        """

        lowered = text.lower()
        needs: list[str] = []
        if "wheelchair" in lowered or "accessible" in lowered:
            needs.append("wheelchair")
        if "kids" in lowered or "family" in lowered:
            needs.append("family-friendly")
        return needs
