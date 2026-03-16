"""
LLM-backed itinerary planning and review helpers.
"""

from __future__ import annotations

import json
import logging

from smarttour.models import (
    Attraction,
    Hotel,
    Itinerary,
    ItineraryPlan,
    ItineraryReview,
    TravelPreferences,
)
from smarttour.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

PLAN_SYSTEM_PROMPT = """
You are an expert travel planner. Given available attractions, hotel location,
and traveler preferences, create an optimal daily itinerary plan.

RULES:
1. Group nearby attractions into the same day to minimize travel time.
2. Consider thematic coherence, such as museums together or outdoor activities together.
3. Prioritize attractions with higher popularity scores from social media.
4. Respect the traveler's pace preference:
   - "relaxed": max 2-3 attractions per day, generous breaks
   - "balanced": 3-4 attractions per day
   - "packed": 4-5 attractions per day, maximize coverage
5. Each day_theme attraction_ids list should contain ONLY IDs from the available attractions.
6. Every attraction should appear in exactly one day.
7. Consider opening hours and visit durations when grouping.
8. Give each day a descriptive theme.
9. When xhs_notes are provided for an attraction, use them as social media insights
   about the place (visitor tips, highlights, best times to visit) to inform your grouping.
10. Keep each day's attractions within a compact geographic area (ideally under 10km radius).
11. Minimize total walking time to under 2 hours per day; prefer transit for longer distances.
""".strip()

REVIEW_SYSTEM_PROMPT = """
You are a travel itinerary quality reviewer. Analyze the generated itinerary and
provide structured feedback.

REVIEW CRITERIA:
1. Geographic flow: Do daily routes avoid excessive backtracking? Are nearby places grouped?
2. Pacing: Does the schedule match the pace preference? Too rushed or too relaxed?
3. Meal timing: Are meals at reasonable times (breakfast 7-9, lunch 11:30-14, dinner 18-20)?
4. Budget: Does total cost stay within budget? Are there cost-saving opportunities?
5. Balance: Are days roughly balanced in activity density?
6. Variety: Is there good mix of attraction types across days?
7. Transport: Are transport durations reasonable? Any segments that should change mode?

SCORING:
- 8-10: Excellent itinerary, minor tweaks only
- 5-7: Good but with notable issues
- 1-4: Significant problems that need rework

For each issue found, provide a specific, actionable suggestion.
""".strip()

PACE_GUIDANCE = {
    "relaxed": "Max 2-3 attractions per day with generous breaks.",
    "balanced": "Aim for 3-4 attractions per day.",
    "packed": "Aim for 4-5 attractions per day and maximize coverage.",
}


class ItineraryPlanner:
    """
    Generate AI planning guidance and post-generation reviews.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """
        Initialize the itinerary planner.

        Args:
            llm_client: Optional LLM client override.
        """

        self.llm_client = llm_client or LLMClient()

    def plan(
        self,
        attractions: list[Attraction],
        hotel: Hotel | None,
        preferences: TravelPreferences,
        popularity: dict[str, float],
        place_notes: dict[str, list[str]] | None = None,
    ) -> ItineraryPlan | None:
        """
        Request a structured pre-generation itinerary plan.

        Args:
            attractions: Available attractions for the trip.
            hotel: Selected accommodation, if available.
            preferences: Structured travel preferences.
            popularity: Xiaohongshu popularity mapping.
            place_notes: Optional XHS note titles per place for richer context.

        Returns:
            Structured plan guidance or ``None`` when unavailable.
        """

        if not self.llm_client.enabled:
            return None
        logger.info(
            "AI planning: %d attractions, hotel=%s, pace=%s",
            len(attractions),
            hotel.name if hotel is not None else "none",
            preferences.pace,
        )

        budget_per_day = (
            round(preferences.budget_total / preferences.trip_days, 2)
            if preferences.budget_total > 0
            else None
        )
        attraction_payloads = []
        for attraction in attractions:
            entry: dict[str, object] = {
                "id": attraction.id,
                "name": attraction.name,
                "category": attraction.category,
                "lat": attraction.latitude,
                "lon": attraction.longitude,
                "rating": attraction.rating,
                "visit_duration": attraction.visit_duration,
                "cost": attraction.cost,
                "opening_hours": attraction.opening_hours,
                "xhs_popularity_score": self._popularity_score(
                    attraction.name, popularity
                ),
            }
            if place_notes and attraction.name in place_notes:
                entry["xhs_notes"] = place_notes[attraction.name]
            attraction_payloads.append(entry)

        hotel_payload: dict[str, object] | None = None
        if hotel is not None:
            hotel_payload = {
                "name": hotel.name,
                "lat": hotel.latitude,
                "lon": hotel.longitude,
            }
            if place_notes and hotel.name in place_notes:
                hotel_payload["xhs_notes"] = place_notes[hotel.name]

        payload = {
            "attractions": attraction_payloads,
            "hotel": hotel_payload,
            "preferences": {
                "destination": preferences.destination,
                "trip_days": preferences.trip_days,
                "budget_total": preferences.budget_total,
                "budget_per_day_guidance": budget_per_day,
                "interests": preferences.interests,
                "travel_mode": preferences.travel_mode,
                "pace": preferences.pace,
                "pace_guidance": PACE_GUIDANCE[preferences.pace],
                "travelers": preferences.travelers,
            },
        }
        user_prompt = (
            "Create a structured itinerary strategy for the following trip context.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        try:
            plan = self.llm_client.parse(
                system_prompt=PLAN_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=ItineraryPlan,
            )
            logger.info(
                "AI planning complete: %d day themes, %d priority attractions",
                len(plan.day_themes),
                len(plan.priority_attractions),
            )
            logger.debug("AI plan reasoning: %s", plan.reasoning[:200])
            return plan
        except Exception:
            logger.exception("AI itinerary planning failed")
            return None

    def review(self, itinerary: Itinerary) -> ItineraryReview | None:
        """
        Request a structured post-generation itinerary review.

        Args:
            itinerary: Generated itinerary to review.

        Returns:
            Structured review feedback or ``None`` when unavailable.
        """

        if not self.llm_client.enabled:
            return None

        budget_total = itinerary.preferences.budget_total
        over_budget_by = (
            round(itinerary.total_estimated_cost - budget_total, 2)
            if budget_total > 0
            else None
        )
        payload = {
            "destination": itinerary.destination,
            "preferences": {
                "trip_days": itinerary.preferences.trip_days,
                "budget_total": budget_total,
                "interests": itinerary.preferences.interests,
                "travel_mode": itinerary.preferences.travel_mode,
                "pace": itinerary.preferences.pace,
                "travelers": itinerary.preferences.travelers,
            },
            "accommodation": (
                {
                    "name": itinerary.accommodation.name,
                    "price_per_night": itinerary.accommodation.price_per_night,
                }
                if itinerary.accommodation is not None
                else None
            ),
            "budget_comparison": {
                "budget_total": budget_total,
                "actual_total": itinerary.total_estimated_cost,
                "within_budget": (
                    itinerary.total_estimated_cost <= budget_total
                    if budget_total > 0
                    else None
                ),
                "over_budget_by": over_budget_by,
            },
            "days": [self._serialize_day(day) for day in itinerary.days],
            "warnings": itinerary.warnings,
        }
        user_prompt = (
            "Review the generated itinerary and provide structured feedback.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        try:
            review = self.llm_client.parse(
                system_prompt=REVIEW_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=ItineraryReview,
            )
            logger.info(
                "AI review: score=%d, %d issues, %d suggestions",
                review.overall_score,
                len(review.issues),
                len(review.suggestions),
            )
            return review
        except Exception:
            logger.exception("AI itinerary review failed")
            return None

    def _serialize_day(self, day) -> dict[str, object]:
        """
        Convert a day plan into LLM review input.

        Args:
            day: Day plan to serialize.

        Returns:
            A JSON-friendly day payload.
        """

        return {
            "day_number": day.day_number,
            "label": day.label,
            "estimated_cost": day.estimated_cost,
            "warnings": day.warnings,
            "route": {
                "total_distance_m": day.route.total_distance_m,
                "total_duration_s": day.route.total_duration_s,
            },
            "slots": [
                {
                    "type": slot.slot_type,
                    "start_time": slot.start_time,
                    "end_time": slot.end_time,
                    "duration_minutes": self._duration_minutes(
                        slot.start_time, slot.end_time
                    ),
                    "name": slot.title,
                    "cost": slot.cost,
                    "transport": (
                        {
                            "mode": slot.transport_from_previous.travel_mode,
                            "duration_minutes": round(
                                slot.transport_from_previous.duration_s / 60
                            ),
                            "distance_m": slot.transport_from_previous.distance_m,
                            "hint": slot.transport_from_previous.navigation_hint,
                        }
                        if slot.transport_from_previous is not None
                        else None
                    ),
                }
                for slot in day.slots
            ],
        }

    def _duration_minutes(self, start_time: str, end_time: str) -> int:
        """
        Compute slot duration from two HH:MM strings.

        Args:
            start_time: Slot start.
            end_time: Slot end.

        Returns:
            Duration in minutes.
        """

        return max(0, self._parse_minutes(end_time) - self._parse_minutes(start_time))

    def _parse_minutes(self, value: str) -> int:
        """
        Parse an HH:MM string into minutes from midnight.

        Args:
            value: Time string.

        Returns:
            Minutes from midnight.
        """

        hours_text, minutes_text = value.split(":")
        return int(hours_text) * 60 + int(minutes_text)

    def _popularity_score(self, name: str, popularity: dict[str, float]) -> float:
        """
        Compute a raw popularity score for one attraction name.

        Args:
            name: Attraction or hotel name.
            popularity: Xiaohongshu popularity mapping.

        Returns:
            Aggregated popularity score.
        """

        normalized_name = name.strip().lower()
        if not normalized_name:
            return 0.0
        return sum(
            score
            for title, score in popularity.items()
            if normalized_name in title.strip().lower()
        )
