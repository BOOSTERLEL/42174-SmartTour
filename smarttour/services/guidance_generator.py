"""
Generate attraction guidance responses using LLM with deterministic fallback.
"""

from __future__ import annotations

import logging

from sqlmodel import Session

from smarttour.models import GuidanceResponse, TravelPreferences
from smarttour.services.data_retrieval import DataRetrievalService
from smarttour.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

GUIDANCE_SYSTEM_PROMPT = (
    "You are a knowledgeable travel guide. "
    "Given an attraction's information, generate rich, engaging content for a traveler.\n\n"
    "Respond with exactly three sections:\n"
    "1. historical_background: 2-3 paragraphs covering the history, cultural significance, "
    "and what makes this place special. Be specific and informative.\n"
    "2. visiting_tips: 3-5 practical tips for visiting (best time, what to wear, "
    "what to bring, how long to spend, nearby food options, etc.)\n"
    "3. practical_notes: 2-4 notes about accessibility, opening hours, "
    "ticket prices, and transport options.\n"
)


class LLMGuidanceOutput:
    """
    Not used as a Pydantic parse target; guidance uses free-text completion
    and structures the output manually for richer content.
    """


class GuidanceGenerator:
    """
    Produce AI-generated attraction guidance with a deterministic fallback.
    """

    def __init__(
        self,
        data_retrieval: DataRetrievalService | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the guidance generator.

        Args:
            data_retrieval: Optional attraction retrieval service.
            llm_client: Optional LLM client.
        """

        self.data_retrieval = data_retrieval or DataRetrievalService()
        self.llm_client = llm_client or LLMClient()

    def explain(
        self,
        session: Session,
        attraction_id: str,
        preferences: TravelPreferences | None = None,
    ) -> GuidanceResponse:
        """
        Generate a narrative explanation for one attraction.

        Uses LLM when available, falls back to template-based generation.

        Args:
            session: Active database session.
            attraction_id: Attraction identifier.
            preferences: Optional travel preferences context.

        Returns:
            Generated guidance content.
        """

        attraction = self.data_retrieval.get_attraction(session, attraction_id)
        if attraction is None:
            raise ValueError(f"Unknown attraction '{attraction_id}'.")

        audience = "a general traveler"
        if preferences is not None and preferences.interests:
            audience = f"a traveler interested in {', '.join(preferences.interests)}"

        if self.llm_client.enabled:
            try:
                return self._generate_with_llm(attraction, audience)
            except Exception:
                logger.exception(
                    "LLM guidance generation failed for '%s', using fallback",
                    attraction.name,
                )

        return self._generate_fallback(attraction, audience)

    def _generate_with_llm(self, attraction, audience: str) -> GuidanceResponse:
        """
        Generate guidance content using the LLM.

        Args:
            attraction: The attraction to describe.
            audience: Description of the target audience.

        Returns:
            LLM-generated guidance response.
        """

        user_prompt = (
            f"Attraction: {attraction.name}\n"
            f"Location: {attraction.destination}\n"
            f"Category: {attraction.category}\n"
            f"Description: {attraction.description}\n"
            f"Rating: {attraction.rating}/5.0\n"
            f"Typical visit duration: {attraction.visit_duration} minutes\n"
            f"Entry cost: {attraction.cost}\n"
            f"Opening hours: {self._format_opening_hours(attraction.opening_hours)}\n"
            f"Accessibility: {attraction.accessibility or 'No specific notes'}\n"
            f"Tags: {', '.join(attraction.tags) if attraction.tags else 'general'}\n"
            f"\nTarget audience: {audience}\n"
            f"\nGenerate detailed guidance with:\n"
            f"1. Historical background and cultural significance (2-3 paragraphs)\n"
            f"2. Visiting tips (3-5 bullet points)\n"
            f"3. Practical notes (2-4 bullet points)"
        )

        raw_response = self.llm_client.complete(GUIDANCE_SYSTEM_PROMPT, user_prompt)
        sections = self._parse_guidance_sections(raw_response)

        return GuidanceResponse(
            attraction=attraction,
            historical_background=sections["background"],
            visiting_tips=sections["tips"],
            practical_notes=sections["notes"],
        )

    def _parse_guidance_sections(self, text: str) -> dict[str, str | list[str]]:
        """
        Parse LLM output into structured guidance sections.

        Args:
            text: Raw LLM output.

        Returns:
            Parsed sections with background, tips, and notes.
        """

        background = ""
        tips: list[str] = []
        notes: list[str] = []

        current_section = "background"
        background_lines: list[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            lower = stripped.lower()

            if any(
                marker in lower
                for marker in [
                    "visiting tip",
                    "tips for visiting",
                    "practical tip",
                    "travel tip",
                ]
            ):
                current_section = "tips"
                continue
            if any(
                marker in lower
                for marker in [
                    "practical note",
                    "practical info",
                    "useful info",
                    "logistics",
                ]
            ):
                current_section = "notes"
                continue

            if stripped.startswith(("2.", "2)")):
                current_section = "tips"
                continue
            if stripped.startswith(("3.", "3)")):
                current_section = "notes"
                continue

            if not stripped:
                if current_section == "background":
                    background_lines.append("")
                continue

            if current_section == "background":
                background_lines.append(stripped)
            elif current_section == "tips":
                cleaned = stripped.lstrip("-*•● ").strip()
                if cleaned:
                    tips.append(cleaned)
            elif current_section == "notes":
                cleaned = stripped.lstrip("-*•● ").strip()
                if cleaned:
                    notes.append(cleaned)

        background = "\n".join(background_lines).strip()
        if not background:
            background = text[:500].strip()
        if not tips:
            tips = ["Check local conditions before visiting."]
        if not notes:
            notes = ["Verify opening hours locally."]

        return {"background": background, "tips": tips, "notes": notes}

    def _generate_fallback(self, attraction, audience: str) -> GuidanceResponse:
        """
        Generate guidance using template interpolation when LLM is unavailable.

        Args:
            attraction: The attraction to describe.
            audience: Description of the target audience.

        Returns:
            Template-based guidance response.
        """

        return GuidanceResponse(
            attraction=attraction,
            historical_background=(
                f"{attraction.name} is one of {attraction.destination}'s signature "
                f"{attraction.category} stops, suited for {audience}. "
                f"{attraction.description}"
            ),
            visiting_tips=[
                f"Plan around {attraction.visit_duration} minutes on site.",
                f"Ticket cost is approximately {attraction.cost:.0f} in local currency units.",
                "Arrive earlier in the day if you prefer lighter crowds.",
            ],
            practical_notes=[
                f"Typical opening guidance: {self._format_opening_hours(attraction.opening_hours)}.",
                f"Accessibility notes: {attraction.accessibility or 'No specific note provided.'}",
                f"Top tags: {', '.join(attraction.tags) if attraction.tags else 'general sightseeing'}.",
            ],
        )

    def _format_opening_hours(self, opening_hours: dict[str, str]) -> str:
        """
        Format opening hours into a compact string.

        Args:
            opening_hours: Weekly opening hours map.

        Returns:
            A readable summary string.
        """

        if not opening_hours:
            return "not available"
        first_key = next(iter(opening_hours))
        return f"{first_key}: {opening_hours[first_key]}"
