"""Application service for itinerary report generation."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from smartour.domain.itinerary import Itinerary, ItineraryDay, ItineraryItem


class ItineraryReport(BaseModel):
    """
    Generated itinerary report payload.
    """

    itinerary_id: str
    title: str
    format: str = "markdown"
    markdown: str
    generated_at: datetime


class ReportService:
    """
    Generates deterministic reports from persisted itineraries.
    """

    def __init__(self, itinerary_repository: Any) -> None:
        """
        Initialize the report service.

        Args:
            itinerary_repository: Repository used to load generated itineraries.
        """
        self.itinerary_repository = itinerary_repository

    async def generate_itinerary_report(
        self, itinerary_id: str
    ) -> ItineraryReport | None:
        """
        Generate a Markdown report for an itinerary.

        Args:
            itinerary_id: The itinerary identifier.

        Returns:
            The itinerary report, or None when the itinerary is missing.
        """
        itinerary = await self.itinerary_repository.get(itinerary_id)
        if itinerary is None:
            return None
        markdown = "\n".join(_report_lines(itinerary)).strip() + "\n"
        return ItineraryReport(
            itinerary_id=itinerary.id,
            title=itinerary.title,
            markdown=markdown,
            generated_at=datetime.now(tz=UTC),
        )


def _report_lines(itinerary: Itinerary) -> list[str]:
    """
    Build Markdown report lines for an itinerary.

    Args:
        itinerary: The generated itinerary.

    Returns:
        Markdown lines.
    """
    lines = [
        f"# {itinerary.title}",
        "",
        f"Destination: {itinerary.destination_name}",
    ]
    if itinerary.hotels:
        lines.extend(["", "## Stay"])
        for hotel in itinerary.hotels:
            hotel_text = hotel.name
            if hotel.google_maps_uri is not None:
                hotel_text = f"[{hotel.name}]({hotel.google_maps_uri})"
            detail_parts = [
                value
                for value in (
                    hotel.address,
                    hotel.price_level,
                    _format_rating(hotel.rating, hotel.user_rating_count),
                )
                if value
            ]
            detail_text = f" - {'; '.join(detail_parts)}" if detail_parts else ""
            lines.append(f"- {hotel_text}{detail_text}")
    lines.extend(["", "## Daily Itinerary"])
    for day in itinerary.days:
        lines.extend(_day_lines(day))
    if itinerary.guide_markdown.strip():
        lines.extend(["", "## Guide", "", itinerary.guide_markdown.strip()])
    return lines


def _day_lines(day: ItineraryDay) -> list[str]:
    """
    Build Markdown report lines for one itinerary day.

    Args:
        day: The itinerary day.

    Returns:
        Markdown lines for the day.
    """
    heading = f"### Day {day.day_number} - {day.theme}"
    if day.date is not None:
        heading = f"{heading} ({day.date})"
    lines = ["", heading, "", day.summary]
    if day.items:
        lines.append("")
        lines.extend(_item_line(item) for item in day.items)
    if day.route is not None:
        lines.extend(
            [
                "",
                (
                    f"Route: {_format_distance(day.route.distance_meters)}, "
                    f"{_format_duration(day.route.duration_seconds)}, "
                    f"{len(day.route.legs)} legs by "
                    f"{day.route.travel_mode.lower().replace('_', ' ')}."
                ),
            ]
        )
    return lines


def _item_line(item: ItineraryItem) -> str:
    """
    Build one Markdown list item for an itinerary stop.

    Args:
        item: The itinerary item.

    Returns:
        A Markdown list item.
    """
    place_text = item.place.name
    if item.place.google_maps_uri is not None:
        place_text = f"[{item.place.name}]({item.place.google_maps_uri})"
    return (
        f"- {item.time} {item.type.value.title()}: {place_text} "
        f"({item.duration_minutes} min)"
    )


def _format_rating(rating: float | None, user_rating_count: int | None) -> str | None:
    """
    Format rating metadata for report text.

    Args:
        rating: The Google rating.
        user_rating_count: The number of Google user ratings.

    Returns:
        A readable rating string when available.
    """
    if rating is None:
        return None
    if user_rating_count is None:
        return f"{rating:.1f} stars"
    return f"{rating:.1f} stars from {user_rating_count} reviews"


def _format_distance(meters: int) -> str:
    """
    Format a route distance.

    Args:
        meters: The distance in meters.

    Returns:
        A readable distance.
    """
    if meters < 1000:
        return f"{meters} m"
    return f"{meters / 1000:.1f} km"


def _format_duration(seconds: int) -> str:
    """
    Format a route duration.

    Args:
        seconds: The duration in seconds.

    Returns:
        A readable duration.
    """
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if remaining_minutes == 0:
        return f"{hours} hr"
    return f"{hours} hr {remaining_minutes} min"
