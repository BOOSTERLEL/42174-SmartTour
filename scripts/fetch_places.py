"""
Fetch attraction and accommodation data from Google Places API into local JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from smarttour.config import DATA_DIR, get_settings

ATTRACTION_TYPES = [
    "tourist_attraction",
    "museum",
    "park",
    "art_gallery",
    "zoo",
    "aquarium",
    "restaurant",
    "cafe",
]

ACCOMMODATION_TYPES = [
    "lodging",
]


def fetch_attractions(destination: str, output_path: Path) -> list[dict[str, object]]:
    """
    Fetch attractions near a destination from Google Places API.

    Args:
        destination: The destination city name.
        output_path: Path to write the JSON output.

    Returns:
        List of attraction dictionaries.
    """

    import googlemaps

    settings = get_settings()
    if not settings.google_maps_api_key:
        print("Error: GOOGLE_MAPS_API_KEY is not set.")
        return []

    gmaps = googlemaps.Client(key=settings.google_maps_api_key)

    geocode_result = gmaps.geocode(destination)
    if not geocode_result:
        print(f"Error: Could not geocode destination '{destination}'.")
        return []

    location = geocode_result[0]["geometry"]["location"]
    center = (location["lat"], location["lng"])
    print(f"Center of '{destination}': {center}")

    attractions: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for place_type in ATTRACTION_TYPES:
        print(f"  Searching for type: {place_type}...")
        try:
            result = gmaps.places_nearby(
                location=center,
                radius=15000,
                type=place_type,
            )
        except Exception as exc:
            print(f"  Warning: API call failed for type '{place_type}': {exc}")
            continue

        for place in result.get("results", []):
            place_id = place["place_id"]
            if place_id in seen_ids:
                continue
            seen_ids.add(place_id)

            detail = _fetch_place_detail(gmaps, place_id)
            category = _map_types_to_category(place.get("types", []))

            attraction = {
                "id": place_id,
                "name": place["name"],
                "destination": destination,
                "category": category,
                "latitude": place["geometry"]["location"]["lat"],
                "longitude": place["geometry"]["location"]["lng"],
                "description": detail.get("editorial_summary", ""),
                "opening_hours": _extract_opening_hours(detail),
                "visit_duration": _estimate_visit_duration(category),
                "cost": 0.0,
                "rating": place.get("rating", 0.0),
                "accessibility": "wheelchair accessible"
                if place.get("wheelchair_accessible_entrance")
                else None,
                "tags": [
                    t
                    for t in place.get("types", [])
                    if t != "point_of_interest" and t != "establishment"
                ],
                "image_url": None,
                "source": "google_places",
            }
            attractions.append(attraction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(attractions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(attractions)} attractions to {output_path}")
    return attractions


def fetch_accommodations(
    destination: str, output_path: Path
) -> list[dict[str, object]]:
    """
    Fetch accommodation options near a destination from Google Places API.

    Args:
        destination: The destination city name.
        output_path: Path to write the JSON output.

    Returns:
        List of accommodation dictionaries.
    """

    import googlemaps

    settings = get_settings()
    if not settings.google_maps_api_key:
        print("Error: GOOGLE_MAPS_API_KEY is not set.")
        return []

    gmaps = googlemaps.Client(key=settings.google_maps_api_key)

    geocode_result = gmaps.geocode(destination)
    if not geocode_result:
        print(f"Error: Could not geocode destination '{destination}'.")
        return []

    location = geocode_result[0]["geometry"]["location"]
    center = (location["lat"], location["lng"])

    accommodations: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for place_type in ACCOMMODATION_TYPES:
        print(f"  Searching for accommodation type: {place_type}...")
        try:
            result = gmaps.places_nearby(
                location=center,
                radius=10000,
                type=place_type,
            )
        except Exception as exc:
            print(f"  Warning: API call failed for type '{place_type}': {exc}")
            continue

        for place in result.get("results", []):
            place_id = place["place_id"]
            if place_id in seen_ids:
                continue
            seen_ids.add(place_id)

            star_rating = round(place.get("rating", 3.0))
            accommodation = {
                "id": place_id,
                "name": place["name"],
                "destination": destination,
                "latitude": place["geometry"]["location"]["lat"],
                "longitude": place["geometry"]["location"]["lng"],
                "price_per_night": _estimate_price(place.get("price_level")),
                "star_rating": max(1, min(5, star_rating)),
                "category": "hotel",
                "amenities": [],
                "booking_url": None,
            }
            accommodations.append(accommodation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(accommodations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(accommodations)} accommodations to {output_path}")
    return accommodations


def _fetch_place_detail(gmaps, place_id: str) -> dict[str, object]:
    """
    Fetch detailed information for a single place.

    Args:
        gmaps: Google Maps client.
        place_id: The place ID to look up.

    Returns:
        A dictionary of place details.
    """

    try:
        result = gmaps.place(
            place_id=place_id,
            fields=[
                "opening_hours",
                "editorial_summary",
                "wheelchair_accessible_entrance",
            ],
        )
        detail = result.get("result", {})
        summary = detail.get("editorial_summary", {})
        return {
            "opening_hours": detail.get("opening_hours", {}),
            "editorial_summary": summary.get("overview", "")
            if isinstance(summary, dict)
            else "",
            "wheelchair_accessible_entrance": detail.get(
                "wheelchair_accessible_entrance", False
            ),
        }
    except Exception:
        return {}


def _extract_opening_hours(detail: dict[str, object]) -> dict[str, str]:
    """
    Extract opening hours from place detail into a weekday map.

    Args:
        detail: Place detail dictionary.

    Returns:
        A map of day abbreviations to opening hours strings.
    """

    oh = detail.get("opening_hours", {})
    if not isinstance(oh, dict):
        return {}

    weekday_text = oh.get("weekday_text", [])
    if not isinstance(weekday_text, list):
        return {}

    day_map = {
        "Monday": "mon",
        "Tuesday": "tue",
        "Wednesday": "wed",
        "Thursday": "thu",
        "Friday": "fri",
        "Saturday": "sat",
        "Sunday": "sun",
    }
    hours: dict[str, str] = {}
    for entry in weekday_text:
        if not isinstance(entry, str) or ":" not in entry:
            continue
        for full_name, abbrev in day_map.items():
            if entry.startswith(full_name):
                time_part = entry.split(":", 1)[1].strip()
                hours[abbrev] = time_part
                break
    return hours


def _map_types_to_category(types: list[str]) -> str:
    """
    Map Google Places types to SmartTour categories.

    Args:
        types: List of Google Places type strings.

    Returns:
        A SmartTour category string.
    """

    type_set = set(types)
    if type_set & {"museum", "art_gallery"}:
        return "museum"
    if type_set & {"park", "garden"}:
        return "park"
    if type_set & {"zoo", "aquarium"}:
        return "wildlife"
    if type_set & {"restaurant", "cafe", "bakery", "bar"}:
        return "food"
    if type_set & {"shopping_mall", "store"}:
        return "shopping"
    if type_set & {"church", "hindu_temple", "mosque", "synagogue"}:
        return "landmark"
    return "landmark"


def _estimate_visit_duration(category: str) -> int:
    """
    Estimate typical visit duration by category.

    Args:
        category: SmartTour attraction category.

    Returns:
        Estimated visit duration in minutes.
    """

    durations = {
        "museum": 120,
        "park": 90,
        "wildlife": 150,
        "food": 60,
        "shopping": 90,
        "landmark": 45,
    }
    return durations.get(category, 90)


def _estimate_price(price_level: int | None) -> float:
    """
    Estimate nightly price from Google's price_level indicator.

    Args:
        price_level: Google's 0-4 price level.

    Returns:
        Estimated price per night.
    """

    levels = {0: 50.0, 1: 80.0, 2: 150.0, 3: 250.0, 4: 400.0}
    return levels.get(price_level or 2, 150.0)


def main() -> None:
    """
    CLI entrypoint for fetching external place data.
    """

    destination = "Melbourne"
    if len(sys.argv) > 1:
        destination = sys.argv[1]

    print(f"Fetching data for destination: {destination}")
    fetch_attractions(destination, DATA_DIR / "attractions.json")
    fetch_accommodations(destination, DATA_DIR / "hotels.json")
    print("Done.")


if __name__ == "__main__":
    main()
