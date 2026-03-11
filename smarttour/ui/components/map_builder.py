"""
Folium map construction helpers with numbered and color-coded markers.
"""

from __future__ import annotations

import folium

from smarttour.models import DayPlan

CATEGORY_COLORS: dict[str, str] = {
    "museum": "blue",
    "park": "green",
    "food": "orange",
    "shopping": "purple",
    "landmark": "red",
    "wildlife": "darkgreen",
    "beach": "lightblue",
    "nightlife": "darkpurple",
    "culture": "cadetblue",
    "sport": "lightred",
}

DEFAULT_COLOR = "gray"


def build_day_map(day_plan: DayPlan) -> folium.Map:
    """
    Build a Folium map for one day plan with numbered, color-coded markers.

    Args:
        day_plan: Day plan to visualize.

    Returns:
        A configured Folium map.
    """

    default_location = [-37.8136, 144.9631]
    if day_plan.slots and day_plan.slots[0].attraction is not None:
        default_location = [
            day_plan.slots[0].attraction.latitude,
            day_plan.slots[0].attraction.longitude,
        ]

    travel_map = folium.Map(
        location=default_location, zoom_start=13, control_scale=True
    )

    route_points: list[list[float]] = []
    for index, slot in enumerate(day_plan.slots, start=1):
        attraction = slot.attraction
        if attraction is None:
            continue

        point = [attraction.latitude, attraction.longitude]
        route_points.append(point)

        color = CATEGORY_COLORS.get(attraction.category, DEFAULT_COLOR)
        icon = folium.DivIcon(
            html=(
                f'<div style="'
                f"background-color: {color}; "
                f"color: white; "
                f"border-radius: 50%; "
                f"width: 28px; "
                f"height: 28px; "
                f"display: flex; "
                f"align-items: center; "
                f"justify-content: center; "
                f"font-weight: bold; "
                f"font-size: 14px; "
                f"border: 2px solid white; "
                f"box-shadow: 0 2px 4px rgba(0,0,0,0.3); "
                f'">{index}</div>'
            ),
            icon_size=(28, 28),
            icon_anchor=(14, 14),
        )

        popup_html = (
            f"<b>{index}. {attraction.name}</b><br>"
            f"Category: {attraction.category}<br>"
            f"Time: {slot.start_time} - {slot.end_time}<br>"
            f"Cost: {slot.cost:.0f}<br>"
            f"Rating: {attraction.rating:.1f}/5.0"
        )

        folium.Marker(
            location=point,
            icon=icon,
            tooltip=f"{index}. {attraction.name} ({attraction.category})",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(travel_map)

    if len(route_points) >= 2:
        folium.PolyLine(
            route_points, weight=4, opacity=0.8, color="#3388ff", dash_array="10"
        ).add_to(travel_map)

    return travel_map
