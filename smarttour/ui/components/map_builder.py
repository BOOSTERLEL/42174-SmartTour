"""
Folium map construction helpers with numbered markers and route rendering.
"""

from __future__ import annotations

import folium

from smarttour.models import DayPlan, TimeSlot

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

MEAL_COLORS: dict[str, str] = {
    "breakfast": "orange",
    "lunch": "beige",
    "dinner": "darkred",
}

SEGMENT_COLORS: dict[str, str] = {
    "walking": "#3388ff",
    "transit": "#28a745",
    "driving": "#dc3545",
}

DEFAULT_COLOR = "gray"


def build_day_map(day_plan: DayPlan) -> folium.Map:
    """
    Build a Folium map for one day plan with markers and routed segments.

    Args:
        day_plan: Day plan to visualize.

    Returns:
        A configured Folium map.
    """

    default_location = [-37.8136, 144.9631]
    if day_plan.slots:
        first_point = _slot_coordinates(day_plan.slots[0])
        if first_point is not None:
            default_location = first_point

    travel_map = folium.Map(
        location=default_location,
        zoom_start=13,
        control_scale=True,
    )

    previous_slot: TimeSlot | None = None
    for index, slot in enumerate(day_plan.slots, start=1):
        point = _slot_coordinates(slot)
        if point is None:
            continue

        if slot.attraction is not None:
            color = CATEGORY_COLORS.get(slot.attraction.category, DEFAULT_COLOR)
            popup_html = (
                f"<b>{index}. {slot.title}</b><br>"
                f"Stop Type: {slot.slot_type.title()}<br>"
                f"Category: {slot.attraction.category}<br>"
                f"Time: {slot.start_time} - {slot.end_time}<br>"
                f"Cost: {slot.cost:.0f}<br>"
                f"Rating: {slot.attraction.rating:.1f}/5.0"
            )
            tooltip = f"{index}. {slot.title} ({slot.attraction.category})"
        elif slot.restaurant is not None:
            color = MEAL_COLORS.get(slot.slot_type, DEFAULT_COLOR)
            popup_html = (
                f"<b>{index}. {slot.title}</b><br>"
                f"Stop Type: {slot.slot_type.title()}<br>"
                f"Cuisine: {slot.restaurant.cuisine}<br>"
                f"Time: {slot.start_time} - {slot.end_time}<br>"
                f"Cost: {slot.cost:.0f}<br>"
                f"Rating: {slot.restaurant.rating:.1f}/5.0"
            )
            tooltip = f"{index}. {slot.title} ({slot.slot_type})"
        else:
            continue

        if previous_slot is not None and slot.transport_from_previous is not None:
            _add_segment_polyline(
                travel_map=travel_map,
                previous_slot=previous_slot,
                current_slot=slot,
            )

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

        folium.Marker(
            location=point,
            icon=icon,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(travel_map)
        previous_slot = slot

    return travel_map


def _slot_coordinates(slot: TimeSlot) -> list[float] | None:
    """
    Extract coordinates from a timeline slot.

    Args:
        slot: Time slot instance.

    Returns:
        Latitude/longitude list or ``None``.
    """

    if slot.attraction is not None:
        return [slot.attraction.latitude, slot.attraction.longitude]
    if slot.restaurant is not None:
        return [slot.restaurant.latitude, slot.restaurant.longitude]
    return None


def _add_segment_polyline(
    travel_map: folium.Map,
    previous_slot: TimeSlot,
    current_slot: TimeSlot,
) -> None:
    """
    Draw one transport segment on the map.

    Args:
        travel_map: Active Folium map.
        previous_slot: Segment origin slot.
        current_slot: Segment destination slot.
    """

    previous_point = _slot_coordinates(previous_slot)
    current_point = _slot_coordinates(current_slot)
    transport = current_slot.transport_from_previous
    if previous_point is None or current_point is None or transport is None:
        return

    segment_points = (
        decode_polyline(transport.polyline)
        if transport.polyline is not None
        else [previous_point, current_point]
    )
    if len(segment_points) < 2:
        segment_points = [previous_point, current_point]

    dash_array = "8,10" if transport.travel_mode == "walking" else None
    color = SEGMENT_COLORS.get(transport.travel_mode, SEGMENT_COLORS["walking"])
    folium.PolyLine(
        locations=segment_points,
        weight=4,
        opacity=0.85,
        color=color,
        dash_array=dash_array,
        tooltip=transport.navigation_hint,
    ).add_to(travel_map)


def decode_polyline(polyline: str) -> list[list[float]]:
    """
    Decode a Google encoded polyline string into latitude/longitude pairs.

    Args:
        polyline: Encoded polyline string.

    Returns:
        Decoded list of coordinates.
    """

    coordinates: list[list[float]] = []
    latitude = 0
    longitude = 0
    index = 0

    while index < len(polyline):
        latitude_delta, index = _decode_polyline_value(polyline, index)
        longitude_delta, index = _decode_polyline_value(polyline, index)
        latitude += latitude_delta
        longitude += longitude_delta
        coordinates.append([latitude / 1e5, longitude / 1e5])

    return coordinates


def _decode_polyline_value(polyline: str, start_index: int) -> tuple[int, int]:
    """
    Decode one polyline coordinate delta.

    Args:
        polyline: Encoded polyline string.
        start_index: Current read index.

    Returns:
        Decoded integer delta and next read index.
    """

    shift = 0
    result = 0
    index = start_index

    while True:
        value = ord(polyline[index]) - 63
        index += 1
        result |= (value & 0x1F) << shift
        shift += 5
        if value < 0x20:
            break

    decoded_value = ~(result >> 1) if result & 1 else result >> 1
    return decoded_value, index
