"""
Preference summary rendering.
"""

import streamlit as st

from smarttour.models import TravelPreferences


def render_preference_card(preferences: TravelPreferences) -> None:
    """
    Render a compact summary of structured preferences.

    Args:
        preferences: Structured preferences to display.
    """

    st.subheader("Structured Preferences")
    left_column, middle_column, right_column = st.columns(3)
    left_column.metric("Destination", preferences.destination)
    middle_column.metric("Trip Days", str(preferences.trip_days))
    right_column.metric("Budget", f"{preferences.budget_total:.0f}")
    st.write(
        {
            "interests": preferences.interests,
            "travel_mode": preferences.travel_mode,
            "pace": preferences.pace,
            "travelers": preferences.travelers,
            "accessibility": preferences.accessibility_needs,
            "time_window": f"{preferences.preferred_start_hour:02d}:00-{preferences.preferred_end_hour:02d}:00",
        }
    )
