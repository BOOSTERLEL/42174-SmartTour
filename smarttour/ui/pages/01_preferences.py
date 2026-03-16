"""
Preference input page.
"""

import httpx
import streamlit as st

from smarttour.models import (
    ItineraryGenerateRequest,
    PreferenceParseRequest,
    TravelPreferences,
    UserInput,
)
from smarttour.ui.api_client import APIClient
from smarttour.ui.components import render_preference_card

client = APIClient()

st.title("Travel Preferences")
st.write(
    "Describe the trip in natural language, then review or edit the structured result."
)

with st.form("preference_parse_form"):
    user_prompt = st.text_area(
        "Trip request",
        value=st.session_state.get(
            "user_prompt",
            "I want to visit Melbourne for 3 days with a $500 budget and lots of food and museums.",
        ),
        height=160,
    )
    parse_submitted = st.form_submit_button("Parse Preferences")

if parse_submitted:
    try:
        response = client.parse_preferences(
            PreferenceParseRequest(user_input=UserInput(text=user_prompt))
        )
        st.session_state["preferences"] = response.preferences
        st.session_state["user_prompt"] = user_prompt
        for warning in response.warnings:
            st.warning(warning)
    except httpx.HTTPError as error:
        st.error(f"Failed to parse preferences: {error}")

preferences = st.session_state.get("preferences")
if preferences is not None:
    render_preference_card(preferences)
    with st.form("preference_edit_form"):
        destination = st.text_input("Destination", value=preferences.destination)
        trip_days = st.number_input(
            "Trip days", min_value=1, max_value=14, value=preferences.trip_days
        )
        budget_total = st.number_input(
            "Budget", min_value=0.0, value=float(preferences.budget_total), step=50.0
        )
        interests_text = st.text_input(
            "Interests", value=", ".join(preferences.interests)
        )
        travel_mode = st.selectbox(
            "Travel mode",
            ["walking_transit", "walking", "transit", "driving"],
            index=["walking_transit", "walking", "transit", "driving"].index(
                preferences.travel_mode
            ),
            format_func=lambda item: {
                "walking_transit": "Walking + Transit",
                "walking": "Walking",
                "transit": "Transit",
                "driving": "Driving",
            }[item],
        )
        pace = st.selectbox(
            "Pace",
            ["relaxed", "balanced", "packed"],
            index=["relaxed", "balanced", "packed"].index(preferences.pace),
        )
        update_submitted = st.form_submit_button("Save Preferences")
    if update_submitted:
        st.session_state["preferences"] = TravelPreferences(
            destination=destination,
            trip_days=int(trip_days),
            budget_total=float(budget_total),
            interests=[
                item.strip() for item in interests_text.split(",") if item.strip()
            ],
            travel_mode=travel_mode,
            pace=pace,
            travelers=preferences.travelers,
            accessibility_needs=preferences.accessibility_needs,
            preferred_start_hour=preferences.preferred_start_hour,
            preferred_end_hour=preferences.preferred_end_hour,
            origin_summary=preferences.origin_summary,
        )
        st.success("Preferences updated.")
    if st.button("Generate Itinerary", type="primary"):
        try:
            planning_session = client.generate_itinerary(
                ItineraryGenerateRequest(
                    preferences=st.session_state["preferences"],
                    user_input=st.session_state.get("user_prompt", ""),
                )
            )
            st.session_state["planning_session"] = planning_session
            st.success("Itinerary generated. Open the Itinerary page.")
        except httpx.HTTPError as error:
            st.error(f"Failed to generate itinerary: {error}")
