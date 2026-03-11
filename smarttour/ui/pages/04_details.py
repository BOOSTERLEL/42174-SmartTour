"""
Attraction details and guidance page.
"""

import httpx
import streamlit as st

from smarttour.models import GuidanceRequest
from smarttour.ui.api_client import APIClient

client = APIClient()

st.title("Attraction Details")
planning_session = st.session_state.get("planning_session")

if planning_session is None:
    st.info("Generate an itinerary from the Preferences page first.")
else:
    itinerary = planning_session.itinerary
    attractions = [
        slot.attraction
        for day in itinerary.days
        for slot in day.slots
        if slot.attraction is not None
    ]
    labels = {attraction.id: attraction.name for attraction in attractions}
    selected_id = st.selectbox(
        "Attraction", options=list(labels), format_func=lambda item: labels[item]
    )
    attraction = next(item for item in attractions if item.id == selected_id)
    st.subheader(attraction.name)
    st.write(attraction.description)
    st.caption(f"Category: {attraction.category} | Rating: {attraction.rating:.1f}")
    if st.button("Generate Guidance"):
        try:
            guidance = client.explain_attraction(
                GuidanceRequest(
                    attraction_id=selected_id,
                    preferences=itinerary.preferences,
                )
            )
            st.session_state["guidance_cache"][selected_id] = guidance
        except httpx.HTTPError as error:
            st.error(f"Failed to generate guidance: {error}")
    guidance = st.session_state["guidance_cache"].get(selected_id)
    if guidance is not None:
        st.markdown(f"**Background**\n\n{guidance.historical_background}")
        st.markdown("**Tips**")
        for item in guidance.visiting_tips:
            st.write(f"- {item}")
        st.markdown("**Practical Notes**")
        for item in guidance.practical_notes:
            st.write(f"- {item}")
