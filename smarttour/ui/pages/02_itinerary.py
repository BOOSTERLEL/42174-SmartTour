"""
Itinerary visualization page.
"""

import streamlit as st

from smarttour.ui.components import render_day_timeline, select_day

st.title("Itinerary")
planning_session = st.session_state.get("planning_session")

if planning_session is None:
    st.info("Generate an itinerary from the Preferences page first.")
else:
    itinerary = planning_session.itinerary
    left_column, middle_column, right_column = st.columns(3)
    left_column.metric("Destination", itinerary.destination)
    middle_column.metric("Days", str(len(itinerary.days)))
    right_column.metric("Estimated Cost", f"{itinerary.total_estimated_cost:.0f}")
    if itinerary.accommodation is not None:
        st.caption(
            f"Suggested stay: {itinerary.accommodation.name} | {itinerary.accommodation.price_per_night:.0f} per night"
        )
    selected_index = select_day(itinerary, key="timeline_day")
    render_day_timeline(itinerary.days[selected_index])
    if itinerary.warnings:
        for warning in itinerary.warnings:
            st.warning(warning)
