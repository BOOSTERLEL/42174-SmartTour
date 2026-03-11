"""
Interactive map page.
"""

import streamlit as st
from streamlit_folium import st_folium

from smarttour.ui.components import build_day_map, select_day

st.title("Trip Map")
planning_session = st.session_state.get("planning_session")

if planning_session is None:
    st.info("Generate an itinerary from the Preferences page first.")
else:
    itinerary = planning_session.itinerary
    selected_index = select_day(itinerary, key="map_day")
    travel_map = build_day_map(itinerary.days[selected_index])
    st_folium(travel_map, width=None, height=520)
