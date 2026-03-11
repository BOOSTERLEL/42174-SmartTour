"""
Timeline rendering helpers.
"""

import streamlit as st

from smarttour.models import DayPlan


def render_day_timeline(day_plan: DayPlan) -> None:
    """
    Render a day plan as a readable timeline.

    Args:
        day_plan: Day plan to visualize.
    """

    st.subheader(day_plan.label)
    for slot in day_plan.slots:
        with st.container(border=True):
            st.markdown(f"**{slot.start_time} - {slot.end_time} | {slot.title}**")
            st.write(slot.description)
            if slot.transport_from_previous is not None:
                minutes = round(slot.transport_from_previous.duration_s / 60)
                st.caption(
                    f"Travel from previous stop: {minutes} min, {slot.transport_from_previous.distance_m} m"
                )
            if slot.attraction is not None:
                st.caption(
                    f"Category: {slot.attraction.category} | Cost: {slot.cost:.0f} | Rating: {slot.attraction.rating:.1f}"
                )
    if day_plan.warnings:
        for warning in day_plan.warnings:
            st.warning(warning)
