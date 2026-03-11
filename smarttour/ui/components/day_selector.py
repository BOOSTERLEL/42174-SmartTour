"""
Day selection helper.
"""

import streamlit as st

from smarttour.models import Itinerary


def select_day(itinerary: Itinerary, key: str = "selected_day") -> int:
    """
    Render a day selector and return the chosen index.

    Args:
        itinerary: Current itinerary.
        key: Session state key for the selector.

    Returns:
        The selected day index.
    """

    labels = [day.label for day in itinerary.days]
    if not labels:
        return 0
    selected_label = st.selectbox("Select day", labels, key=key)
    return labels.index(selected_label)
