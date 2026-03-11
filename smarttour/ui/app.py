"""
Streamlit entrypoint for SmartTour.
"""

from pathlib import Path

import streamlit as st

from smarttour.config import get_settings

PAGES_DIR = Path(__file__).resolve().parent / "pages"


def _initialize_state() -> None:
    """
    Initialize persistent Streamlit session state.
    """

    st.session_state.setdefault("preferences", None)
    st.session_state.setdefault("planning_session", None)
    st.session_state.setdefault("guidance_cache", {})


def main() -> None:
    """
    Run the Streamlit application.
    """

    settings = get_settings()
    st.set_page_config(page_title=settings.app_name, layout="wide")
    _initialize_state()
    st.sidebar.title(settings.app_name)
    st.sidebar.caption(f"Backend: {settings.backend_url}")
    st.sidebar.write(
        "Use the pages below to parse preferences, generate an itinerary, inspect the route, and export results."
    )
    pages = [
        st.Page(str(PAGES_DIR / "01_preferences.py"), title="Preferences"),
        st.Page(str(PAGES_DIR / "02_itinerary.py"), title="Itinerary"),
        st.Page(str(PAGES_DIR / "03_map.py"), title="Map"),
        st.Page(str(PAGES_DIR / "04_details.py"), title="Details"),
        st.Page(str(PAGES_DIR / "05_export.py"), title="Export"),
    ]
    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
