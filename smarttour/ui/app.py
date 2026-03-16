"""
Streamlit entrypoint for SmartTour.
"""

from pathlib import Path

import streamlit as st

from smarttour.config import get_settings
from smarttour.logging_config import configure_logging
from smarttour.services.xiaohongshu_client import XiaohongshuClient

PAGES_DIR = Path(__file__).resolve().parent / "pages"


def _initialize_state() -> None:
    """
    Initialize persistent Streamlit session state.
    """

    st.session_state.setdefault("preferences", None)
    st.session_state.setdefault("planning_session", None)
    st.session_state.setdefault("guidance_cache", {})


@st.cache_data(ttl=30, show_spinner=False)
def _xhs_login_status(enabled: bool, chrome_port: int) -> bool:
    """
    Cache the Xiaohongshu login status for a short interval.

    Args:
        enabled: Whether Xiaohongshu integration is enabled.
        chrome_port: Configured Chrome debugging port.

    Returns:
        ``True`` when Xiaohongshu is enabled and logged in.
    """

    _ = chrome_port
    if not enabled:
        return False
    return XiaohongshuClient().check_login_status()


def main() -> None:
    """
    Run the Streamlit application.
    """

    configure_logging()
    settings = get_settings()
    st.set_page_config(page_title=settings.app_name, layout="wide")
    _initialize_state()
    st.sidebar.title(settings.app_name)
    st.sidebar.caption(f"Backend: {settings.backend_url}")
    st.sidebar.write(
        "Use the pages below to parse preferences, generate an itinerary, inspect the route, and export results."
    )
    if settings.xiaohongshu_enabled:
        logged_in = _xhs_login_status(
            settings.xiaohongshu_enabled,
            settings.xiaohongshu_chrome_port,
        )
        st.sidebar.divider()
        st.sidebar.subheader("Xiaohongshu")
        if logged_in:
            st.sidebar.success("Popularity search is enabled and logged in.")
        else:
            st.sidebar.warning(
                "Popularity search is enabled, but Xiaohongshu is not logged in."
            )
            st.sidebar.code(
                "uv run --directory vendor/xiaohongshu-skills python scripts/cli.py --port 19222 check-login\n"
                "# Scan the QR code in Chrome, then run:\n"
                "uv run --directory vendor/xiaohongshu-skills python scripts/cli.py --port 19222 wait-login",
                language="bash",
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
