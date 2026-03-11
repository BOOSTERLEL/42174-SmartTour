"""
Export page for itinerary output.
"""

import json

import streamlit as st


def _build_markdown(planning_session) -> str:
    """
    Convert the planning session into Markdown text.

    Args:
        planning_session: The saved planning session.

    Returns:
        Markdown export content.
    """

    itinerary = planning_session.itinerary
    lines = [f"# {itinerary.destination} Itinerary", ""]
    for day in itinerary.days:
        lines.append(f"## {day.label}")
        for slot in day.slots:
            lines.append(f"- {slot.start_time}-{slot.end_time}: {slot.title}")
        lines.append("")
    return "\n".join(lines)


st.title("Export")
planning_session = st.session_state.get("planning_session")

if planning_session is None:
    st.info("Generate an itinerary from the Preferences page first.")
else:
    markdown_output = _build_markdown(planning_session)
    json_output = json.dumps(planning_session.model_dump(mode="json"), indent=2)
    st.download_button(
        "Download Markdown", markdown_output, file_name="smarttour-itinerary.md"
    )
    st.download_button(
        "Download JSON", json_output, file_name="smarttour-itinerary.json"
    )
    st.code(markdown_output, language="markdown")
