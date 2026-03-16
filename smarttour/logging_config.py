"""
Shared logging configuration for API and Streamlit entrypoints.
"""

from __future__ import annotations

import logging
import logging.config
import os

from smarttour.config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """
    Configure application logging using a single console handler.

    Args:
        settings: Optional settings override.
    """

    active_settings = settings or get_settings()
    log_level = str(active_settings.log_level or "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", active_settings.log_format).strip().lower()
    formatter_name = "json" if log_format == "json" else "text"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "text": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                },
                "json": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": formatter_name,
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"],
            },
        }
    )
    logging.getLogger().setLevel(log_level)
