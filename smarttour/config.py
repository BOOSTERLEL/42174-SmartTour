"""
Application settings for SmartTour.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datas"


class Settings(BaseSettings):
    """
    Runtime settings loaded from environment variables and `.env`.
    """

    app_name: str = "SmartTour"
    app_version: str = "0.1.0"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    google_maps_api_key: str | None = None
    database_url: str = f"sqlite:///{(DATA_DIR / 'smarttour.db').as_posix()}"
    backend_url: str = "http://localhost:8000"
    log_level: str = "INFO"
    default_destination: str = "Melbourne"
    default_trip_days: int = 3
    max_attractions_per_day: int = 4
    default_start_hour: int = 9
    default_end_hour: int = 18
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached settings instance.

    Returns:
        The application settings.
    """

    return Settings()


def reset_settings_cache() -> None:
    """
    Clear cached settings for tests or environment changes.
    """

    get_settings.cache_clear()
