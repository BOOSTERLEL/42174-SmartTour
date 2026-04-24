"""Application configuration for Smartour."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Runtime settings loaded from environment variables and `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_maps_api_key: str = Field(default="", validation_alias="GOOGLE_MAPS_API_KEY")
    google_maps_timeout_seconds: float = Field(
        default=10.0, validation_alias="GOOGLE_MAPS_TIMEOUT_SECONDS"
    )
    sqlite_path: str = Field(
        default="data/smartour.sqlite3", validation_alias="SMARTOUR_SQLITE_PATH"
    )
    google_maps_cache_ttl_seconds: int = Field(
        default=86400, validation_alias="GOOGLE_MAPS_CACHE_TTL_SECONDS"
    )
    google_maps_routes_cache_ttl_seconds: int = Field(
        default=0, validation_alias="GOOGLE_MAPS_ROUTES_CACHE_TTL_SECONDS"
    )
    itinerary_job_rate_limit_count: int = Field(
        default=5, validation_alias="ITINERARY_JOB_RATE_LIMIT_COUNT"
    )
    itinerary_job_rate_limit_window_seconds: int = Field(
        default=3600, validation_alias="ITINERARY_JOB_RATE_LIMIT_WINDOW_SECONDS"
    )
    openai_api_baseurl: str | None = Field(
        default=None, validation_alias="OPENAI_API_BASEURL"
    )
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_api_model: str | None = Field(
        default=None, validation_alias="OPENAI_API_MODEL"
    )

    @model_validator(mode="after")
    def validate_google_maps_api_key(self) -> Self:
        """
        Validate that the Google Maps API key is configured.

        Returns:
            The validated settings model.
        """
        if not self.google_maps_api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY is required")
        return self

    def has_openai_config(self) -> bool:
        """
        Return whether OpenAI API settings are configured.

        Returns:
            True when the API key and model are available.
        """
        return bool(self.openai_api_key and self.openai_api_model)
