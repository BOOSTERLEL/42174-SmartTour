"""Application configuration for Smartour."""

from typing import Self

from pydantic import AliasChoices, Field, model_validator
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
    google_maps_places_unit_cost_usd: float = Field(
        default=0.0, validation_alias="GOOGLE_MAPS_PLACES_UNIT_COST_USD"
    )
    google_maps_routes_unit_cost_usd: float = Field(
        default=0.0, validation_alias="GOOGLE_MAPS_ROUTES_UNIT_COST_USD"
    )
    google_maps_geocoding_unit_cost_usd: float = Field(
        default=0.0, validation_alias="GOOGLE_MAPS_GEOCODING_UNIT_COST_USD"
    )
    google_maps_timezone_unit_cost_usd: float = Field(
        default=0.0, validation_alias="GOOGLE_MAPS_TIMEZONE_UNIT_COST_USD"
    )
    requirement_model_path: str | None = Field(
        default=None, validation_alias="REQUIREMENT_MODEL_PATH"
    )
    requirement_model_confidence_threshold: float = Field(
        default=0.35, validation_alias="REQUIREMENT_MODEL_CONFIDENCE_THRESHOLD"
    )
    requirement_model_development_fallback_enabled: bool = Field(
        default=True,
        validation_alias="REQUIREMENT_MODEL_DEVELOPMENT_FALLBACK_ENABLED",
    )
    itinerary_job_conversation_rate_limit_count: int = Field(
        default=5,
        validation_alias=AliasChoices(
            "ITINERARY_JOB_CONVERSATION_RATE_LIMIT_COUNT",
            "ITINERARY_JOB_RATE_LIMIT_COUNT",
        ),
    )
    itinerary_job_ip_rate_limit_count: int = Field(
        default=50, validation_alias="ITINERARY_JOB_IP_RATE_LIMIT_COUNT"
    )
    itinerary_job_rate_limit_window_seconds: int = Field(
        default=3600, validation_alias="ITINERARY_JOB_RATE_LIMIT_WINDOW_SECONDS"
    )
    admin_usernames: str = Field(
        default="admin", validation_alias="SMARTOUR_ADMIN_USERNAMES"
    )
    session_ttl_seconds: int = Field(
        default=604800, validation_alias="SMARTOUR_SESSION_TTL_SECONDS"
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
