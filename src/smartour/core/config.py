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
