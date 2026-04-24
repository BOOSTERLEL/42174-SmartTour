"""OpenAI SDK connectivity probe for Smartour."""

import json

from smartour.core.config import Settings
from smartour.core.errors import ExternalServiceError
from smartour.integrations.openai.requirement_extractor import (
    OpenAIRequirementExtractor,
)

SAMPLE_MESSAGE = (
    "I want to visit Tokyo for 4 days with 2 adults, medium budget, "
    "relaxed pace, food and museums, stay near Shinjuku, use transit."
)


def main() -> None:
    """
    Run a sanitized OpenAI requirement extraction probe.
    """
    settings = Settings()
    if not settings.has_openai_config():
        raise SystemExit("OPENAI_API_KEY and OPENAI_API_MODEL are required")
    extractor = OpenAIRequirementExtractor(
        api_key=settings.openai_api_key or "",
        model=settings.openai_api_model or "",
        base_url=settings.openai_api_baseurl,
    )
    try:
        requirement_update = extractor.extract(SAMPLE_MESSAGE)
    except ExternalServiceError as error:
        raise SystemExit(f"OpenAI probe failed: {error}") from None
    print(
        json.dumps(
            {
                "status": "ok",
                "openai_base_url_configured": bool(settings.openai_api_baseurl),
                "extracted": requirement_update.model_dump(exclude_none=True),
            },
            sort_keys=True,
        )
    )
