"""OpenAI-backed travel requirement extraction."""

from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from smartour.core.errors import ExternalServiceError
from smartour.domain.requirement import Travelers, TravelRequirementUpdate

BudgetLevel = Literal["low", "medium", "high"]
TravelPace = Literal["relaxed", "balanced", "packed"]
TransportationMode = Literal["walking", "transit", "drive"]

SYSTEM_PROMPT = """
Extract travel planning requirements from the user's message.
Return null for scalar fields that are not explicitly present.
Return empty arrays for list fields that are not explicitly present.
Do not infer missing details.
Normalize budget_level to low, medium, or high.
Normalize travel_pace to relaxed, balanced, or packed.
Normalize transportation_mode to walking, transit, or drive.
Use concise city, region, neighborhood, and interest names.
Use ISO 639-1 language codes only when the user explicitly requests a language.
""".strip()


class OpenAIRequirementExtraction(BaseModel):
    """
    Structured output schema for OpenAI travel requirement extraction.
    """

    destination: str | None
    trip_dates: str | None
    trip_length_days: int | None = Field(ge=1)
    adults: int | None = Field(ge=1)
    children: int | None = Field(ge=0)
    budget_level: BudgetLevel | None
    travel_pace: TravelPace | None
    interests: list[str]
    hotel_area: str | None
    transportation_mode: TransportationMode | None
    food_preferences: list[str]
    language: str | None

    def to_requirement_update(self) -> TravelRequirementUpdate:
        """
        Convert the structured extraction to a domain requirement update.

        Returns:
            The travel requirement update.
        """
        travelers = None
        if self.adults is not None or self.children is not None:
            travelers = Travelers(adults=self.adults, children=self.children or 0)
        return TravelRequirementUpdate(
            destination=self.destination,
            trip_dates=self.trip_dates,
            trip_length_days=self.trip_length_days,
            travelers=travelers,
            budget_level=self.budget_level,
            travel_pace=self.travel_pace,
            interests=self.interests,
            hotel_area=self.hotel_area,
            transportation_mode=self.transportation_mode,
            food_preferences=self.food_preferences,
            language=self.language,
        )


class OpenAIRequirementExtractor:
    """
    Extracts travel requirement updates using the official OpenAI Python SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        """
        Initialize the OpenAI requirement extractor.

        Args:
            api_key: The OpenAI API key.
            model: The OpenAI model name.
            base_url: An optional OpenAI-compatible API base URL.
            client: An optional SDK client used by tests.
        """
        self.model = model
        self.client = client or self._create_client(api_key, base_url)

    def extract(self, message: str) -> TravelRequirementUpdate:
        """
        Extract requirement updates from a user message.

        Args:
            message: The raw user message.

        Returns:
            The extracted requirement updates.

        Raises:
            ExternalServiceError: Raised when both SDK parse paths fail.
        """
        try:
            extraction = self._extract_with_responses_api(message)
        except Exception:
            try:
                extraction = self._extract_with_chat_completions_api(message)
            except Exception:
                raise ExternalServiceError(
                    "openai", "OpenAI requirement extraction failed"
                ) from None
            if not extraction:
                raise ExternalServiceError(
                    "openai", "OpenAI requirement extraction returned no data"
                ) from None
        return extraction.to_requirement_update()

    def _create_client(self, api_key: str, base_url: str | None) -> OpenAI:
        """
        Create an official OpenAI SDK client.

        Args:
            api_key: The OpenAI API key.
            base_url: An optional OpenAI-compatible API base URL.

        Returns:
            The configured OpenAI SDK client.
        """
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)

    def _extract_with_responses_api(self, message: str) -> OpenAIRequirementExtraction:
        """
        Extract requirements with the Responses API parse helper.

        Args:
            message: The raw user message.

        Returns:
            The parsed structured extraction.
        """
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            text_format=OpenAIRequirementExtraction,
        )
        return self._extract_from_responses_result(response)

    def _extract_with_chat_completions_api(
        self, message: str
    ) -> OpenAIRequirementExtraction:
        """
        Extract requirements with the Chat Completions parse helper.

        Args:
            message: The raw user message.

        Returns:
            The parsed structured extraction.
        """
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            response_format=OpenAIRequirementExtraction,
        )
        parsed_message = completion.choices[0].message
        refusal = getattr(parsed_message, "refusal", None)
        if refusal:
            raise ExternalServiceError(
                "openai", "OpenAI refused requirement extraction"
            )
        return self._coerce_extraction(getattr(parsed_message, "parsed", None))

    def _extract_from_responses_result(
        self, response: Any
    ) -> OpenAIRequirementExtraction:
        """
        Extract a parsed schema object from a Responses API result.

        Args:
            response: The SDK response object.

        Returns:
            The parsed structured extraction.
        """
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return self._coerce_extraction(parsed)
        for output in getattr(response, "output", []):
            if getattr(output, "type", None) != "message":
                continue
            for item in getattr(output, "content", []):
                if getattr(item, "type", None) == "refusal":
                    raise ExternalServiceError(
                        "openai", "OpenAI refused requirement extraction"
                    )
                parsed_item = getattr(item, "parsed", None)
                if parsed_item is not None:
                    return self._coerce_extraction(parsed_item)
        raise ExternalServiceError("openai", "OpenAI returned no parsed content")

    def _coerce_extraction(self, value: Any) -> OpenAIRequirementExtraction:
        """
        Coerce a parsed SDK value into the extraction schema.

        Args:
            value: The SDK parsed value.

        Returns:
            The validated structured extraction.
        """
        if isinstance(value, OpenAIRequirementExtraction):
            return value
        if value is None:
            raise ExternalServiceError("openai", "OpenAI returned no parsed content")
        return OpenAIRequirementExtraction.model_validate(value)


class HybridRequirementExtractor:
    """
    Uses OpenAI extraction first and falls back to another extractor on failure.
    """

    def __init__(
        self,
        primary_extractor: Any,
        fallback_extractor: Any,
    ) -> None:
        """
        Initialize the hybrid requirement extractor.

        Args:
            primary_extractor: The OpenAI-backed extractor.
            fallback_extractor: The fallback extractor.
        """
        self.primary_extractor = primary_extractor
        self.fallback_extractor = fallback_extractor

    def extract(self, message: str) -> TravelRequirementUpdate:
        """
        Extract requirement updates with a resilient fallback.

        Args:
            message: The raw user message.

        Returns:
            The extracted requirement updates.
        """
        try:
            return self.primary_extractor.extract(message)
        except Exception:
            return self.fallback_extractor.extract(message)
