"""Conversation orchestration service for travel requirement collection."""

import re
from typing import Any

from smartour.application.requirement_extractor import RequirementExtractor
from smartour.domain.conversation import (
    Conversation,
    ConversationState,
    MessageRole,
)
from smartour.domain.requirement import TravelRequirement, TravelRequirementUpdate

MISSING_SLOT_GUIDANCE = {
    "destination": "destination, such as Tokyo, Paris, or Sydney",
    "trip_dates_or_length": (
        "travel dates or trip length, such as 2026-07-01 to 2026-07-05 or 5 days"
    ),
    "travelers": "traveler count, such as 2 adults or 2 adults and 1 child",
    "budget_level": "budget level: low, medium, or high",
    "travel_pace": "travel pace: relaxed, balanced, or packed",
    "interests": (
        "main interests, such as food, museums, history, nature, shopping, "
        "nightlife, or family"
    ),
    "hotel_area": (
        "preferred hotel area, such as city center, downtown, near public "
        "transit, near the waterfront, old town, or a specific neighborhood"
    ),
    "transportation_mode": "primary transportation mode: transit, walking, or drive",
}
HOTEL_AREA_ALIASES = {
    "city center": "city center",
    "city centre": "city center",
    "downtown": "downtown",
    "near public transit": "near public transit",
    "near transit": "near public transit",
    "public transit": "near public transit",
    "near the waterfront": "near the waterfront",
    "near waterfront": "near the waterfront",
    "waterfront": "near the waterfront",
    "old town": "old town",
    "cbd": "CBD",
    "central business district": "CBD",
}
TRANSPORTATION_ALIASES = {
    "transit": "transit",
    "public transport": "transit",
    "public transportation": "transit",
    "metro": "transit",
    "subway": "transit",
    "walking": "walking",
    "walk": "walking",
    "walkable": "walking",
    "drive": "drive",
    "driving": "drive",
    "car": "drive",
    "rental car": "drive",
}
BUDGET_ALIASES = {
    "low": "low",
    "cheap": "low",
    "budget": "low",
    "budget-friendly": "low",
    "medium": "medium",
    "moderate": "medium",
    "mid-range": "medium",
    "high": "high",
    "luxury": "high",
    "premium": "high",
}
PACE_ALIASES = {
    "relaxed": "relaxed",
    "slow": "relaxed",
    "balanced": "balanced",
    "normal": "balanced",
    "packed": "packed",
    "intensive": "packed",
}
SHORT_HOTEL_AREA_BLOCKLIST = (
    set(TRANSPORTATION_ALIASES) | set(BUDGET_ALIASES) | set(PACE_ALIASES)
)


class ConversationService:
    """
    Coordinates conversation state, requirement extraction, and assistant replies.
    """

    def __init__(
        self,
        conversation_repository: Any,
        requirement_extractor: RequirementExtractor,
    ) -> None:
        """
        Initialize the conversation service.

        Args:
            conversation_repository: The repository used to persist conversations.
            requirement_extractor: The component used to extract requirement updates.
        """
        self.conversation_repository = conversation_repository
        self.requirement_extractor = requirement_extractor

    async def create_conversation(
        self, initial_message: str | None = None
    ) -> Conversation:
        """
        Create a conversation and optionally process the first user message.

        Args:
            initial_message: An optional first user message.

        Returns:
            The created conversation.
        """
        conversation = Conversation()
        await self.conversation_repository.save(conversation)
        if initial_message:
            updated_conversation = await self.handle_user_message(
                conversation.id, initial_message
            )
            if updated_conversation is not None:
                return updated_conversation
        assistant_message = self._build_missing_slots_reply(conversation.requirement)
        conversation.add_message(MessageRole.ASSISTANT, assistant_message)
        await self.conversation_repository.save(conversation)
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Return a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The conversation when found.
        """
        return await self.conversation_repository.get(conversation_id)

    async def handle_user_message(
        self, conversation_id: str, message: str
    ) -> Conversation | None:
        """
        Process a user message and update conversation requirements.

        Args:
            conversation_id: The conversation ID.
            message: The raw user message.

        Returns:
            The updated conversation when found.
        """
        conversation = await self.conversation_repository.get(conversation_id)
        if conversation is None:
            return None
        conversation.add_message(MessageRole.USER, message)
        missing_slots = conversation.requirement.missing_required_slots()
        updates = self.requirement_extractor.extract(message)
        updates = self._apply_contextual_missing_slot_updates(
            missing_slots, updates, message
        )
        conversation.requirement = conversation.requirement.merge(updates)
        if conversation.requirement.missing_required_slots():
            conversation.state = ConversationState.COLLECTING_REQUIREMENTS
            assistant_message = self._build_missing_slots_reply(
                conversation.requirement
            )
        else:
            conversation.state = ConversationState.CONFIRMING_REQUIREMENTS
            assistant_message = self._build_confirmation_reply(conversation.requirement)
        conversation.add_message(MessageRole.ASSISTANT, assistant_message)
        await self.conversation_repository.save(conversation)
        return conversation

    def _apply_contextual_missing_slot_updates(
        self,
        missing_slots: list[str],
        updates: TravelRequirementUpdate,
        message: str,
    ) -> TravelRequirementUpdate:
        """
        Interpret short replies as answers to the slots just requested.

        Args:
            missing_slots: The slots missing before this user message.
            updates: The model or rule-based extraction result.
            message: The raw user message.

        Returns:
            The extraction result with contextual slot answers filled in.
        """
        values = updates.model_dump()
        if "hotel_area" in missing_slots and updates.hotel_area is None:
            hotel_area = self._extract_contextual_hotel_area(message)
            if hotel_area is not None:
                values["hotel_area"] = hotel_area
        if (
            "transportation_mode" in missing_slots
            and updates.transportation_mode is None
        ):
            transportation_mode = self._extract_contextual_alias(
                message, TRANSPORTATION_ALIASES
            )
            if transportation_mode is not None:
                values["transportation_mode"] = transportation_mode
        if "budget_level" in missing_slots and updates.budget_level is None:
            budget_level = self._extract_contextual_alias(message, BUDGET_ALIASES)
            if budget_level is not None:
                values["budget_level"] = budget_level
        if "travel_pace" in missing_slots and updates.travel_pace is None:
            travel_pace = self._extract_contextual_alias(message, PACE_ALIASES)
            if travel_pace is not None:
                values["travel_pace"] = travel_pace
        return TravelRequirementUpdate.model_validate(values)

    def _extract_contextual_hotel_area(self, message: str) -> str | None:
        """
        Extract a hotel-area answer when the conversation just asked for it.

        Args:
            message: The raw user message.

        Returns:
            The preferred hotel area when the reply looks like one.
        """
        cleaned_message = self._clean_contextual_answer(message)
        lower_message = cleaned_message.lower()
        if not lower_message:
            return None
        for alias, value in HOTEL_AREA_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower_message):
                return value
        hotel_area_patterns = [
            r"(?:preferred\s+)?hotel\s+area\s+(?:is\s+)?(.+)$",
            r"(?:stay|staying)\s+(?:near|in|around)\s+(.+)$",
            r"hotel\s+(?:near|in|around)\s+(.+)$",
            r"(?:a|an)\s+(.+?)\s+hotel\s+area\b",
        ]
        for pattern in hotel_area_patterns:
            match = re.search(pattern, cleaned_message, flags=re.IGNORECASE)
            if match:
                return self._clean_contextual_answer(match.group(1))
        if self._can_use_short_answer_as_hotel_area(lower_message):
            return cleaned_message
        return None

    def _extract_contextual_alias(
        self, message: str, aliases: dict[str, str]
    ) -> str | None:
        """
        Extract a canonical value from a short option answer.

        Args:
            message: The raw user message.
            aliases: Supported aliases mapped to canonical values.

        Returns:
            The canonical option value when detected.
        """
        cleaned_message = self._clean_contextual_answer(message)
        lower_message = cleaned_message.lower()
        for alias, value in aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower_message):
                return value
        return None

    def _can_use_short_answer_as_hotel_area(self, lower_message: str) -> bool:
        """
        Return whether a short reply can safely fill the hotel area.

        Args:
            lower_message: The lowercase cleaned user message.

        Returns:
            True when the reply looks like a place or area answer.
        """
        if lower_message in SHORT_HOTEL_AREA_BLOCKLIST:
            return False
        if re.search(
            r"\d|adult|child|kid|days?|budget|pace|transport|transit|walk|drive|car",
            lower_message,
        ):
            return False
        return len(lower_message.split()) <= 4

    def _clean_contextual_answer(self, message: str) -> str:
        """
        Clean a short contextual answer.

        Args:
            message: The raw user message.

        Returns:
            The cleaned answer text.
        """
        return re.sub(r"\s+", " ", message.strip(" ,.;:")).strip()

    async def confirm_requirements(self, conversation_id: str) -> Conversation | None:
        """
        Confirm the current requirement snapshot.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The updated conversation when found.
        """
        conversation = await self.conversation_repository.get(conversation_id)
        if conversation is None:
            return None
        missing_slots = conversation.requirement.missing_required_slots()
        if missing_slots:
            conversation.state = ConversationState.COLLECTING_REQUIREMENTS
            assistant_message = self._build_missing_slots_reply(
                conversation.requirement
            )
        else:
            conversation.state = ConversationState.PLANNING
            assistant_message = (
                "Requirements confirmed. Itinerary generation can start."
            )
        conversation.add_message(MessageRole.ASSISTANT, assistant_message)
        await self.conversation_repository.save(conversation)
        return conversation

    def _build_missing_slots_reply(self, requirement: TravelRequirement) -> str:
        """
        Build an assistant reply for missing required requirement slots.

        Args:
            requirement: The current requirement snapshot.

        Returns:
            A concise assistant reply.
        """
        missing_slots = requirement.missing_required_slots()
        missing_text = self._format_missing_slot_guidance(missing_slots[:2])
        return (
            f"I still need your {missing_text}. You can reply with one of these "
            "options or use your own wording so I can plan the trip."
        )

    def _format_missing_slot_guidance(self, missing_slots: list[str]) -> str:
        """
        Build readable guidance for the next missing requirement slots.

        Args:
            missing_slots: The internal missing slot names.

        Returns:
            The user-facing slot guidance text.
        """
        guidance_items = [
            MISSING_SLOT_GUIDANCE.get(missing_slot, missing_slot)
            for missing_slot in missing_slots
        ]
        if len(guidance_items) <= 1:
            return "".join(guidance_items)
        return "; and your ".join(guidance_items)

    def _build_confirmation_reply(self, requirement: TravelRequirement) -> str:
        """
        Build a confirmation reply for a complete requirement snapshot.

        Args:
            requirement: The completed requirement snapshot.

        Returns:
            A concise confirmation reply.
        """
        trip_timing = requirement.trip_length_days or requirement.trip_dates
        interests_text = ", ".join(requirement.interests)
        return (
            "I have enough details to plan the trip: "
            f"{requirement.destination}, {trip_timing}, "
            f"{requirement.travelers.adults} traveler(s), {requirement.budget_level}, "
            f"{requirement.travel_pace} pace, interests in {interests_text}. "
            "Please confirm before I generate the itinerary."
        )
