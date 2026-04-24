"""Conversation orchestration service for travel requirement collection."""

from smartour.application.requirement_extractor import RequirementExtractor
from smartour.domain.conversation import (
    Conversation,
    ConversationState,
    MessageRole,
)
from smartour.domain.requirement import TravelRequirement
from smartour.infrastructure.repositories.conversations import (
    InMemoryConversationRepository,
)


class ConversationService:
    """
    Coordinates conversation state, requirement extraction, and assistant replies.
    """

    def __init__(
        self,
        conversation_repository: InMemoryConversationRepository,
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

    def create_conversation(self, initial_message: str | None = None) -> Conversation:
        """
        Create a conversation and optionally process the first user message.

        Args:
            initial_message: An optional first user message.

        Returns:
            The created conversation.
        """
        conversation = Conversation()
        self.conversation_repository.save(conversation)
        if initial_message:
            updated_conversation = self.handle_user_message(
                conversation.id, initial_message
            )
            if updated_conversation is not None:
                return updated_conversation
        assistant_message = self._build_missing_slots_reply(conversation.requirement)
        conversation.add_message(MessageRole.ASSISTANT, assistant_message)
        self.conversation_repository.save(conversation)
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Return a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The conversation when found.
        """
        return self.conversation_repository.get(conversation_id)

    def handle_user_message(
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
        conversation = self.conversation_repository.get(conversation_id)
        if conversation is None:
            return None
        conversation.add_message(MessageRole.USER, message)
        updates = self.requirement_extractor.extract(message)
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
        self.conversation_repository.save(conversation)
        return conversation

    def confirm_requirements(self, conversation_id: str) -> Conversation | None:
        """
        Confirm the current requirement snapshot.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The updated conversation when found.
        """
        conversation = self.conversation_repository.get(conversation_id)
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
        self.conversation_repository.save(conversation)
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
        missing_text = ", ".join(missing_slots[:2])
        return (
            f"I still need {missing_text}. Please provide those details so I can "
            "plan the trip."
        )

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
