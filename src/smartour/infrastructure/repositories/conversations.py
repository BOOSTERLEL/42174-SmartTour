"""Conversation repository implementations."""

from smartour.domain.conversation import Conversation


class InMemoryConversationRepository:
    """
    Process-local in-memory conversation repository.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """
        self.conversations: dict[str, Conversation] = {}

    def save(self, conversation: Conversation) -> None:
        """
        Save a conversation.

        Args:
            conversation: The conversation to save.
        """
        self.conversations[conversation.id] = conversation.model_copy(deep=True)

    def get(self, conversation_id: str) -> Conversation | None:
        """
        Return a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The conversation when found.
        """
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            return None
        return conversation.model_copy(deep=True)
