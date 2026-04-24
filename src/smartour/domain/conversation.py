"""Conversation domain models."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from smartour.domain.requirement import TravelRequirement


def _new_id(prefix: str) -> str:
    """
    Generate a prefixed unique identifier.

    Args:
        prefix: The identifier prefix.

    Returns:
        The generated identifier.
    """
    return f"{prefix}_{uuid4().hex}"


def _utc_now() -> datetime:
    """
    Return the current UTC datetime.

    Returns:
        The current UTC datetime.
    """
    return datetime.now(tz=UTC)


class MessageRole(StrEnum):
    """
    Supported conversation message roles.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationState(StrEnum):
    """
    Supported travel planning conversation states.
    """

    COLLECTING_REQUIREMENTS = "collecting_requirements"
    CONFIRMING_REQUIREMENTS = "confirming_requirements"
    PLANNING = "planning"
    READY_FOR_REVIEW = "ready_for_review"
    COMPLETED = "completed"
    FAILED = "failed"


class ConversationMessage(BaseModel):
    """
    A single message in a travel planning conversation.
    """

    id: str = Field(default_factory=lambda: _new_id("msg"))
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=_utc_now)


class Conversation(BaseModel):
    """
    A travel planning conversation with canonical requirement state.
    """

    id: str = Field(default_factory=lambda: _new_id("conv"))
    state: ConversationState = ConversationState.COLLECTING_REQUIREMENTS
    messages: list[ConversationMessage] = Field(default_factory=list)
    requirement: TravelRequirement = Field(default_factory=TravelRequirement)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    def add_message(self, role: MessageRole, content: str) -> ConversationMessage:
        """
        Add a message to the conversation.

        Args:
            role: The message role.
            content: The message content.

        Returns:
            The created conversation message.
        """
        message = ConversationMessage(role=role, content=content)
        self.messages.append(message)
        self.updated_at = _utc_now()
        return message

    def latest_assistant_message(self) -> str | None:
        """
        Return the latest assistant message content.

        Returns:
            The latest assistant message content when available.
        """
        for message in reversed(self.messages):
            if message.role == MessageRole.ASSISTANT:
                return message.content
        return None
