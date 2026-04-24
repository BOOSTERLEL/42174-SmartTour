"""Conversation API routes for travel requirement collection."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from smartour.api.dependencies import get_conversation_service
from smartour.application.conversation_service import ConversationService
from smartour.domain.conversation import Conversation, ConversationState
from smartour.domain.requirement import TravelRequirement

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    """
    Request model for creating a conversation.
    """

    initial_message: str | None = None


class SendMessageRequest(BaseModel):
    """
    Request model for sending a user message.
    """

    message: str


class ConversationResponse(BaseModel):
    """
    Response model for conversation state.
    """

    conversation_id: str
    state: ConversationState
    assistant_message: str | None
    requirement_snapshot: TravelRequirement
    missing_required_slots: list[str]


@router.post(
    "", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    request: CreateConversationRequest,
    conversation_service: Annotated[
        ConversationService, Depends(get_conversation_service)
    ],
) -> ConversationResponse:
    """
    Create a new travel planning conversation.

    Args:
        request: The conversation creation request.
        conversation_service: The conversation service.

    Returns:
        The new conversation state.
    """
    conversation = await conversation_service.create_conversation(
        request.initial_message
    )
    return _conversation_response(conversation)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    conversation_service: Annotated[
        ConversationService, Depends(get_conversation_service)
    ],
) -> ConversationResponse:
    """
    Return a conversation by ID.

    Args:
        conversation_id: The conversation ID.
        conversation_service: The conversation service.

    Returns:
        The current conversation state.
    """
    conversation = await conversation_service.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return _conversation_response(conversation)


@router.post("/{conversation_id}/messages", response_model=ConversationResponse)
async def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    conversation_service: Annotated[
        ConversationService, Depends(get_conversation_service)
    ],
) -> ConversationResponse:
    """
    Send a user message to a conversation.

    Args:
        conversation_id: The conversation ID.
        request: The user message request.
        conversation_service: The conversation service.

    Returns:
        The updated conversation state.
    """
    conversation = await conversation_service.handle_user_message(
        conversation_id, request.message
    )
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return _conversation_response(conversation)


@router.post("/{conversation_id}/confirm", response_model=ConversationResponse)
async def confirm_conversation(
    conversation_id: str,
    conversation_service: Annotated[
        ConversationService, Depends(get_conversation_service)
    ],
) -> ConversationResponse:
    """
    Confirm collected requirements for a conversation.

    Args:
        conversation_id: The conversation ID.
        conversation_service: The conversation service.

    Returns:
        The updated conversation state.
    """
    conversation = await conversation_service.confirm_requirements(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return _conversation_response(conversation)


def _conversation_response(conversation: Conversation) -> ConversationResponse:
    """
    Convert a domain conversation to an API response.

    Args:
        conversation: The domain conversation.

    Returns:
        The API conversation response.
    """
    return ConversationResponse(
        conversation_id=conversation.id,
        state=conversation.state,
        assistant_message=conversation.latest_assistant_message(),
        requirement_snapshot=conversation.requirement,
        missing_required_slots=conversation.requirement.missing_required_slots(),
    )
